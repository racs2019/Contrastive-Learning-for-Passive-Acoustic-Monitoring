# === Vanilla SimCLR (reef acoustics) — PAM-MATCHED CROPS + SAME AUGS ==========

import os, math, random, time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import torchaudio as ta
from typing import Optional, Tuple, Dict

# ----------------------------
# Speed / determinism
# ----------------------------
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# ----------------------------
# Backbone: ResNet18 (1-channel)
# ----------------------------
from torchvision.models import resnet18

def _resnet18_1ch():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Identity()
    return m

# ----------------------------
# Encoder + projector (Vanilla SimCLR)
# ----------------------------
class EncoderProj(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = _resnet18_1ch()
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.projector(h), dim=1)
        return z

# ----------------------------
# NT-Xent (Vanilla)
# ----------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)           # (2B, D)
        logits = (z @ z.T) / self.tau            # (2B, 2B)
        logits.fill_diagonal_(-1e9)
        target = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        return F.cross_entropy(logits, target)

# ----------------------------
# Augmentations (EXACTLY from PAM code)
# ----------------------------
class SpectroAug(nn.Module):
    def __init__(
        self,
        time_mask_param=32,
        freq_mask_param=12,
        p_time_mask=0.8,
        p_freq_mask=0.8,
        p_notch=0.3, notch_width=6,
        p_shift=0.8,
        p_noise=0.5,
        max_shift_frac=0.12,
        noise_std=0.01,
        p_time_crop=0.5,
        crop_frac_range=(0.7, 1.0),
        keep_length=True,
    ):
        super().__init__()
        self.tm = ta.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.fm = ta.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_notch = p_notch
        self.notch_width = notch_width
        self.p_shift = p_shift
        self.p_noise = p_noise
        self.max_shift_frac = max_shift_frac
        self.noise_std = noise_std
        self.p_time_crop = p_time_crop
        self.crop_lo, self.crop_hi = crop_frac_range
        self.keep_length = keep_length

    def forward(self, x):  # x: (B,1,F,T) OR (1,F,T) depending on callsite
        # This module is used on GPU in training loop on (B,1,F,T).
        # We implement it batch-safe.
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B, C, Fdim, T0 = x.shape

        # time crop (per-sample)
        if torch.rand(1, device=x.device) < self.p_time_crop and T0 > 16:
            # same crop frac for whole batch is fine, but we can do per-sample crop start
            frac = float(torch.empty(1, device=x.device).uniform_(self.crop_lo, self.crop_hi))
            new_T = max(16, int(T0 * frac))
            if new_T < T0:
                starts = torch.randint(0, T0 - new_T + 1, (B,), device=x.device)
                x_new = torch.zeros((B, C, Fdim, new_T), device=x.device, dtype=x.dtype)
                for i in range(B):
                    s = int(starts[i].item())
                    x_new[i] = x[i, :, :, s:s + new_T]
                x = x_new

        # time/freq masks (these operate per example internally)
        if torch.rand(1, device=x.device) < self.p_time_mask:
            x = self.tm(x)
        if torch.rand(1, device=x.device) < self.p_freq_mask:
            x = self.fm(x)

        # spectral notch
        if torch.rand(1, device=x.device) < self.p_notch:
            width = min(self.notch_width, Fdim)
            start = int(torch.randint(0, max(1, Fdim - width + 1), (1,), device=x.device))
            x[:, :, start:start + width, :] = 0.0

        # time shift
        if torch.rand(1, device=x.device) < self.p_shift:
            max_shift = max(1, int(self.max_shift_frac * x.size(-1)))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device))
            x = torch.roll(x, shifts=shift, dims=-1)

        # noise
        if torch.rand(1, device=x.device) < self.p_noise:
            x = x + self.noise_std * torch.randn_like(x)

        # keep original length
        if self.keep_length and x.size(-1) != T0:
            pad = T0 - x.size(-1)
            if pad > 0:
                x = F.pad(x, (0, pad), mode="constant", value=0.0)
            else:
                x = x[:, :, :, :T0]

        x = x.contiguous()
        if squeeze_back:
            x = x.squeeze(0)
        return x

# ----------------------------
# Dataset: .npy segments + PAM-matched center/jitter cropping
# ----------------------------
class VanillaPAMCropNPYDataset(Dataset):
    """
    Loads .npy segments (float32 wave, typically length 32000 @ 16kHz).
    Crop mechanics matches PAM:
      - center computed from BASE mel: (64,1024,512)
      - crops taken from VIEW mel (single config for vanilla): view_mel_choice
      - crop_T uses center + jitter
      - returns n_global views (2) as (1,F,T) tensors (CPU)
    """
    def __init__(
        self,
        audio_dir: str,
        sr: int = 16000,
        n_global: int = 2,
        global_T: int = 256,
        event_jitter_T: int = 24,
        per_sample_norm: bool = True,
        resize_to: Tuple[int, int] = (128, 256),
        # PAM base mel for center detection:
        base_mel: Tuple[int, int, int] = (64, 1024, 512),
        # Vanilla single mel for actual views:
        view_mel: Tuple[int, int, int] = (128, 2048, 256),
        # Robustness
        log_bad_files: bool = True,
        bad_log_path: str = "bad_npy.txt",
        max_retries: int = 8,
    ):
        self.audio_paths = sorted(
            [
                os.path.join(audio_dir, f)
                for f in os.listdir(audio_dir)
                if f.endswith(".npy") and f != "labels.npy" and f != "labels_original.npy"
            ]
        )
        if len(self.audio_paths) == 0:
            raise RuntimeError(f"No .npy segments found in: {audio_dir}")

        self.sr = sr
        self.n_global = n_global
        self.global_T = global_T
        self.event_jitter_T = event_jitter_T
        self.per_sample_norm = per_sample_norm
        self.resize_to = resize_to

        self.base_mel = base_mel
        self.view_mel = view_mel

        self.log_bad_files = log_bad_files
        self.bad_log_path = bad_log_path
        self.max_retries = max_retries

        # Mel cache (like PAM, but just two configs)
        self._mel_cache: Dict[Tuple[int, int, int], ta.transforms.MelSpectrogram] = {}
        for n_mels, n_fft, hop in set([self.base_mel, self.view_mel]):
            self._mel_cache[(n_mels, n_fft, hop)] = ta.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            )

    def __len__(self):
        return len(self.audio_paths)

    def _try_log_bad(self, path, err):
        if not self.log_bad_files:
            return
        try:
            with open(self.bad_log_path, "a", encoding="utf-8") as f:
                f.write(f"{path}\t{repr(err)}\n")
        except Exception:
            pass

    def _wave_to_logmel(self, y_np: np.ndarray, mel_choice: Tuple[int, int, int]) -> torch.Tensor:
        n_mels, n_fft, hop = mel_choice
        y = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        mx = float(np.max(np.abs(y)))
        mx = 1.0 if (not np.isfinite(mx) or mx < 1e-8) else mx
        y = (y / mx).astype(np.float32, copy=False)

        y_t = torch.from_numpy(y).unsqueeze(0)  # (1, T)
        mel = self._mel_cache[(n_mels, n_fft, hop)](y_t)  # (1, n_mels, Tm)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        logmel = torch.nan_to_num(logmel, nan=0.0, posinf=0.0, neginf=0.0)

        if self.per_sample_norm:
            m, s = logmel.mean(), logmel.std()
            if (not torch.isfinite(s)) or (s < 1e-6):
                s = torch.tensor(1e-6, device=logmel.device)
            logmel = (logmel - m) / s
        return logmel  # (1, F, T)

    def _energy_center(self, logmel: torch.Tensor) -> int:
        energy = logmel.mean(dim=1).squeeze(0)  # (T,)
        return int(torch.argmax(energy).item())

    def _crop_T(self, x: torch.Tensor, T_win: int, center: Optional[int], jitter_T: int) -> torch.Tensor:
        T = x.size(2)
        if T <= T_win:
            return x
        if center is None:
            start = int(torch.randint(0, T - T_win + 1, (1,)))
        else:
            start = max(0, min(center - T_win // 2, T - T_win))
            j = int(torch.randint(-jitter_T, jitter_T + 1, (1,)))
            start = max(0, min(start + j, T - T_win))
        return x[:, :, start:start + T_win]

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize_to is None:
            return x
        Ft, Tt = self.resize_to
        x4 = x.unsqueeze(0)  # (1,1,F,T)
        x4 = F.interpolate(x4, size=(Ft, Tt), mode="bilinear", align_corners=False)
        return x4.squeeze(0)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            path = self.audio_paths[idx]
            try:
                y = np.load(path, allow_pickle=False).astype(np.float32, copy=False)  # (T,)
                y = np.asarray(y, dtype=np.float32).flatten()

                # center on BASE mel (exactly like PAM)
                base = self._wave_to_logmel(y, self.base_mel)
                center = self._energy_center(base)

                # build global views using SINGLE view mel
                views = []
                for _ in range(self.n_global):
                    lm = self._wave_to_logmel(y, self.view_mel)
                    crop = self._crop_T(lm, self.global_T, center=center, jitter_T=self.event_jitter_T)
                    crop = self._maybe_resize(crop)
                    views.append(crop.contiguous())
                return views

            except Exception as e:
                self._try_log_bad(path, e)
                idx = random.randint(0, len(self.audio_paths) - 1)

        raise RuntimeError(f"Failed to load npy after {self.max_retries} retries. See {self.bad_log_path}.")

# ----------------------------
# Collate
# ----------------------------
def multicrop_collate(batch):
    n_views = len(batch[0])
    return [torch.stack([b[v] for b in batch], 0) for v in range(n_views)]

# ----------------------------
# Scheduler (match PAM warmup+cosine)
# ----------------------------
def warmup_cosine_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * t))

def pick_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ----------------------------
# Train (Vanilla SimCLR, but PAM-matched crops+augs+schedule)
# ----------------------------
def train_simclr_vanilla_pamfair(
    audio_dir,
    epochs=200,
    batch_size=256,
    lr=3e-4,
    weight_decay=1e-4,
    temperature=0.1,
    num_workers=4,
    seed=42,
    train_fraction=1.0,
    # optimizer mechanics parity
    accum_steps=2,
    warmup_epochs=10,
    clip_grad_norm=1.0,
    # saving/logging
    save_path="vanilla_simclr_pamfair.pth",
    save_every=5,
    # dataset knobs (match PAM cropping)
    sr=16000,
    global_T=256,
    event_jitter_T=24,
    resize_to=(128, 256),
    base_mel=(64, 1024, 512),
    view_mel=(128, 2048, 256),  # keep single mel for vanilla
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp_dtype(device)
    use_amp = (device.type == "cuda") and (amp_dtype is not None)

    print(f"Device: {device} | AMP dtype: {amp_dtype}", flush=True)

    # Dataset
    full_ds = VanillaPAMCropNPYDataset(
        audio_dir=audio_dir,
        sr=sr,
        n_global=2,
        global_T=global_T,
        event_jitter_T=event_jitter_T,
        per_sample_norm=True,
        resize_to=resize_to,
        base_mel=base_mel,
        view_mel=view_mel,
    )

    if 0 < train_fraction < 1.0:
        n_total = len(full_ds)
        n_sub = max(1, int(train_fraction * n_total))
        train_ds, _ = random_split(full_ds, [n_sub, n_total - n_sub], generator=torch.Generator().manual_seed(seed))
        print(f"Training on subset: {n_sub}/{n_total} ({train_fraction:.2%})", flush=True)
    else:
        train_ds = full_ds
        print(f"Training on full dataset: {len(train_ds)} samples", flush=True)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin,
        collate_fn=multicrop_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Model/loss/opt
    model = EncoderProj(proj_dim=128).to(device).to(memory_format=torch.channels_last)
    loss_fn = NTXentLoss(temperature=temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (match PAM style)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, accum_steps)))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_lambda(step, warmup_steps, total_steps))

    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # Augmentations (PAM aug)
    aug = SpectroAug(keep_length=True).to(device)

    base, ext = os.path.splitext(save_path)
    ext = ext if ext else ".pth"

    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for it, views in enumerate(train_loader):
            v1 = views[0].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            v2 = views[1].to(device, non_blocking=True).to(memory_format=torch.channels_last)

            # Apply same aug op as PAM
            v1 = aug(v1)
            v2 = aug(v2)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)
                loss = loss_fn(z1, z2)
                loss = loss / max(1, accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.detach().cpu()) * max(1, accum_steps)

            # optimizer step on accumulation boundary
            if (it + 1) % max(1, accum_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if clip_grad_norm and clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}] loss={running/len(train_loader):.4f} | lr={cur_lr:.2e} | time={epoch_time:.1f}s",
              flush=True)

        if (epoch % save_every == 0) or (epoch == epochs):
            ckpt_path = f"{base}_epoch{epoch:03d}{ext}"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), f"{base}_latest{ext}")

    print("Training complete.", flush=True)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    train_simclr_vanilla_pamfair(
        audio_dir=r"C:\Users\Documents\SimClr\train_preprocessed",  # <-- folder with .npy segments
        epochs=50,
        batch_size=256,
        lr=3e-4,
        weight_decay=1e-4,
        temperature=0.1,
        accum_steps=2,
        warmup_epochs=10,
        num_workers=8,
        save_path="vanilla_simclr.pth",
        train_fraction=1,   # set 1.0 for full run
        save_every=5,
        # crop settings aligned with PAM
        sr=10000,
        global_T=256,
        event_jitter_T=24,
        resize_to=(128, 256),
        base_mel=(64, 1024, 512),
        view_mel=(128, 2048, 256),
    )


