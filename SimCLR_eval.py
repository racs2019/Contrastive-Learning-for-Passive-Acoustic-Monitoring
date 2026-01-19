# ===== Unsupervised Report (NO UMAP) — Vanilla SimCLR checkpoints + 10kHz data =====

import os
import math, time, contextlib
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict, Any, Literal

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------------
# Import model definition (EncoderProj) from training code
# ------------------------------------------------------------------------------------
import sys
from pathlib import Path

try:
    simclr_dir = Path(r"C:\Users\Documents\SimClr") #adjust to SimClr.py
    if str(simclr_dir) not in sys.path:
        sys.path.insert(0, str(simclr_dir))

    from vanilla_simclrv2 import EncoderProj  # noqa
    print("[INFO] Imported EncoderProj from vanilla_simclrv2.")
except Exception as e:
    raise RuntimeError(f"Failed to import EncoderProj. Check sys.path/module name. Error: {e}")

# ------------------------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------------------------
def hungarian_accuracy(y_true, y_pred) -> float:
    """
    Maximum bipartite matching accuracy between predicted cluster IDs and true labels.
    Works with strings too.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_vals = np.unique(y_true)
    pred_vals = np.unique(y_pred)

    true_ids = {t: i for i, t in enumerate(true_vals)}
    pred_ids = {c: i for i, c in enumerate(pred_vals)}

    y_true_i = np.vectorize(true_ids.get)(y_true)
    y_pred_i = np.vectorize(pred_ids.get)(y_pred)

    W = np.zeros((len(pred_vals), len(true_vals)), dtype=np.int64)
    for i in range(y_pred_i.size):
        W[y_pred_i[i], y_true_i[i]] += 1

    row_ind, col_ind = linear_sum_assignment(W.max() - W)  # maximize matches
    return float(W[row_ind, col_ind].sum() / max(1, y_pred_i.size))

# ------------------------------------------------------------------------------------
# Label loading
# ------------------------------------------------------------------------------------
def list_audio_npy_files(folder: str):
    files = []
    for f in os.listdir(folder):
        if f.endswith(".npy") and f not in ("labels.npy", "labels_original.npy"):
            stem = os.path.splitext(f)[0]
            try:
                key = int(stem)
            except ValueError:
                key = stem
            files.append((key, os.path.join(folder, f)))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]

def load_labels_mapping_if_any(folder: str) -> Optional[Dict[str, Any]]:
    """
    Supports:
      - labels.npy as 0-D object array containing dict {stem_key: label} (your case)
      - labels.npy as 1-D aligned list (legacy)
    Returns dict mapping key -> label.
    """
    lbl_path = os.path.join(folder, "labels.npy")
    if not os.path.isfile(lbl_path):
        return None

    try:
        arr = np.load(lbl_path, allow_pickle=True)

        # Case A: scalar object containing dict
        if getattr(arr, "shape", None) == () and isinstance(arr.item(), dict):
            d = arr.item()
            out: Dict[str, Any] = {}
            for k, v in d.items():
                ks = os.path.splitext(str(k))[0]  # allow keys with/without .npy
                out[ks] = v
            return out

        # Case B: 1-D array aligned with sorted files
        if hasattr(arr, "ndim") and arr.ndim >= 1:
            labels = np.asarray(arr, dtype=object).reshape(-1)
            files = list_audio_npy_files(folder)
            stems = [os.path.splitext(os.path.basename(p))[0] for p in files]
            return {stems[i]: labels[i] for i in range(min(len(stems), len(labels)))}

    except Exception as e:
        print(f"[WARN] labels.npy exists but failed to load: {e}")
        return None

    return None

def label_for_file(labels_map: Optional[Dict[str, Any]], fp: str) -> Any:
    """
    Map file path -> label using dict keys like '0_000'.
    If file stem is '0', we also try '0_000'.
    """
    if labels_map is None:
        return "Unknown"
    stem = os.path.splitext(os.path.basename(fp))[0]

    if stem in labels_map:
        return labels_map[stem]

    if stem.isdigit():
        k = f"{int(stem)}_000"
        return labels_map.get(k, "Unknown")

    return "Unknown"

# ------------------------------------------------------------------------------------
# Vanilla-training-matched waveform->logmel center crop + resize
# ------------------------------------------------------------------------------------

def _wave_to_spec_1xFxT(
    y_np: np.ndarray,
    sr: int = 10000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512
) -> torch.Tensor:
    y_np = np.nan_to_num(y_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mx = np.max(np.abs(y_np))
    mx = 1.0 if (not np.isfinite(mx) or mx < 1e-8) else mx
    y_t = torch.from_numpy(y_np / mx).unsqueeze(0)  # (1, T)

    try:
        import torchaudio as ta
        mel = ta.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        S = mel(y_t)                        # (1, F, T)
        S = torch.clamp(S, min=1e-10).log()
        return S.contiguous().float()
    except Exception:
        # fallback if torchaudio missing
        from scipy.signal import spectrogram, get_window
        win = get_window("hann", n_fft, fftbins=True)
        f, t, Z = spectrogram(
            y_np, fs=sr, window=win, nperseg=n_fft,
            noverlap=n_fft - hop_length, mode="magnitude"
        )
        S = np.log(np.maximum(Z, 1e-10)).astype(np.float32)  # (F_lin, T)
        if S.shape[0] != n_mels:
            idx = np.linspace(0, S.shape[0] - 1, num=n_mels).astype(np.int32)
            S = S[idx]
        S = torch.from_numpy(S).unsqueeze(0)  # (1, F, T)
        return S.contiguous().float()


class TwoViewPrecomputed(Dataset):
    """
    OLD REPORT PROTOCOL:
    - If waveform .npy: convert to (1, 64, Tm) log-mel with (n_fft=1024, hop=512)
    - If spectrogram .npy: accept it as (1,F,T)
    - Per-sample z-norm
    - Returns two identical views (x1, x2)
    """
    def __init__(
        self,
        spec_dir: str,
        T: int = 256,  # kept for signature parity
        per_sample_norm: bool = True,
        sr: int = 10000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop: int = 512,
        wave_policy: str = "convert",   # "convert" or "skip"
        file_paths=None,
    ):
        if file_paths is not None:
            self.paths = list(file_paths)
        else:
            self.paths = list_audio_npy_files(spec_dir)

        self.T = T
        self.per_sample_norm = per_sample_norm
        self.sr, self.n_mels, self.n_fft, self.hop = sr, n_mels, n_fft, hop
        self.wave_policy = wave_policy

        if self.wave_policy == "skip":
            kept = []
            for p in self.paths:
                arr = np.load(p, mmap_mode="r")
                if arr.ndim == 1:
                    continue
                kept.append(p)
            self.paths = kept

    def __len__(self):
        return len(self.paths)

    def _to_1xFxT(self, arr: np.ndarray) -> torch.Tensor:
        arr = np.array(arr, copy=True)
        if arr.ndim == 1:
            if self.wave_policy == "skip":
                raise RuntimeError("Waveform encountered but wave_policy='skip'.")
            return _wave_to_spec_1xFxT(
                arr, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop
            )
        x = torch.from_numpy(arr)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1,F,T)
        elif x.ndim == 3:
            if x.shape[0] != 1:
                # best-effort: keep first channel
                x = x[:1]
        else:
            raise RuntimeError(f"Unexpected array shape {tuple(x.shape)}")
        return x.contiguous().float()

    def __getitem__(self, idx):
        feat_np = np.load(self.paths[idx], mmap_mode="r")
        feat = self._to_1xFxT(feat_np)

        if self.per_sample_norm:
            m, s = feat.mean(), feat.std()
            if (not torch.isfinite(s)) or (s < 1e-6):
                s = torch.tensor(1e-6, device=feat.device)
            feat = (feat - m) / s

        x1 = feat
        x2 = feat.clone()
        return x1, x2


# ------------------------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------------------------
def choose_clusterer(algorithm: str, embeddings: np.ndarray, n_clusters: int):
    if algorithm == "kmeans":
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            n_init="auto",
            max_iter=200,
            random_state=42,
        ).fit(embeddings)
        return clusterer.labels_
    elif algorithm == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
        return clusterer.labels_
    elif algorithm == "gmm":
        clusterer = GaussianMixture(
            n_components=n_clusters, covariance_type="full", random_state=42
        ).fit(embeddings)
        return clusterer.predict(embeddings)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

# ------------------------------------------------------------------------------------
# Encoder loading
# ------------------------------------------------------------------------------------
def _torch_load_safe(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def _load_encoder(encoder_ckpt_path: str, device: torch.device, proj_dim: int = 128):
    enc = EncoderProj(proj_dim=proj_dim).to(device).to(memory_format=torch.channels_last)

    ckpt = _torch_load_safe(encoder_ckpt_path, device)

    if isinstance(ckpt, dict) and "online" in ckpt:
        state_dict = ckpt["online"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Checkpoint format unexpected. Expected dict/state_dict.")

    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("encoder.", "").replace("module.", "")
        new_state[new_key] = v

    missing, unexpected = enc.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys (showing up to 10): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 10): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    enc.eval()
    print(f"[INFO] Encoder loaded on {device} • CUDA: {torch.cuda.is_available()}")
    return enc

# ------------------------------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------------------------------
def extract_embeddings_from_specdir(
    encoder_ckpt_path: str,
    spec_dir: str,
    *,
    batch_size: int = 128,
    n_workers: int = 0,
    device_str: Optional[str] = None,
    subset_fraction: float = 1.0,
    subset_seed: int = 42,
    subset_strategy: Literal["random", "tail", "head"] = "random",
    wave_policy: str = "convert",
    proj_dim: int = 128,
    repr_mode: Literal["backbone", "proj"] = "backbone",
):
    all_files = list_audio_npy_files(spec_dir)
    if len(all_files) == 0:
        raise RuntimeError(f"No .npy files found in {spec_dir}")

    n_sub = max(1, int(math.ceil(len(all_files) * subset_fraction)))
    if subset_strategy == "random":
        rng = np.random.RandomState(subset_seed)
        chosen_idx = np.sort(rng.choice(len(all_files), size=n_sub, replace=False))
    elif subset_strategy == "tail":
        chosen_idx = np.arange(len(all_files) - n_sub, len(all_files))
    elif subset_strategy == "head":
        chosen_idx = np.arange(0, n_sub)
    else:
        raise ValueError(f"Unknown subset_strategy: {subset_strategy}")

    chosen_files = [all_files[i] for i in chosen_idx]

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

    enc = _load_encoder(encoder_ckpt_path, device, proj_dim=proj_dim)

    ds = TwoViewPrecomputed(
        spec_dir,
        T=256,
        per_sample_norm=True,
        sr=10000,
        n_mels=64,
        n_fft=1024,
        hop=512,
        wave_policy=wave_policy,
        file_paths=chosen_files,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda" and n_workers == 0),
        persistent_workers=False,
    )

    feats = []
    use_cuda = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else contextlib.nullcontext

    t0 = time.perf_counter()
    n_seen = 0
    print(f"[INFO] Embedding pass: {len(ds)} items • repr_mode={repr_mode} • device={device}")

    with torch.inference_mode():
        with autocast_ctx():
            for step, (x1, _) in enumerate(loader):
                n_seen += x1.size(0)
                x = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                if repr_mode == "backbone":
                    h = enc.backbone(x)   # (B,512)
                elif repr_mode == "proj":
                    h = enc(x)            # (B,proj_dim)
                else:
                    raise ValueError("repr_mode must be 'backbone' or 'proj'")

                if h.ndim > 2:
                    h = torch.flatten(h, start_dim=1)

                feats.append(h.detach().cpu().float())

                if use_cuda:
                    torch.cuda.synchronize()
                if (step == 0) or ((step + 1) % 25 == 0):
                    dt = time.perf_counter() - t0
                    print(f"[INFO] step={step+1} • seen={n_seen} • {n_seen/max(dt,1e-9):.1f} items/s", flush=True)

    H = torch.cat(feats, dim=0).numpy().astype("float32")
    print(f"[INFO] Done embeddings: N={H.shape[0]} D={H.shape[1]} in {time.perf_counter()-t0:.2f}s")
    return H, chosen_files

# ------------------------------------------------------------------------------------
# Run metrics (NO UMAP, NO HTML)
# ------------------------------------------------------------------------------------
def run_metrics_only(
    *,
    encoder_ckpt_path: str,
    dataset_paths: list[str],
    labels_list: list[str],
    cluster_method: Literal["kmeans", "agglomerative", "gmm"] = "kmeans",
    n_clusters: int = 60,
    subset_fraction: float = 1.0,
    subset_seed: int = 42,
    subset_strategy: Literal["random", "tail", "head"] = "random",
    wave_policy: str = "convert",
    proj_dim: int = 128,
    repr_mode: Literal["backbone", "proj"] = "backbone",
):
    assert len(dataset_paths) == len(labels_list), "dataset_paths and labels_list must match length"

    embeddings_all = []
    class_labels_all = []
    location_labels_all = []

    for path, loc_label in zip(dataset_paths, labels_list):
        H, chosen_files = extract_embeddings_from_specdir(
            encoder_ckpt_path,
            path,
            batch_size=128,
            n_workers=0,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
            subset_strategy=subset_strategy,
            wave_policy=wave_policy,
            proj_dim=proj_dim,
            repr_mode=repr_mode,
        )
        embeddings_all.append(H)
        location_labels_all.extend([loc_label] * H.shape[0])

        labels_map = load_labels_mapping_if_any(path)
        class_labels_all.extend([label_for_file(labels_map, p) for p in chosen_files])

    embeddings = np.vstack(embeddings_all).astype(np.float32)
    class_labels = np.asarray(class_labels_all, dtype=object)
    location_labels = np.asarray(location_labels_all, dtype=object)

    # Cluster labels
    print(f"[INFO] Clustering: method={cluster_method} k={n_clusters} on N={embeddings.shape[0]} D={embeddings.shape[1]}")
    cluster_labels = choose_clusterer(cluster_method, embeddings, n_clusters)

    # Validity checks (sklearn metrics need >=2 clusters and N>k)
    uniq = np.unique(cluster_labels)
    valid_internal = (len(uniq) > 1) and (embeddings.shape[0] > len(uniq))

    def safe_internal(fn):
        try:
            return float(fn(embeddings, cluster_labels)) if valid_internal else float("nan")
        except Exception:
            return float("nan")

    sil = safe_internal(silhouette_score)           # ↑
    dbi = safe_internal(davies_bouldin_score)       # ↓
    ch  = safe_internal(calinski_harabasz_score)    # ↑

    # External vs labels
    def safe_external(fn):
        try:
            return float(fn(class_labels, cluster_labels)) if valid_internal else float("nan")
        except Exception:
            return float("nan")

    ari  = safe_external(adjusted_rand_score)
    ami  = safe_external(adjusted_mutual_info_score)
    hacc = safe_external(hungarian_accuracy)

    print("\n========== METRICS ==========")
    print(f"repr_mode: {repr_mode}  (backbone=512d, proj={proj_dim}d)")
    print(f"clusters : {cluster_method}  k={n_clusters}")
    print(f"N        : {embeddings.shape[0]}")
    print(f"D        : {embeddings.shape[1]}")
    print("")
    print(f"Silhouette (↑): {sil}")
    print(f"Davies–Bouldin (↓): {dbi}")
    print(f"Calinski–Harabasz (↑): {ch}")
    print(f"ARI (↑): {ari}")
    print(f"AMI (↑): {ami}")
    print(f"Hungarian Accuracy (↑): {hacc}")
    print("=============================\n")

    return {
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
        "ari": ari,
        "ami": ami,
        "hungarian_accuracy": hacc,
        "n": int(embeddings.shape[0]),
        "d": int(embeddings.shape[1]),
        "k": int(n_clusters),
        "repr_mode": repr_mode,
        "cluster_method": cluster_method,
    }

# ------------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    encoder_ckpt_path = r"C:\Users\Documents\SimClr\vanilla_simclr_latest.pth"
    dataset_paths = [
        r"C:\Users\Documents\SimClr\test_preprocessed",
    ]
    labels_list = [
        "Test",
    ]

    run_metrics_only(
        encoder_ckpt_path=encoder_ckpt_path,
        dataset_paths=dataset_paths,
        labels_list=labels_list,
        cluster_method="kmeans",
        n_clusters=60,
        subset_fraction=1.0,
        subset_seed=45,
        subset_strategy="random",
        wave_policy="convert",
        proj_dim=128,
        repr_mode="backbone",
    )
