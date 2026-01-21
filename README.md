# 🪸 Contrastive Learning for Passive Acoustic Monitoring (PAM)

> **A scalable, unsupervised framework for sound source discovery and cross-site comparison in marine soundscapes**

This repository implements the framework introduced in  
**“Contrastive Learning for Passive Acoustic Monitoring: A Framework for Sound Sources Discovery and Cross-Site Comparison in Marine Soundscapes”**  
by *Richard Acs, Ali Ibrahim, Hanqi Zhuang, and Laurent M. Chérubin.*

This repository is intended as a **research companion** to the manuscript. It provides
reference implementations of all models, preprocessing, and evaluation pipelines used
in the study, enabling transparency, methodological inspection, and reproduction of the
reported analyses given access to compatible passive acoustic monitoring (PAM) datasets.

---

## Overview

Marine Passive Acoustic Monitoring (PAM) enables long-term, non-invasive biodiversity
observation but faces key challenges, including overlapping sources, high noise, and
limited labeled data. This framework introduces a **domain-adapted contrastive learning
pipeline** tailored for noisy marine environments, enabling **unsupervised organization
of acoustic recordings**, discovery of recurring acoustic patterns, and comparison across
sites and years.

The approach produces embeddings with **strong intrinsic cluster structure** and
supports exploratory analysis of biological and anthropogenic sound sources, while
requiring **no manual annotations**.

---

## Key Features

- **Unsupervised discovery** of recurring biological and anthropogenic acoustic patterns  
- **Teacher-guided multi-positive SimCLR** with local–global invariance and VICReg regularization  
- **Acoustically informed augmentations** for underwater environments  
- **Cross-site comparison** across Caribbean reef monitoring sites and years  
- **Contrastive embeddings** supporting clustering, acoustic signature discovery,
  and exploratory ecological interpretation  

---

## Framework Overview

<img width="1935" height="1382" alt="Framework overview" src="https://github.com/user-attachments/assets/cace26fb-5970-471f-b4cd-202d0c40f2f5" />

### Architecture
- **Encoder:** ResNet-18 (1-channel log-Mel spectrogram input)  
- **Projection Head:** 512 → 256 → 128 (BN–ReLU–ℓ2)  
- **Predictor (SimSiam-style):** 128 → 256 → 128  
- **Teacher Model:** EMA (momentum 0.99 → 0.9995)  
- **Feature Bank:** FIFO queue (8,192 embeddings)

### Objective

The total loss combines contrastive alignment, local–global invariance, and variance
regularization:

$$
L_{\text{total}} = \alpha L_{\text{ctr}} + \beta L_{\text{siam}} + \gamma L_{\text{vic}}
$$

with weights **α = 1.0, β = 0.1, γ = 0.1**.

---

## Dataset

Recordings were collected from **seven core Caribbean spawning aggregation sites
(2017–2024)**, with additional sites included during downstream clustering analysis:

| Location | Dominant Species | # Recordings |
|--------|------------------|--------------|
| Puerto Rico (ALS, ALS Deep, BDS, Mona Elbow) | Red Hind, Nassau Grouper | 260k+ |
| St. Thomas (RHB, GB) | Red Hind, Yellowfin Grouper | 140k+ |
| Mexico (Xcalak, Punta Allen, San Juan) | Nassau Grouper, Toadfish | 100k+ |

Each 20 s clip is segmented into 2 s windows, downsampled to **10 kHz**, and converted
into **log-Mel spectrograms** (typically **128 Mel bands**, 0–5 kHz).

---

## Preprocessing & Augmentations

- Mono conversion and amplitude normalization to [−1, 1]  
- **Event-centric cropping:** 2 global (256 frames) + 2 local (96 frames) views  
- **Augmentations:**  
  - Time/frequency masking (SpecAugment)  
  - Spectral notch dropout  
  - Temporal shift/truncation (≤ 12%)  
  - Gaussian noise (σ = 0.01)

---

## Clustering & Evaluation

| Algorithm | Parameters | Notes |
|---------|------------|------|
| **K-Means** | k = 6 or k = 60 | Fast, interpretable |
| **GMM** | Full covariance | Used for acoustic signature discovery |
| **DBSCAN / HDBSCAN** | Adaptive | Often returns few clusters |
| **Spectral Clustering** | 10-NN graph | Moderate performance |

### Evaluation Metrics
- **External:** ARI, AMI, Hungarian Accuracy  
- **Internal:** Silhouette, Davies–Bouldin Index (DBI), Calinski–Harabasz (CH)

---

## Results Summary

| Method | ARI | AMI | Hungarian | Silhouette | DBI | CH |
|------|-----|-----|-----------|------------|-----|----|
| GTCC + MFCC | 0.089 | 0.140 | 0.333 | 0.114 | 2.02 | 422 |
| CNN-SupCon | **0.372** | **0.396** | **0.646** | 0.302 | 1.18 | 29,187 |
| **PAM-SimCLR (Ours)** | 0.070 | 0.121 | 0.317 | **0.220** | **1.28** | **16,200** |

> PAM-SimCLR exhibits the strongest **intrinsic cluster structure** among unsupervised
methods evaluated, while CNN-SupCon maximizes agreement with coarse labels.

<img width="2207" height="910" alt="UMAP results" src="https://github.com/user-attachments/assets/8442fb23-b98c-4f68-a1d2-da331cc677cd" />

---

## Acoustic Signature Discovery

The framework identifies **recurring acoustic patterns** across sites:

| Cluster | Sites | Frequency (Hz) | Notes |
|--------|------|----------------|------|
| 6 | RHB | 200–800 | Red hind pulse trains |
| 14 | BDS | 200–600 | Marine mammal calls |
| 16 | Xcalak, Punta Allen | 0–600 | Toadfish harmonics |
| 19 | Multi-site | 0–100 | Vessel noise |
| 33 | Multi-site | 0–800 | Narrowband biotic pulses |

<img width="2015" height="1385" alt="Acoustic signatures" src="https://github.com/user-attachments/assets/2e28e33d-c6af-4130-ba6d-0158c9c68f6f" />

---

## Implementation

**Environment**
- Python **3.11**
- PyTorch **2.1**, torchaudio **2.1**
- scikit-learn **1.3**
- GPU tested: **NVIDIA RTX A6000 (48 GB)**

**Training**
- Optimizer: AdamW (3e−4, weight decay 1e−4)
- Warmup 10 epochs → cosine decay
- Batch size 256 (grad accumulation = 2)
- Mixed precision (FP16/BF16)

---

## Repository Structure

Jupyter notebooks are used intentionally to integrate model definition, training,
evaluation, and visualization in a single auditable workflow reflecting how analyses
were performed during the study.

```text
GTCC + MFCC          GTCC+MFCC.ipynb
Log-Mel + PCA        Spectrogram + PCA.ipynb
CNN Latent Space     FADAR_Embeddings.ipynb
CNN-SupCon           CNN SupCon.ipynb
VAE + GMM            VAE+GMM.ipynb
PAM-SimCLR           SimCLR(main).ipynb
Vanilla SimCLR       Vanilla_SimCLR.py
Evaluation           SimCLR_eval.py
```

## Reproducing Results from the Paper

| Paper Component | Code Reference |
|----------------|---------------|
| PAM-SimCLR training | `SimCLR(main).ipynb` |
| Vanilla SimCLR baseline | `Vanilla_SimCLR.py` |
| CNN-SupCon baseline | `CNN SupCon.ipynb` |
| Quantitative metrics (Tables 5–6) | `SimCLR_eval.py` |
| UMAP visualizations (Figs. 5, 7) | Report cells in notebooks |
| Acoustic signature discovery | GMM clustering + report generation |

---

## Data Availability

Training and test sets, as well as representative  subsets of the passive
acoustic datasets used in this study, are provided via the OSF repository.
All data is provided after FADAR classification for reproducibility and 
significant reduction of uninformative noise samples:

https://doi.org/10.17605/OSF.IO/RJ4CN

---

## Citation

If you use this repository, please cite:

> **Acs, R., Ibrahim, A., Zhuang, H., & Chérubin, L. M. (2025).**  
> *Contrastive Learning for Passive Acoustic Monitoring: A Framework for Sound Sources
Discovery and Cross-Site Comparison in Marine Soundscapes.*  
> Manuscript under review.

---

## License

This project is released under the **MIT License**.
