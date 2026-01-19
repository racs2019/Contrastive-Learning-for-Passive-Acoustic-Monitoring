# 🪸 Contrastive Learning for Passive Acoustic Monitoring (PAM)

> **A scalable, unsupervised framework for sound source discovery and cross-site comparison in marine soundscapes**

This repository implements the framework introduced in  
**“Contrastive Learning for Passive Acoustic Monitoring: A Framework for Sound Sources Discovery and Cross-Site Comparison in Marine Soundscapes”**  
by *Richard Acs, Ali Ibrahim, Hanqi Zhuang, and Laurent M. Chérubin.*

---

## Overview

Marine Passive Acoustic Monitoring (PAM) enables long-term, non-invasive biodiversity observation but faces key challenges — overlapping sources, high noise, and limited labeled data.  
This framework introduces a **contrastive learning pipeline** tailored for noisy marine environments, capable of unsupervised discovery of biotic and anthropogenic sound sources and comparison across sites and years.

The approach outperforms classical cepstral and generative baselines in **clustering coherence** and **acoustic pattern interpretability**, while requiring **no manual annotations**.

---

## Key Features

- **Unsupervised discovery** of fish calls and anthropogenic signals in reef soundscapes  
- **Teacher-guided multi-positive SimCLR** with local–global invariance and VICReg regularization  
- **Optimized augmentations** for underwater acoustics (spectral notch, frequency masking, noise, shifts)  
- **Cross-site generalization** across 10+ Caribbean reef monitoring sites  
- **Contrastive embeddings** that support downstream clustering, call signature discovery, and ecological interpretation  

---

## Framework Overview

### Architecture
- **Encoder:** ResNet-18 (1-channel)  
- **Projection Head:** 512 → 256 → 128 (BN–ReLU–ℓ2)  
- **Predictor (SimSiam):** 128 → 256 → 128  
- **Teacher Model:** EMA (momentum 0.99 → 0.9995)  
- **Feature Bank:** FIFO queue (size 8,192)

### Objective
The total loss combines contrastive alignment, local–global invariance, and variance regularization:

$$
L_{\text{total}} = \alpha L_{\text{ctr}} + \beta L_{\text{siam}} + \gamma L_{\text{vic}}
$$

with weights: **α = 1.0, β = 0.1, γ = 0.1**

---

## Dataset

Recordings were collected from **seven Caribbean spawning aggregation sites (2017–2024)**, including:

| Location | Dominant Species | # of Recordings |
|-----------|------------------|-----------------|
| Puerto Rico (ALS, ALS Deep, BDS, Mona Elbow) | Red Hind, Nassau Grouper | 260k+ |
| St. Thomas (RHB, GB) | Red Hind, Yellowfin Grouper | 140k+ |
| Mexico (Xcalak, Punta Allen, San Juan) | Nassau Grouper, Toadfish | 100k+ |

Each 20 s clip is segmented into 2 s windows, downsampled to **10 kHz**, and converted into **log-Mel spectrograms** (typically **128 Mel bands**, 0–5 kHz).

---

## Preprocessing & Augmentations

- **Mono conversion & amplitude normalization** to [−1, 1]  
- **Event-centric cropping:** 2 global (256 frames) + 2 local (96 frames) crops  
- **Augmentations:**  
  - Time/frequency masking (SpecAugment)  
  - Spectral notch dropout  
  - Temporal shift/truncation (≤ 12%)  
  - Gaussian noise (σ = 0.01)  

---

## Clustering & Evaluation

| Algorithm | Parameters | Notes |
|------------|-------------|-------|
| **K-Means** | k = 6–60 | Fast, interpretable |
| **GMM** | full covariance | Best balance of cohesion & separation |
| **DBSCAN/HDBSCAN** | ε = 0.5, min_samples = 10 | Adaptive cluster counts |
| **Spectral Clustering** | 10-NN graph | Good for complex manifolds |

### Evaluation Metrics
- **External:** Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), Hungarian Accuracy  
- **Internal:** Silhouette, Davies–Bouldin Index (DBI), Calinski–Harabasz (CH)  

---

## Results Summary

| Method | ARI | AMI | Hungarian | Silhouette | DBI | CH |
|--------|------|------|------------|-------------|------|------|
| GTCC + MFCC | 0.089 | 0.140 | 0.333 | 0.114 | 2.02 | 422 |
| CNN SupCon | **0.372** | **0.396** | **0.646** | 0.302 | 1.18 | 29,187 |
| **SimCLR (Ours)** | 0.070 | 0.121 | 0.317 | **0.220** | **1.28** | **16,200** |

> SimCLR achieves the best **unsupervised** cluster structure.

---

## Acoustic Signature Discovery

Across Caribbean reef sites, the framework identified **recurring call signatures**, e.g.:

| Cluster ID | Sites | Frequency (Hz) | Notes |
|-------------|-------|----------------|-------|
| 6 | RHB | 200–800 | Red hind pulse trains |
| 14 | BDS | 200–600 | Marine mammal tones |
| 16 | Xcalak, Punta Allen | 0–600 | Toadfish harmonic calls |
| 19 | GB, ALS | 0–100 | Boat noise |
| 33 | Multi-site | Full (0–800) | Biotic narrowband pulse |

See **Appendix A** for the full call signature dictionary.

---

## Implementation

**Environment:**
- Python **3.11**  
- PyTorch **2.1**, torchaudio **2.1**  
- scikit-learn **1.3** for clustering  
- GPU (tested): **NVIDIA RTX A6000 (48 GB)**

**Training:**
- Optimizer: **AdamW** (3e−4, wd = 1e−4)  
- Warmup **10 epochs** → cosine schedule  
- Batch **256** (grad accum = 2)  
- Mixed precision (FP16/BF16)

---

## Repository Structure

All benchmarking methods and the main SimCLR model are contained in their separate 
.ipynb files, each containing the model code, training/testing runner, and code to generate 
a report file with clustering UMAP and metrics. The files below represent the following methods:
~~~text

GTCC+MFCC          GTCC+MFCC.ipynb
Log-Mel + PCA      Spectrogram + PCA.ipynb
CNN Latent Space   FADAR_Embeddings.ipynb
CNN-SupCon         CNN SupCon.ipynb
VAE + GMM          VAE+GMM.ipynb
PAM-SimCLR         SimCLR(main).ipynb
Vanilla SimCLR     Vanilla_SimCLR.py

~~~

---

## Data Availability

Training and test sets, as well as preprocessed subsets of the passive acoustic datasets used in this study are provided in the Zenodo repository linked below:

---

## Citation

If you use this repository, please cite:

> **Acs, R., Ibrahim, A., Zhuang, H., & Chérubin, L. M. (2025).**  
> *Contrastive Learning for Passive Acoustic Monitoring: A Framework for Sound Sources Discovery and Cross-Site Comparison in Marine Soundscapes.*  
> Manuscript. (Under review, DOI will be added when available)

---

## License

This project is released under the **MIT License**.  
