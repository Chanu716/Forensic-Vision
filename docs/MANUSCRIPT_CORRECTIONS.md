# Manuscript Correction & Technical Alignment Guide

This document provides a systematic, section-by-section audit of the IEEE-style research manuscript against the verified codebase of **Forensic-Vision**. It details exact technical inaccuracies in earlier drafts, specifies corrected equations and architecture diagrams, and establishes precise, verifiable wording for publication.

---

## 1. Executive Summary of Implementation Realities

| Area / Component | Manuscript Claim / Earlier Draft | Verified Code Implementation | Required Correction in Paper |
| :--- | :--- | :--- | :--- |
| **Model Pretraining** | Initialized with pretrained spatio-temporal backbones. | Random initialization. `use_pretrained: true` is an inert config flag; no external checkpoint is loaded. | State clearly that all models are trained from scratch with random weight initialization. |
| **Frame Differencing** | Sobel spatial gradient-guided frame differencing. | Absolute frame differencing $|K_f - K_{f-1}|$. Sobel filtering is implemented in `frame_diff.py` but never invoked by `DualStreamR2Plus1D`. | Replace Sobel gradient claims with absolute consecutive frame differencing. |
| **Temporal Localization** | Multi-Scale SSIM (MS-SSIM) with 5-scale pyramid and MAD-based threshold. | Single-scale SSIM (`use_multiscale: false` in `enhanced.yaml`). Multiscale branch uses 3 scales, not 5. Threshold uses fixed median drop ($\tau = \text{median} - 0.14$). `adaptive_sensitivity` is unused. | Update localization section to describe single-scale SSIM and clipped median-relative drop thresholding. |
| **Blur Perturbation** | Gaussian blur with continuous $\sigma \in \{0.5, 1.0, 1.5\}$. | Spatial average pooling (`F.avg_pool2d`). Both $\sigma=0.5$ and $\sigma=1.0$ evaluate identical $3\times3$ box filters; $\sigma=1.5$ evaluates a $5\times5$ box filter. | Re-label transformation as 2D spatial box blur / average pooling of size $3\times3$ and $5\times5$. |
| **Macro ROC-AUC** | Single uniform ROC-AUC reported as 0.9838 vs baseline 0.9676. | Discrepant metric definitions: Primary script computed trapezoidal interpolated ROC curve AUC (0.9838); comparison script computed scikit-learn One-vs-Rest macro ROC-AUC (0.9824). | Report both metric definitions explicitly, or standardize on scikit-learn OvR Macro AUC (0.9824 vs 0.9676). |
| **Clip Test Accuracy** | Claims proposed model outperforms baseline on test accuracy. | Both proposed model and baseline achieve exactly **94.24%** (262/278) clip test accuracy ($p = 0.7518$, McNemar test). | State clearly that nominal clip accuracy is identical; advantage lies in ROC-AUC separation and temporal boundary localization. |
| **Localization Precision** | "Sub-1-frame localization accuracy" ($\le 1$ frame mean error). | Mean boundary localization error across test videos is **34.39 frames**. Median error is **2.0 frames** (48.1% of videos $\le 1$ frame, 51.9% $\le 2$ frames). | Report median error (2.0 frames) and clarify that complex scene motion causes boundary ambiguity in a minority of videos. |
| **Dataset & Forgery Scope** | Real-world in-the-wild forensic validation / VIFFD generalization. | Evaluated strictly on synthetic frame insertion and deletion forgeries constructed from UCF-101 source videos. | Restrict claims to synthetic inter-frame tampering benchmarks derived from UCF-101. |

---

## 2. Section-by-Section Manuscript Corrections

### 2.1 Abstract and Introduction
* **Claim to Remove**: Do not claim "state-of-the-art" (SOTA) or "sub-one-frame global localization error."
* **Claim to Remove**: Do not claim cross-dataset evaluation on VIFFD unless VIFFD inference has actually been executed.
* **Corrected Text**:
  > "We present an enhanced dual-stream framework for video inter-frame forgery identification and temporal boundary localization. The proposed network processes raw RGB appearance alongside consecutive absolute frame differences using factorized R(2+1)D residual blocks with 3D Convolutional Block Attention Modules (CBAM). To alleviate the temporal dilution of isolated single-frame tampering seams under global average pooling, we introduce Temporal Peak-Preserving Pooling (TP-Pool), which explicitly captures maximum temporal feature contrast. Evaluated on a leak-free, source-video-partitioned benchmark derived from UCF-101, our method achieves 94.24% clip classification accuracy, an OvR Macro ROC-AUC of 0.9824 (improving from 0.9676 in the baseline 3D-CNN), and 93.02% end-to-end video-level detection accuracy with a median localization error of 2.0 frames."

---

### 2.2 Proposed Architecture (Section III)

#### 2.2.1 Two-Stream Inputs & Spatial-Temporal Convolutions
* **Incorrect Earlier Equation**:
  $$D_f = |K_f - K_{f-1}| \odot (1 + \|\nabla_{\text{Sobel}} K_f\|)$$
* **Correct Verified Equation**:
  $$D_f = |K_f - K_{f-1}|, \quad f = 2, \dots, T$$
  Input dimensions are $(B, C, T, H, W)$ for Stream 1 (RGB) and $(B, C, T-1, H, W)$ for Stream 2 (Motion difference).
* **Convolutions**:
  Each convolutional layer factorizes 3D spatio-temporal operations into a 2D spatial convolution of size $(1, 3, 3)$ followed by a 1D temporal convolution of size $(3, 1, 1)$:
  $$\text{Conv2Plus1D}(X) = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{\text{temp}}\left(\text{ReLU}\left(\text{BN}\left(\text{Conv}_{\text{spat}}(X)\right)\right)\right)\right)\right)$$

#### 2.2.2 3D Convolutional Block Attention Module (3D-CBAM)
The architecture applies channel attention followed by spatial-temporal attention inside each residual block:
1. **Channel Attention**:
   $$M_c(F) = \sigma\left(\text{MLP}(\text{AvgPool3D}(F)) + \text{MLP}(\text{MaxPool3D}(F))\right)$$
   where pooling targets shape $(B, C, 1, 1, 1)$, and the MLP consists of $1\times 1\times 1$ 3D convolutions with reduction ratio $r = 16$.
2. **Spatial-Temporal Attention**:
   $$M_s(F') = \sigma\left(f^{7\times 7\times 7}\left([\text{Mean}_c(F'); \text{Max}_c(F')]\right)\right)$$
   operating across spatial and temporal dimensions with a single 3D convolution of kernel size $7\times 7\times 7$ and padding 3.

#### 2.2.3 Temporal Peak-Preserving Pooling (TP-Pool)
* **Correct Verified Formulation**:
  Given feature map $F \in \mathbb{R}^{B \times C \times T \times H \times W}$ at the output of the final residual stage ($C = 128$):
  1. Spatial average pooling over height and width yields $F_{\text{sp}} \in \mathbb{R}^{B \times C \times T}$:
     $$F_{\text{sp}}(b, c, t) = \frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W F(b, c, t, h, w)$$
  2. Temporal statistics across the clip duration $T$:
     $$\mu_t = \frac{1}{T} \sum_{t=1}^T F_{\text{sp}}(\cdot, \cdot, t), \quad m_t = \max_{1 \le t \le T} F_{\text{sp}}(\cdot, \cdot, t), \quad \Delta_t = m_t - \mu_t$$
  3. Feature projection:
     $$F_{\text{pool}} = \text{GELU}\left(W_{\text{proj}} \left[\mu_t \,\|\, m_t \,\|\, \Delta_t\right] + b_{\text{proj}}\right) \in \mathbb{R}^{B \times C}$$
     where $W_{\text{proj}} \in \mathbb{R}^{C \times 3C}$.

#### 2.2.4 Gated Cross-Stream Fusion & Classification Head
* **Verified Formulation**:
  Let $F_{\text{rgb}} \in \mathbb{R}^{B \times C}$ and $F_{\text{diff}} \in \mathbb{R}^{B \times C}$ denote the pooled outputs of the RGB and difference streams:
  $$g = \sigma\left(W_g [F_{\text{rgb}} \,\|\, F_{\text{diff}}] + b_g\right) \in \mathbb{R}^{B \times C}$$
  $$F_{\text{fused}} = g \odot F_{\text{rgb}} + (1 - g) \odot F_{\text{diff}}$$
  **Crucial Classifier Detail**: The input to the dense classifier is the concatenation of the fused representation AND the difference stream:
  $$F_{\text{class}} = [F_{\text{fused}} \,\|\, F_{\text{diff}}] \in \mathbb{R}^{B \times 2C}$$
  $$\hat{y} = W_2 \left(\text{Dropout}_{0.3}\left(\text{GELU}\left(\text{BN}\left(W_1 F_{\text{class}} + b_1\right)\right)\right)\right) + b_2$$
  where $W_1 \in \mathbb{R}^{128 \times 256}$ and $W_2 \in \mathbb{R}^{3 \times 128}$.

---

### 2.3 Localization Algorithm (Section IV)

#### 2.3.1 Metric and Pyramid Mode
* **Verified Configuration**: The primary evaluated model runs with `use_multiscale: false`, utilizing single-scale SSIM (`skimage.metrics.structural_similarity` with $7\times 7$ window).
* If multi-scale SSIM is described as an optional extension, document that it utilizes a **3-scale pyramid** with weights $[0.0448, 0.2856, 0.3001]$ normalized to sum to 1.0 (not 5 scales).

#### 2.3.2 Calibrated Adaptive Thresholding
* **Verified Formulation**:
  For consecutive pairwise similarity scores $S = [s_1, s_2, \dots, s_{M-1}]$ across an $M$-frame video:
  $$\tau_{\text{eff}} = \max\left(0.35, \, \min\left(\tau_0, \, \text{median}(S) - 0.14\right)\right)$$
  where nominal threshold $\tau_0 = 0.85$.
  *(Note: `adaptive_sensitivity` is configured but inert in the code; do not describe a sensitivity parameter unless documenting its inert status).*

#### 2.3.3 Boundary Clustering and Forgery Type Classification
1. Flagged indices: $\mathcal{I}_{\text{susp}} = \{i \mid s_i < \tau_{\text{eff}}, \, 1 \le i < M - 1\}$.
2. Consecutive or closely-spaced dips (gap $\le 2$ frames) are merged into clusters $\mathcal{C}_1, \dots, \mathcal{C}_k$.
3. **Automated Signature Logic**:
   - $k = 0 \implies$ Authentic
   - $k = 1 \implies$ Frame Deletion (estimated seam: $[\min \mathcal{C}_1, \max \mathcal{C}_1]$)
   - $k \ge 2 \implies$ Frame Insertion (estimated span: $[\min \mathcal{C}_1, \max \mathcal{C}_k]$)

---

### 2.4 Experimental Results & Evaluation Consistency (Section V)

#### 2.4.1 Classification Performance: Baseline vs Proposed
Report the standardized continuous One-vs-Rest Macro ROC-AUC alongside clip accuracy and F1:

| Metric | Baseline 3D-CNN (Gowda & Pawar Reimpl.) | Proposed Enhanced Dual-Stream | Difference / Status |
| :--- | :---: | :---: | :---: |
| **Clip Accuracy** | 94.24% (262/278) [95% CI: 91.37–96.76%] | 94.24% (262/278) [95% CI: 91.37–96.76%] | 0.00% ($p=0.7518$) |
| **Macro F1-Score** | 0.9331 [95% CI: 0.9014–0.9633] | 0.9338 [95% CI: 0.9010–0.9614] | +0.0007 |
| **Macro ROC-AUC (OvR Standard)** | 0.9676 [95% CI: 0.9461–0.9852] | 0.9824 [95% CI: 0.9690–0.9929] | +0.0148 (+1.48%) |
| **Macro ROC-AUC (Interpolated)** | 0.9688 | 0.9838 | +0.0150 (+1.50%) |
| **Frame Deletion ROC-AUC** | 0.9448 | 0.9660 | +0.0212 (+2.12%) |
| **Frame Insertion ROC-AUC** | 0.9997 | 1.0000 | +0.0003 |
| **Video-Level Accuracy (Unified Protocol)** | 86.05% (37/43) | 93.02% (40/43) | +6.97% (+3 videos) |
| **Video-Level Accuracy (Original Protocol)** | 83.72% (36/43) | 93.02% (40/43) | +9.30% (+4 videos) |
| **Authentic Video Specificity** | 86.67% (13/15) | 100.00% (15/15) | +13.33% (0 false alarms) |
| **Deletion Video Recall (Unified Protocol)** | 69.23% (9/13) | 76.92% (10/13) | +7.69% (+1 video) |
| **Deletion Video Recall (Original Protocol)** | 61.54% (8/13) | 76.92% (10/13) | +15.38% (+2 videos) |
| **Median Localization Error (Unified Protocol)** | 2.0 frames ($N=27$) | 2.0 frames ($N=27$) | Parity (SSIM on raw frames) |
| **Mean Localization Error (Unified Protocol)** | 34.39 frames | 34.39 frames | Parity (skewed by 4 pans) |

*\*Note on Protocols & Localization*:
1. **Protocol Separation**: Under the unified single-scale protocol (`use_multiscale: false`), baseline video accuracy is 86.05% (deletion recall 69.23%). Under the original multi-scale config protocol (`use_multiscale: true`), baseline video accuracy is 83.72% (deletion recall 61.54%). These two protocols must not be blended in the paper.
2. **Localization Independence**: Temporal localization is executed via structural similarity on raw video frames without using model features. Both models achieve an identical median error of 2.0 frames across detected sequences under single-scale SSIM. Localization accuracy must not be attributed to model architectures.
3. **Statistical Significance**: Overlapping or non-overlapping bootstrap confidence intervals do not establish statistically significant ROC-AUC superiority; a paired test (e.g. DeLong) should be cited if formal significance is claimed.

#### 2.4.2 Perturbation Stress Test Labels
In Section V-D (Robustness), replace references to continuous Gaussian blur filtering with:
- **Spatial Box Blur / Average Pooling**:
  - Intensity 1: $3 \times 3$ average pooling ($kernel\_size = 3$, padding 1) $\implies$ 93.88% Accuracy, 0.9299 F1.
  - Intensity 2: $5 \times 5$ average pooling ($kernel\_size = 5$, padding 2) $\implies$ 93.88% Accuracy, 0.9299 F1.
- **Additive Tensor Noise**: Zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ clamped to $[0, 1]$ ($\sigma = 0.01, 0.03, 0.05$).
- **Illumination Scaling**: Pixel multiplier $I' = \text{clip}(\alpha I, 0, 1)$ with $\alpha \in \{0.85, 1.15\}$ ($\pm 15\%$ scale).

---

## 3. Dataset Integrity & Partition Guarantee

The experimental partitions are strictly partitioned at the source-video level:
- Total raw videos: 100 UCF-101 action videos across 8 classes.
- Partition split: 75 Train (1,478 clips), 10 Validation (185 clips), 15 Test (278 clips).
- **Leakage Verification**:
  - Overlap between Train and Test source videos: **0**
  - Overlap between Train and Test donor videos: **0**
  - Overlap between Train source and Test donor videos: **0**
  - Overlap between Test source and Train donor videos: **0**
  - Total unique clip samples: **1,941 / 1,941 (100% unique)**

This guarantees that the reported performance reflects genuine generalization to unseen source and donor videos under identical experimental protocol.
