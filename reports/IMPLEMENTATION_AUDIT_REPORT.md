# Forensic-Vision: Comprehensive Implementation Audit & Verification Report

This document delivers an exhaustive, evidence-based audit of the **Forensic-Vision** codebase against the reference paper (*Gowda & Pawar, 2023*) and the associated research manuscript.

---

## 1. Audit Table

The table below catalogs every key claim, the verified reality in the implementation, and the exact required action:

| Claim or Behavior | Actual Implementation | Paper Accurately Describes It? | Required Action | Action Type |
| :--- | :--- | :---: | :--- | :---: |
| **Three-Class Task** | Authentic, Frame Insertion, Frame Deletion (`num_classes: 3`). | Yes | Retain as is. | Documentation Only |
| **Dual-Stream Inputs** | Stream 1: RGB raw frames ($B, 3, T, H, W$); Stream 2: Consecutive frame absolute differences $\|K_f - K_{f-1}\|$ ($B, 3, T-1, H, W$). | Partially (claimed Sobel weighting). | Update paper to state absolute frame differences; remove Sobel gradient claims for dual-stream. | Documentation Only |
| **Spatio-Temporal Factorization** | (2+1)D factorized blocks: 2D Spatial Conv ($1\times 3\times 3$) + 1D Temporal Conv ($3\times 1\times 1$) with BatchNorm3D and ReLU. | Yes | Retain architecture; document exact kernel factorization formula. | Documentation Only |
| **3D-CBAM Attention** | Channel attention (reduction ratio 16, AdaptiveAvgPool3D + AdaptiveMaxPool3D) followed by Spatio-temporal attention ($7\times 7\times 7$ 3D Conv). | Yes | Retain architecture; update equations in paper to reflect 3D spatio-temporal attention. | Documentation Only |
| **TP-Pool Pooling** | Pools spatially $(H, W) \to (1, 1)$, computes temporal mean ($\mu$), temporal max ($m$), and temporal contrast ($m - \mu$), concatenated and projected via Linear($3C, C$) + GELU. | Partially | Clarify in paper that projection is Linear($3C, C$) + GELU; input is 3 statistics. | Documentation Only |
| **Gated Cross-Stream Fusion** | Sigmoid gate: $g = \sigma(W[F_{\text{rgb}} \,\|\, F_{\text{diff}}] + b)$; $F_{\text{fused}} = g \cdot F_{\text{rgb}} + (1 - g) \cdot F_{\text{diff}}$. | Yes | Retain mathematical formulation in paper. | Documentation Only |
| **Dense Classifier Head** | Classifier input is $[F_{\text{fused}} \,\|\, F_{\text{diff}}]$ (dimension $2C = 256$), followed by Linear(256, 128), BatchNorm1d, GELU, Dropout(0.3), Linear(128, 3). | No (manuscript omitted difference stream concatenation in head). | Correct architecture text and diagram to explicitly include difference stream bypass to classifier. | Documentation Only |
| **Pretrained Weights** | `use_pretrained: true` in config and `pretrained: bool = True` in constructor. | No (inert flag; weights are randomly initialized). | Add warning for inert option; update paper to state training from scratch. | Documentation Only |
| **Gradient Guidance** | `use_gradient_guidance: true` in `configs/enhanced.yaml`. | No (inert flag; `DualStreamR2Plus1D` directly computes absolute differences). | Add warning for inert option; remove Sobel weighting claims. | Documentation Only |
| **Localization Multi-Scale Mode** | `use_multiscale: false` in `configs/enhanced.yaml`. Code uses single-scale SSIM (`skimage.metrics.structural_similarity`). | No (paper described 5-scale MS-SSIM). | Update paper to describe single-scale SSIM as evaluated; note that optional multiscale code uses 3 scales. | Documentation Only |
| **Localization Adaptive Threshold** | Clipped median drop: $\tau_{\text{eff}} = \max(0.35, \min(\tau, \text{median} - 0.14))$. | No (manuscript described MAD formula). | Update equation in manuscript to clipped median drop ($\tau = \text{median} - 0.14$). | Documentation Only |
| **Adaptive Sensitivity** | `adaptive_sensitivity: 3.5` in config and function signature. | No (inert parameter in threshold calculation). | Document that `adaptive_sensitivity` is unused by the threshold equation. | Documentation Only |
| **Localization Error & Model Independence** | Claimed overall $\le 1$ frame localization error (1.33 frames) driven by model enhancements. | SSIM detector operates strictly on raw video frames without using model features. Under unified single-scale protocol, both baseline and proposed models achieve 2.0 frames median error (51.9% $\le 2$ frames). True mean is 34.39 frames due to 4 rapid rotational camera pans (`Basketball`). | Report median error (2.0 frames); clarify that localization is independent of the neural network architecture. | Documentation Only |
| **Blur Perturbation** | Claimed Gaussian blur with continuous $\sigma \in \{0.5, 1.0, 1.5\}$. | No (evaluated using 2D average pooling; $\sigma=0.5$ and $1.0$ both use $3\times 3$ box filter; $\sigma=1.5$ uses $5\times 5$). | Correct labels to 2D Spatial Box Blur / Average Pooling ($3\times 3$ and $5\times 5$). | Documentation Only |
| **Macro ROC-AUC Metric** | Discrepancy: Primary script used trapezoidal interpolated ROC curve AUC (0.9838); comparison script used scikit-learn OvR Macro AUC (0.9824). | Discrepant between scripts. | Standardize evaluation on both definitions; document explicitly; save separately. | Evaluation Only |
| **Clip Test Accuracy** | Claimed proposed model outperforms baseline on test accuracy. | Baseline: 94.24% (262/278); Proposed: 94.24% (262/278). McNemar $p = 0.7518$. | State nominal clip accuracy is identical; advantage is continuous AUC margin (0.9824 vs 0.9676) and video localization. | Documentation Only |
| **Dataset Leakage** | Suspected source-video leakage across train/val/test splits. | Verified 0 source video overlap, 0 donor video overlap, 0 cross-split contamination (1,941/1,941 unique clips). | Document verified leak-free dataset partition guarantee. | Documentation Only |
| **In-Place Autograd Bug** | In `R2Plus1DResidualBlock`, `out += identity` caused autograd runtime error when `use_cbam=False`. | Genuine software bug in residual block. | Fixed to `out = out + identity` (preserves identical forward pass while fixing backward autodiff). | Code Fix |
| **Ablation Study** | Missing complete 5-variant empirical ablation study. | Baseline and Proposed checkpoints existed; intermediate variants B, C, D were unexecuted. | Implement modular ablation runner; train Variants B, C, D; evaluate all 5 variants under unified protocol. | New Experiments |

---

## 2. In-Depth Technical Verification

### 2.1 Classification Architecture
1. **Model Backbone**:
   `DualStreamR2Plus1D` consists of two parallel factorized R(2+1)D backbones:
   - Stream 1 operates on raw RGB frames: `clips` $(B, 3, T, H, W)$.
   - Stream 2 operates on consecutive absolute frame differences: `torch.abs(clips[:, :, 1:] - clips[:, :, :-1])` $(B, 3, T-1, H, W)$.
   - Each backbone comprises:
     - Stage 1: `Conv2Plus1D(3, 32, stride=(2, 2, 2))` $\to$ `R2Plus1DResidualBlock(32, 32, stride=(1, 1, 1))`
     - Stage 2: `R2Plus1DResidualBlock(32, 64, stride=(2, 2, 2))`
     - Stage 3: `R2Plus1DResidualBlock(64, 128, stride=(2, 2, 2))`
   - Attention: When `use_cbam=True`, each residual block applies `CBAM3D(out_channels)`.
2. **Temporal Pooling**:
   Both streams are pooled via `TemporalPeakPooling(128)`:
   - Spatial average pool: `AdaptiveAvgPool3d((None, 1, 1))` $\implies (B, C, T)$.
   - Temporal statistics: $t_{\text{mean}}$, $t_{\text{max}}$, and $t_{\text{contrast}} = t_{\text{max}} - t_{\text{mean}}$.
   - Concatenated to $(B, 384)$ and projected via `Linear(384, 128)` followed by `GELU()`.
3. **Stream Fusion**:
   - Concatenation: $[F_{\text{rgb}}, F_{\text{diff}}] \in \mathbb{R}^{B \times 256}$.
   - Gate: $g = \sigma(\text{Linear}(256, 128))$.
   - Fused feature: $F_{\text{fused}} = g \odot F_{\text{rgb}} + (1 - g) \odot F_{\text{diff}} \in \mathbb{R}^{B \times 128}$.
4. **Classifier Head**:
   - Input: Concatenation $[F_{\text{fused}}, F_{\text{diff}}] \in \mathbb{R}^{B \times 256}$.
   - Dense projection: `Linear(256, 128)` $\to$ `BatchNorm1d(128)` $\to$ `GELU()` $\to$ `Dropout(0.3)` $\to$ `Linear(128, 3)`.

### 2.2 Temporal Localization Algorithm
- Mode: Single-scale structural similarity (`use_multiscale: false`), computed with window size 7 and channel axis -1.
- Dynamic Threshold:
  $$\tau_{\text{eff}} = \max\left(0.35, \, \min\left(0.85, \, \text{median}(S) - 0.14\right)\right)$$
- Anomaly Clustering: Dips with gap $\le 2$ frames are clustered.
- Decision Logic:
  - 0 clusters $\implies$ Authentic
  - 1 cluster $\implies$ Frame Deletion (start and end bounded by cluster)
  - $\ge 2$ clusters $\implies$ Frame Insertion (spanned from start of first to end of last cluster)

### 2.3 Dataset Integrity
The dataset split strictly isolates raw videos at the source level:
- Total raw videos: 100
- Train split: 75 videos (1,478 clips: 430 authentic, 666 insertion, 382 deletion)
- Validation split: 10 videos (185 clips: 54 authentic, 85 insertion, 46 deletion)
- Test split: 15 videos (278 clips: 92 authentic, 114 insertion, 72 deletion)
- Cross-split source overlap: **0**
- Cross-split donor overlap: **0**
- Duplicate clips across splits: **0**
