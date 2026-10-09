# Forensic-Vision Results Reconciliation Report

**Date**: October 10, 2026  
**Repository**: `https://github.com/Chanu716/Forensic-Vision`  
**Purpose**: Comprehensive audit, provenance mapping, and reconciliation of experimental metrics across all baseline and proposed model configurations on the UCF-101 inter-frame video forgery benchmark.

---

## 1. Executive Summary & Audit Scope

This document reconciles all experimental numbers, evaluation protocols, and technical claims across historical project artifacts, comparative summaries, and the five-variant architectural ablation study.

### Key Audit Conclusions
1. **Clip Classification Accuracy Parity**: On the held-out test split of 278 clips, both the baseline 3D-CNN (Variant A) and the proposed enhanced dual-stream model (Variant E) achieve **identical 94.24% accuracy** (262/278 correct). McNemar's paired test yields $\chi^2 = 0.1000, p = 0.7518$, confirming **no statistically significant difference** in discrete clip classification.
2. **True Source of Model Advantage**: The empirical advantages of the proposed model reside in:
   - **Continuous Discriminative Margin**: One-vs-Rest (OvR) Macro ROC-AUC of **0.9824** vs. **0.9676** (+1.48%), with frame deletion ROC-AUC of **0.9660** vs. **0.9448** (+2.12%).
   - **Full Video-Level Temporal Aggregation**: Video accuracy of **93.02% (40/43)** vs. **86.05% (37/43)** under unified single-scale protocol (or **83.72% (36/43)** under original multi-scale config protocol).
   - **Zero False Alarms on Authentic Videos**: 100.0% specificity (15/15) vs. 86.67% (13/15).
3. **Decoupled Localization Reality**: The temporal boundary localization algorithm (`compute_frame_msssim_scores`) operates **exclusively on raw inter-frame structural similarity differences** and is decoupled from the neural network. Localization performance must **never** be attributed to convolutional backbones, CBAM attention, or TP-Pool. Under identical single-scale SSIM, both baseline and proposed models achieve an identical **median error of 2.0 frames**.
4. **Correction of Technical Claims**:
   - The second stream processes consecutive frame differences ($\Delta F_t = |F_t - F_{t-1}|$), not Sobel/Laplacian gradient guidance.
   - Models are initialized from scratch using Kaiming normal initialization, not pretrained on Kinetics.
   - The deployed localization pipeline uses single-scale SSIM (`use_multiscale: false`), not multi-scale MS-SSIM.
   - The adaptive threshold formula is a clipped median drop $\tau_{\text{eff}} = \text{clip}(\text{median}(S) - 0.14, 0.35, 0.85)$.
   - Perturbation robustness tests evaluate spatial box-blur average pooling ($3\times3$ and $5\times5$ filters), not continuous Gaussian blur. No H.264/HEVC compression re-encoding was evaluated.

---

## 2. Verified Metric Definitions & Statistical Standards

To prevent ambiguities across publication drafts, all metrics are mapped to explicit mathematical formulations:

### 2.1 Primary Metric: Multiclass One-vs-Rest (OvR) Macro ROC-AUC
* **Authoritative Implementation**: `sklearn.metrics.roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr")`.
* **Mathematical Definition**:
  $$\text{AUC}_{\text{macro}}^{\text{OvR}} = \frac{1}{C} \sum_{c=1}^{C} \text{AUC}\left( \mathbf{y}^{(c)}, \hat{\mathbf{p}}^{(c)} \right)$$
  where $\mathbf{y}^{(c)} \in \{0, 1\}^N$ is the binary ground-truth indicator for class $c$, and $\hat{\mathbf{p}}^{(c)} \in [0, 1]^N$ is the predicted softmax probability for class $c$.
* **Verified Values on 278 Held-Out Test Clips**:
  - Baseline 3D-CNN (Variant A): **0.9676** [95% CI: 0.9461 – 0.9852]
  - Proposed Dual-Stream (Variant E): **0.9824** [95% CI: 0.9690 – 0.9929]

### 2.2 Secondary Metric: Interpolated Macro ROC-AUC
* **Implementation**: Trapezoidal integration over the union of false-positive rate points after linearly interpolating per-class true-positive rates (`np.interp` followed by `sklearn.metrics.auc(all_fpr, mean_tpr)`).
* **Produced by**: `scripts/generate_roc_auc_evaluation.py`.
* **Verified Values**:
  - Baseline 3D-CNN (Variant A): **0.9688**
  - Proposed Dual-Stream (Variant E): **0.9838**
* **Reporting Recommendation**: Report the scikit-learn standard **0.9824** as the primary score. If the interpolated curve score (0.9838) is reported, explicitly label it as *"Interpolated ROC curve AUC"*.

### 2.3 Statistical Significance Guidance
* **Clip Accuracy**: McNemar's test with continuity correction on paired 278-clip predictions yields:
  - Both correct ($n_{11}$): 257
  - Baseline only correct ($n_{10}$): 5
  - Proposed only correct ($n_{01}$): 5
  - Both incorrect ($n_{00}$): 11
  - Test Statistic: $\chi^2 = \frac{(|5 - 5| - 1)^2}{5 + 5} = 0.1000, \quad p = 0.7518$
  - **Conclusion**: Fail to reject the null hypothesis. There is no nominal accuracy superiority.
* **ROC-AUC Confidence Intervals**: The 95% bootstrap intervals are [0.9461, 0.9852] (baseline) and [0.9690, 0.9929] (proposed).
  > **Methodological Rule**: Do not claim statistically significant ROC-AUC superiority solely based on overlapping or non-overlapping bootstrap confidence intervals. If formal hypothesis testing of ROC-AUC superiority is required by reviewers, conduct a paired DeLong test or paired bootstrap test on the predictions.

---

## 3. Prediction & Checkpoint Provenance

All test predictions were traced back to specific disk artifacts and verified for bit-for-bit consistency:

### 3.1 Saved Test Predictions Equivalence
| Architecture | Artifact Path 1 | Artifact Path 2 | Clip Count | Discordant Labels | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Variant A** (Baseline) | `outputs/ablation/variant_a/test_predictions.csv` | `outputs/reports/standardized_test_predictions_base.csv` | 278 | **0** | Verified Identical |
| **Variant E** (Proposed) | `outputs/ablation/variant_e/test_predictions.csv` | `outputs/reports_enhanced/standardized_test_predictions_enhanced.csv` | 278 | **0** | Verified Identical |

### 3.2 Checkpoint Provenance & Training Configurations
| Variant | Source Checkpoint | Parameters | Training Epochs | Optimizer & Loss | Best Epoch Selected | Validation Metric at Checkpoint |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Variant A** | `outputs/checkpoints/best.pt` | 280,131 | 15 epochs | Adam ($\text{lr}=10^{-3}$), CrossEntropy | Epoch 14 | Val Acc: 95.68%, Val F1: 0.9520 |
| **Variant B** | `outputs/ablation/variant_b/best.pt` | 917,462 | 10 epochs | Adam ($\text{lr}=3\times 10^{-4}$), Focal Loss, $T_{\max}=10$ | Epoch 10 | Val Acc: 91.35%, Val F1: 0.8992 |
| **Variant C** | `outputs/ablation/variant_c/best.pt` | 1,867,049 | 10 epochs | Adam ($\text{lr}=3\times 10^{-4}$), Focal Loss, $T_{\max}=10$ | Epoch 4 | Val Acc: 93.51%, Val F1: 0.9259 |
| **Variant D** | `outputs/ablation/variant_d/best.pt` | 1,876,541 | 10 epochs | Adam ($\text{lr}=3\times 10^{-4}$), Focal Loss, $T_{\max}=10$ | Epoch 2 | Val Acc: 96.22%, Val F1: 0.9551 |
| **Variant E** | `outputs/checkpoints_enhanced/best.pt` | 1,975,101 | 20 epochs | Adam ($\text{lr}=3\times 10^{-4}$), Focal Loss, $T_{\max}=20$ | **Epoch 8** | Val Acc: 94.59%, Val F1: 0.9367 |

### 3.3 Learning Rate Scheduler Nuance for Variant E
Variant E's historical checkpoint was trained with a 20-epoch cosine annealing schedule ($T_{\max}=20$). The best validation F1 occurred at **Epoch 8**, and was retained without modification in adherence to non-negotiable repository constraints.
* **Important Scientific Clarification**: Achieving the best validation score at Epoch 8 does **not** make the training schedule identical to a 10-epoch schedule. Under $T_{\max}=20$, the learning rate at Epoch 8 was:
  $$\eta_8 = \frac{1}{2}\eta_0 \left(1 + \cos\left(\frac{8\pi}{20}\right)\right) \approx 0.5 \times (1 + 0.3090) \times 0.0003 \approx \mathbf{1.96 \times 10^{-4}}$$
  Whereas under a 10-epoch schedule ($T_{\max}=10$), at Epoch 8 the learning rate would have decayed to:
  $$\eta_8 = \frac{1}{2}\eta_0 \left(1 + \cos\left(\frac{8\pi}{10}\right)\right) \approx 0.5 \times (1 - 0.8090) \times 0.0003 \approx \mathbf{2.87 \times 10^{-5}}$$
* **Reproducible Matched 10-Epoch Option**: If a reviewer requires a strictly controlled 10-epoch schedule to compare Variant E directly with Variant D, run:
  ```bash
  python scripts/run_ablation_study.py --variant e_10ep --epochs 10
  ```
  This trains Variant E from scratch for 10 epochs ($T_{\max}=10$, `seed=42`) and writes results to `outputs/ablation/variant_e_10ep/`, leaving the benchmark checkpoint untouched.

---

## 4. Separation of Video-Level Evaluation Protocols

Historical discrepancies in video-level accuracy (83.72% vs. 86.05%) and deletion recall (61.54% vs. 69.23%) arose because two distinct localization configurations were used in the project. These protocols must **never** be blended into a single comparison.

### Protocol 1: Unified Single-Scale Evaluation Protocol
* **Configuration**: `threshold: 0.85`, `use_adaptive_threshold: true`, `adaptive_sensitivity: 3.5`, `use_multiscale: false`.
* **Used by**: `scripts/run_ablation_study.py` (applied identically to Variants A, B, C, D, and E).
* **Results**:
  - Baseline 3D-CNN (Variant A): Video Accuracy = **86.05% (37/43)**, Authentic Specificity = **86.67% (13/15)**, Deletion Recall = **69.23% (9/13)**, Median Loc Error = **2.0 frames** ($N=27$), Mean Loc Error = **34.39 frames**.
  - Proposed Dual-Stream (Variant E): Video Accuracy = **93.02% (40/43)**, Authentic Specificity = **100.00% (15/15)**, Deletion Recall = **76.92% (10/13)**, Median Loc Error = **2.0 frames** ($N=27$), Mean Loc Error = **34.39 frames**.

### Protocol 2: Original Configuration Evaluation Protocol
* **Configuration**:
  - Baseline 3D-CNN: Evaluated via `configs/base.yaml` using multi-scale SSIM (`use_multiscale: true`, `threshold: 0.80`).
  - Proposed Dual-Stream: Evaluated via `configs/enhanced.yaml` using single-scale SSIM (`use_multiscale: false`, `threshold: 0.85`).
* **Used by**: `scripts/evaluate_video_dataset.py`, `BENCHMARK_RESULTS.md`, and `docs/COMPARISON_AND_VALIDATION.md`.
* **Results**:
  - Baseline 3D-CNN: Video Accuracy = **83.72% (36/43)**, Authentic Specificity = **86.67% (13/15)**, Deletion Recall = **61.54% (8/13)**, Median Loc Error = **1.0 frame** ($N=23$), Mean Loc Error = **14.39 frames**.
  - Proposed Dual-Stream: Video Accuracy = **93.02% (40/43)**, Authentic Specificity = **100.00% (15/15)**, Deletion Recall = **76.92% (10/13)**, Median Loc Error = **2.0 frames** ($N=27$), Mean Loc Error = **34.39 frames**.

### 4.2 Combined Model-Plus-SSIM Video Decision Pipeline
Video-level classification is **not** a simple average or majority vote of clip predictions. It is evaluated as the end-to-end performance of a **hybrid decision fusion pipeline** combining deep model confidence with structural similarity (SSIM) transition clustering:

```python
# Video Decision Aggregation Rules (evaluate_video_dataset.py & run_ablation_study.py)
if max_insertion >= 0.65 and max_insertion > max_deletion:
    pred_label = "frame_insertion"
elif max_deletion >= 0.75 and max_deletion > max_insertion and localization.forgery_type != "authentic":
    pred_label = "frame_deletion"
elif localization.forgery_type == "frame_insertion" and max_insertion >= 0.20:
    pred_label = "frame_insertion"
elif localization.forgery_type == "frame_deletion" and max_deletion >= 0.10:
    pred_label = "frame_deletion"
elif max_deletion >= 0.50 and max_deletion > max_insertion and max_deletion > mean_auth:
    pred_label = "frame_deletion"
else:
    pred_label = "authentic"
```

### 4.3 Why Deletion Recall and Localization Differed Across Protocols
1. **Decision Rule Coupling**: 
   - Under Multi-Scale SSIM (`use_multiscale: true`, threshold 0.80), transition dips on `v_ApplyEyeMakeup_g12_c05_delete` and `v_ApplyEyeMakeup_g15_c06_delete` were smoothed out across scales, returning `authentic`. Because model deletion probability fell below the unassisted threshold, both videos were classified as authentic (8/13 correct = 61.54%).
   - Under Single-Scale SSIM (`use_multiscale: false`, threshold 0.85), sharp localized frame-to-frame seams were detected, triggering the override rule `localization.forgery_type == "frame_deletion" and max_del >= 0.10` (9/13 correct = 69.23%).
2. **Localization Denominator & Missed Detection Handling**:
   - The test set contains **43 full videos**: 15 authentic, 15 frame insertion, and 13 frame deletion. Total tampered videos with ground-truth seams: **28 videos**.
   - **Multi-Scale SSIM ($N=23$ detected)**: Successfully detected transitions in 23 of 28 tampered videos; **5 deletion videos were completely missed** (`loc_err = None`: `v_BandMarching_g04_c02_delete`, `v_Basketball_g12_c01_delete`, `v_ApplyEyeMakeup_g12_c05_delete`, `v_ApplyEyeMakeup_g19_c02_delete`, `v_ApplyEyeMakeup_g15_c06_delete`). The reported mean of **14.39 frames** and median of **1.0 frame** used denominator $N = 23$, measuring **conditional error only on detected sequences**. Missed detections were completely dropped from the error calculation rather than penalized.
   - **Single-Scale SSIM ($N=27$ detected)**: Successfully detected transitions in 27 of 28 tampered videos; **only 1 deletion video was missed** (`v_ApplyEyeMakeup_g19_c02_delete`). The reported median of **2.0 frames** and mean of **34.39 frames** used denominator $N = 27$ (conditional on detected sequences).
   - **Impact of Missed Detections**: Dropping missed detections artificially flatters the multi-scale metric because the 5 hardest, low-contrast deletion seams were excluded from its average. Single-scale SSIM successfully resolved 4 of those 5 difficult seams, increasing localization recall from 82.1% (23/28) to 96.4% (27/28), while retaining a median error of 2.0 frames.

---

## 5. Comprehensive 5-Variant Ablation Findings

The complete ablation study across 5 progressive configurations provides the empirical foundation for architectural analysis:

| Variant | Description | Parameters | Clip Acc (%) | Macro F1 | OvR ROC-AUC | Deletion AUC | Video Acc (%) | Authentic Spec (%) | Deletion Rec (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Baseline 3D-CNN (Diff only, GAP) | 280,131 | 94.24% | 0.9331 | 0.9676 | 0.9448 | 86.05% | 86.67% | 69.23% |
| **B** | Single-Stream R(2+1)D (RGB only, GAP) | 917,462 | 88.13% | 0.8687 | 0.9726 | 0.9540 | 76.74% | 73.33% | 84.62% |
| **C** | Dual-Stream R(2+1)D (RGB + Diff, GAP) | 1,867,049 | 92.45% | 0.9144 | 0.9914 | 0.9837 | 81.40% | 80.00% | 76.92% |
| **D** | Dual-Stream + CBAM (GAP) | 1,876,541 | 83.45% | 0.7994 | 0.9748 | 0.9481 | 69.77% | 86.67% | 38.46% |
| **E** | Proposed Full (CBAM + TP-Pool) | 1,975,101 | **94.24%** | **0.9338** | **0.9824** | **0.9660** | **93.02%** | **100.00%** | **76.92%** |

### Critical Architectural & Methodological Attribution
1. **Necessity of Dual Modalities (B vs. C)**: The single RGB stream alone (Variant B) achieves 88.13% clip accuracy. Adding the explicit frame difference stream ($\Delta F_t$) in Variant C elevates accuracy to 92.45% (+4.32 pp) and OvR ROC-AUC to 0.9914 (+0.0188), proving motion discontinuity is essential.
2. **Confounded Comparison Between Variant D and Variant E (Training Schedule Nuance)**:
   - While Variant E (with TP-Pool) demonstrates substantially higher test performance than Variant D (with GAP) — +10.79 pp in clip accuracy (94.24% vs. 83.45%), +38.46 pp in deletion recall (76.92% vs. 38.46%), and +23.25 pp in video accuracy (93.02% vs. 69.77%) — **this performance difference cannot be attributed solely to TP-Pool**.
   - **Confounding Factor**: Variant D was trained from scratch under a 10-epoch cosine schedule ($T_{\max}=10$) with its optimal validation checkpoint selected at **Epoch 2**. Conversely, Variant E's checkpoint was trained under a 20-epoch schedule ($T_{\max}=20$) and snapshotted at **Epoch 8** with 4x more training iterations and a different learning rate ($\eta_8 \approx 1.96 \times 10^{-4}$ vs. Variant D's $\eta_2 \approx 2.71 \times 10^{-4}$).
   - **Scientific Assessment**: Although the theoretical motivation for TP-Pool (preventing temporal dilution of localized 1-frame spikes) is sound, the empirical gap between D and E reflects **both** architectural pooling design and optimization trajectory differences. For a strict unconfounded claim, a matched 10-epoch experiment (`python scripts/run_ablation_study.py --variant e_10ep --epochs 10`) is required.

---

## 6. Implementation Realities & Paper Claims Reconciliation

| Paper Claim / Prior Assumption | Code Reality | Discrepancy Level | Reconciliation Action |
| :--- | :--- | :---: | :--- |
| **Gradient Guidance** | Code ignores `use_gradient_guidance: true`; second stream is consecutive frame differences ($\Delta F$). | Medium | Replace "gradient guidance" with "inter-frame difference stream $\Delta F_t = \|F_t - F_{t-1}\|$". |
| **Pretrained Weights** | Code ignores `use_pretrained: true`; models are initialized randomly using Kaiming normal. | Medium | State models are trained from scratch; remove Kinetics pretraining claims. |
| **MS-SSIM Multi-Scale** | Code explicitly sets `use_multiscale: false` (single-scale SSIM with Gaussian window size 11, $\sigma=1.5$). | Medium | Update technical description to single-scale SSIM; document historical multi-scale exploration. |
| **Adaptive Threshold** | Code ignores `adaptive_sensitivity: 3.5`; uses clipped median drop $\tau_{\text{eff}} = \text{clip}(\text{median}(S) - 0.14, 0.35, 0.85)$. | Medium | State the exact clipped median formula; remove $\mu - k\sigma$ equations. |
| **Robustness Filters** | "Gaussian blur" is spatial average pooling (`F.avg_pool2d`) with identical $3\times3$ box filters for $\sigma=0.5, 1.0$ and $5\times5$ for $\sigma=1.5$. No video compression was tested. | High | Replace "Gaussian blur" with "spatial average pooling blur"; remove H.264/HEVC compression claims. |
| **Localization Precision** | SSIM detector is independent of neural network; both models achieve 2.0 frames median error. True mean is 34.39 frames due to 4 athletic panning failures. | High | Report median error (2.0 frames) and interquartile range; do not claim 1.33-frame mean accuracy or model-induced localization gains. |
| **Split Integrity** | 100 raw UCF-101 videos partitioned into 75 Train (1,478 clips), 10 Val (185 clips), 15 Test (278 clips). | Sound | Confirmed: 0 source-video overlap, 0 donor overlap, 0 cross-split leakage. |

---

## 7. Unresolved Methodological Limitations

1. **Camera Panning Sensitivity**: In scenes with rapid rotational motion and complex geometric transformations (e.g., `Basketball`), inter-frame structural similarity naturally drops across untampered frames, creating secondary dips that confuse boundary clustering and produce 4 large localization errors (skewing mean error to 34.39 frames).
2. **Synthetic vs. Real-World Inpainting**: Synthetic frame deletion and insertion introduce sharp frame-level boundaries. Real-world commercial deepfakes or generative inpainting may incorporate temporal blending or optical-flow alignment that diminishes structural seam sharpness.
3. **Single-Source Insertion**: The synthetic benchmark inserts sequences from a single donor video rather than multi-source composites, which may limit generalizability to multi-donor splicing.
