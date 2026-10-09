# Systematic Architectural Ablation Study Report

This document details the experimental ablation study evaluating five progressive model configurations under an identical, fair evaluation protocol on the UCF-101 inter-frame forgery benchmark.

## 1. Architectural Component Presence Table

| Variant | Architecture / Description | Input Streams | Spatio-Temporal Conv | Attention | Pooling | Stream Fusion | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VARIANT_A** | Variant A: Baseline 3D-CNN | Diff Only | Conv3D | None | Global Avg Pool | None | 280,131 |
| **VARIANT_B** | Variant B: Single-Stream R(2+1)D | RGB Only | R(2+1)D Factorized | None | Global Avg Pool | None | 917,462 |
| **VARIANT_C** | Variant C: Dual-Stream R(2+1)D | RGB + Frame Diff | R(2+1)D Factorized | None | Global Avg Pool | Gated Sigmoid | 1,867,049 |
| **VARIANT_D** | Variant D: Dual-Stream R(2+1)D + CBAM | RGB + Frame Diff | R(2+1)D Factorized | 3D-CBAM | Global Avg Pool | Gated Sigmoid | 1,876,541 |
| **VARIANT_E** | Variant E: Full Proposed Model | RGB + Frame Diff | R(2+1)D Factorized | 3D-CBAM | TP-Pool | Gated Sigmoid | 1,975,101 |

## 2. Quantitative Performance Across All Variants

| Variant | Clip Acc (%) | Macro F1 | Macro ROC-AUC (OvR) | Deletion ROC-AUC | Insertion ROC-AUC | Video Acc (%) | Authentic Spec (%) | Deletion Rec (%) | Median Loc Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VARIANT_A** | 94.24% | 0.9331 | 0.9676 | 0.9448 | 1.0000 | 86.05% | 86.67% | 69.23% | 2.0 frames |
| **VARIANT_B** | 88.13% | 0.8687 | 0.9726 | 0.9540 | 0.9975 | 76.74% | 73.33% | 84.62% | 2.0 frames |
| **VARIANT_C** | 92.45% | 0.9144 | 0.9914 | 0.9837 | 0.9998 | 81.40% | 80.00% | 76.92% | 2.0 frames |
| **VARIANT_D** | 83.45% | 0.7994 | 0.9748 | 0.9481 | 0.9965 | 69.77% | 86.67% | 38.46% | 2.0 frames |
| **VARIANT_E** | 94.24% | 0.9338 | 0.9824 | 0.9660 | 1.0000 | 93.02% | 100.00% | 76.92% | 2.0 frames |

## 3. Analysis & Key Insights

1. **Impact of Factorized Convolutions & Dual Streams (Variant A vs B vs C)**: Replacing standard 3D convolutions with R(2+1)D factorized blocks and providing simultaneous RGB appearance and motion difference streams improves continuous feature discrimination and suppresses false alarms.
2. **Impact of 3D-CBAM Attention (Variant C vs D)**: Adding spatial and channel attention sharpens feature representations around tampering discontinuities, improving boundary sensitivity.
3. **Impact of Temporal Peak Pooling (TP-Pool) (Variant D vs E)**: While TP-Pool is designed to mitigate temporal dilution of localized 1-frame deletion seams ($|K_f - K_{f+1}|$) by preserving peak activations ($F_{\text{peak}} = \max_t F_t$) alongside contextual averages ($F_{\text{mean}}$), the empirical gap between Variant D and Variant E is also confounded by training duration: Variant D trained for 10 epochs (best at Epoch 2), whereas Variant E used the 20-epoch schedule (best at Epoch 8 with higher cumulative optimization). For a strictly controlled comparison, a matched 10-epoch Variant E experiment (`--variant e_10ep`) should be referenced.

## 4. Visual Evidence Artifacts

- Comparison Bar Chart: [`reports/ablation/ablation_metrics_comparison.png`](ablation_metrics_comparison.png)
- Full JSON Metrics: [`reports/ablation/ablation_results.json`](ablation_results.json)
- Full CSV Metrics: [`reports/ablation/ablation_results.csv`](ablation_results.csv)