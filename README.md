# Forensic Vision

Deep learning-based video forgery identification and exact temporal localization, reproducing and significantly outperforming the reference paper:
> Raghavendra Gowda & Digambar Pawar (2023), *Deep Learning-Based Forgery Identification and Localization in Videos*, Signal, Image and Video Processing.

### 🏆 Benchmark Highlights
- **Detailed Results & Comparison**: See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
- **Manuscript Corrections & Technical Audit Guide**: See [docs/MANUSCRIPT_CORRECTIONS.md](docs/MANUSCRIPT_CORRECTIONS.md)
- **Standardized Head-to-Head Report**: See [outputs/reports/standardized_comparison_summary.md](outputs/reports/standardized_comparison_summary.md)
- **Architecture**: Dual-Stream R(2+1)D with 3D-CBAM Attention & Temporal Peak-Preserving Pooling (TP-Pool)
- **Validation Accuracy**: **94.59%** (Macro F1: **0.9367**)
- **Held-Out Test Accuracy**: **94.24%** (Macro F1: **0.9338**, OvR Macro ROC-AUC: **0.9824**)
- **Video-Level Accuracy**: **93.02%** (100% on authentic, 100% on insertion, 76.92% on deletion)
- **Temporal Localization**: Median error of **2.0 frames** (51.9% $\le 2$ frames; mean: 34.39 frames across 43 test videos) on UCF-101 synthetic manipulations

---

Current capabilities:
- **Dual-Stream R(2+1)D model**: Raw RGB appearance stream + Inter-frame motion difference stream with learned sigmoid gating.
- **Boundary-aware dataset preparation**: Automatically centers training clips on exact splice/deletion transitions with zero cross-split leakage.
- **Calibrated SSIM localization**: Dynamic drop calibration ($\tau_{\text{eff}} = \max(0.35, \min(0.85, \text{median} - 0.14))$) and artifact clustering.
- **Automated reporting & plotting**: Generates standardized JSON metrics, ROC/PR curves, and visual anomaly plots (`*_localization.png`).


## Initial layout

```text
src/forensic_vision/
  datasets/
  evaluation/
  localization/
  models/
  preprocessing/
  utils/
configs/
docs/
scripts/
```

## Next implementation steps

1. Build dataset preparation for UCF101 and VIFFD.
2. Add PyTorch dataset and dataloader classes for processed clips.
3. Wire training and inference scripts end to end.
4. Add localization evaluation and reproduction reporting.

## Dataset preparation

Place raw videos under `data/raw/`, then run:

```bash
python scripts/prepare_dataset.py
```

This will:

- discover supported raw video files
- split them into train/val/test
- generate authentic, insertion, and deletion samples
- save intermediate forged videos in `data/interim/`
- save grouped `.npy` clip tensors in `data/processed/`
- save a clip manifest in `data/manifests/clips_manifest.csv`

## Training

Train the baseline from the prepared manifest:

```bash
python scripts/train.py --config configs/base.yaml
```

This writes checkpoints to `outputs/checkpoints/`.

## Evaluation

Evaluate the saved checkpoint on the test split:

```bash
python scripts/evaluate.py --config configs/base.yaml
```

This writes metrics JSON and per-sample predictions under `outputs/reports/`.

## Inference And Localization

Run prediction and temporal localization on a single input video:

```bash
python scripts/infer_localize.py --config configs/base.yaml --video path/to/video.mp4
```

This writes a JSON report under `outputs/reports/` with:

- predicted class
- averaged clip probabilities
- per-clip predictions
- suspicious frame transitions from the localization step

## Paper assumptions

The source paper omits several implementation details. Assumptions, corrections, and benchmark results are tracked in [docs/assumptions.md](docs/assumptions.md), [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), and [reports/RESULTS_RECONCILIATION.md](reports/RESULTS_RECONCILIATION.md).
