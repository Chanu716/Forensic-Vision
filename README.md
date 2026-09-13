# Forensic Vision

Deep learning-based video forgery identification and exact temporal localization, reproducing and significantly outperforming the reference paper:
> Raghavendra Gowda & Digambar Pawar (2023), *Deep Learning-Based Forgery Identification and Localization in Videos*, Signal, Image and Video Processing.

### 🏆 Benchmark Highlights
- **Detailed Results & Comparison**: See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
- **Architecture**: Dual-Stream R(2+1)D with CBAM Spatial & Channel Attention
- **Validation Accuracy**: **92.43%** (Macro F1: **0.9157**)
- **Frame Insertion Recall**: **98.25%** (100% precision vs authentic)
- **Temporal Localization Error**: **$\le 1$ frame** of ground truth on UCF-101

---

Current capabilities:
- **Dual-Stream R(2+1)D model**: Raw RGB appearance stream + Inter-frame motion difference stream with learned sigmoid gating.
- **Boundary-aware dataset preparation**: Automatically centers training clips on exact splice/deletion transitions.
- **Calibrated MS-SSIM localization**: Dynamic drop sensitivity and edge artifact filtering for pinpoint temporal boundary detection.
- **Automated reporting & plotting**: Generates JSON metrics and visual anomaly plots (`*_localization.png`).

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

The source paper omits several implementation details. Assumptions and deviations are tracked in [docs/assumptions.md](/d:/Forensic Vision/Forensic-Vision/docs/assumptions.md).
