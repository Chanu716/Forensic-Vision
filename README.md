# Forensic Vision

This repository contains the Phase 1 reproduction baseline for the paper `Deep learning-based forgery identification and localization in videos`.

Current status:

- project scaffold created
- baseline modules added for:
  - configuration
  - reproducibility utilities
  - absolute frame differencing
  - 3DCNN classification model
  - MS-SSIM localization
  - evaluation metrics
- dataset preparation pipeline added for:
  - raw video discovery
  - reproducible split generation
  - insertion/deletion forgery generation
  - grouped clip export
  - manifest export

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

## Paper assumptions

The source paper omits several implementation details. Assumptions and deviations are tracked in [docs/assumptions.md](/d:/Forensic Vision/Forensic-Vision/docs/assumptions.md).
