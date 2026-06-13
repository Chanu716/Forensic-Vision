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
2. Implement frame extraction and grouped sample generation.
3. Wire training and inference scripts end to end.
4. Add localization evaluation and reproduction reporting.

## Paper assumptions

The source paper omits several implementation details. Assumptions and deviations are tracked in [docs/assumptions.md](/d:/Forensic Vision/Forensic-Vision/docs/assumptions.md).
