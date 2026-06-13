# Phase 1 Assumptions

This file records assumptions required to rebuild the paper without the original implementation.

## Current assumptions

1. Input frame size is set to `112x112` as a practical 3DCNN baseline pending dataset experiments.
2. The paper states a group size of `49 frames`; this is used as the default clip length.
3. The exact Conv3D channel sizes are not provided in the paper. The baseline uses `[32, 64, 128]`.
4. The paper mentions Adam optimization, learning rate `0.001`, weight decay `0.0005`, batch size `16`, and `40` epochs. These are used directly where applicable.
5. The absolute-difference preprocessing is implemented as frame-wise differencing and exposed as a configurable preprocessing step so we can run the required ablation.
6. The localization threshold defaults to `0.8` as described in the paper, but remains configurable for validation.

## Pending assumptions

1. Exact UCF101 sampling strategy for selecting source videos.
2. Exact forgery generation rules for insertion and deletion ranges.
3. Exact train/validation/test split used by the authors.
4. Exact input normalization and codec choices beyond the paper’s description.
