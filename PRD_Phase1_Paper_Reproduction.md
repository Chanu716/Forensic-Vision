# Product Requirements Document

## Title

Reproduction of "Deep Learning-Based Forgery Identification and Localization in Videos"

## Version

`v1.0`

## Date

`2026-06-13`

## Owner

Forensic Vision Research Team

## Status

Draft

## 1. Purpose

This PRD defines the requirements for **Phase 1** of the project: a faithful, research-oriented reimplementation of the paper **"Deep Learning-Based Forgery Identification and Localization in Videos"** by Raghavendra Gowda and Digambar Pawar.

The immediate objective is to rebuild the complete inter-frame forgery detection and localization pipeline from scratch, without access to the original implementation or repository, and reproduce the reported results as closely as possible.

This phase is limited to:

- inter-frame forgery detection
- frame insertion detection
- frame deletion detection
- localization of inserted/deleted temporal regions

This phase explicitly excludes:

- random frame manipulation extensions beyond the baseline
- duplication and shuffling extensions
- intra-frame/object-level forgery analysis
- architectural improvements not needed for baseline reproduction

## 2. Problem Statement

The paper presents a pipeline for detecting and localizing inter-frame video forgery using:

- absolute difference preprocessing
- a 3D CNN classifier
- MS-SSIM-based localization

However, the original source code is unavailable. To establish a credible research baseline, we need a reproducible implementation that reconstructs the methodology, documents missing details, and validates whether the reported performance can be matched under transparent engineering assumptions.

## 3. Goal

Rebuild the complete Phase 1 pipeline described in the paper and produce a reproducible baseline implementation that:

- follows the paper’s methodology as closely as possible
- uses the datasets referenced in the paper where accessible
- achieves classification performance comparable to the reported result of approximately `98% accuracy`
- supports detection and localization of frame insertion and frame deletion
- is modular and suitable for future research extensions

## 4. Success Criteria

The project will be considered successful for Phase 1 when all of the following are true:

- a complete end-to-end pipeline exists for dataset preparation, training, inference, localization, and evaluation
- the implementation is runnable from a clean project setup with documented steps
- all assumptions and deviations from the paper are recorded
- evaluation reproduces the paper’s task definition:
  - authentic
  - frame insertion forgery
  - frame deletion forgery
- metrics include:
  - accuracy
  - precision
  - recall
  - F1-score
- localization results are produced using MS-SSIM and validated qualitatively and quantitatively where possible
- the final baseline performance is within a reasonable reproduction band of the paper’s result

## 5. Target Outcome

Primary target:

- reproduce the reported performance as closely as possible, targeting `~98% accuracy`

Secondary target:

- establish a stable and auditable baseline for Phase 2 research extensions

## 6. Scope

### In Scope

- reproduction of the inter-frame forgery detection pipeline
- recreation of the frame preprocessing pipeline
- implementation of the absolute difference module
- implementation of the 3DCNN architecture
- training and validation pipeline
- inference pipeline
- MS-SSIM-based localization
- evaluation scripts
- result comparison against the paper
- reproducibility report
- modular project structure

### Out of Scope

- random frame insertion/deletion modeling beyond the paper baseline
- frame-wise anomaly scoring methods beyond paper reproduction
- optical flow based extensions
- Bi-LSTM or transformer temporal models
- object insertion/removal detection
- YOLO, ByteTrack, or intra-frame analysis

## 7. Users and Stakeholders

Primary users:

- internal research engineers
- ML engineers reproducing the paper
- future contributors extending the baseline

Stakeholders:

- project owner
- research team
- downstream teams working on Phase 2 and Phase 3

## 8. Research Questions

The Phase 1 implementation must answer the following:

1. Can the paper’s reported performance be reproduced from the textual description alone?
2. How sensitive is the result to implementation assumptions not specified in the paper?
3. Does the absolute difference preprocessing materially improve classification accuracy?
4. How reliable is MS-SSIM for localizing insertion and deletion boundaries?
5. How well does the baseline generalize across UCF101-derived forged data and VIFFD?

## 9. Functional Requirements

### FR1. Dataset Preparation Pipeline

The system must:

- ingest UCF101 videos
- ingest VIFFD videos if available and usable
- generate forged videos for:
  - frame insertion
  - frame deletion
- preserve metadata for:
  - source video
  - forgery type
  - inserted/deleted frame range
  - codec/compression settings
  - train/validation/test split
- support reproducible dataset generation with fixed random seeds

### FR2. Frame Extraction and Preprocessing

The system must:

- extract ordered video frames from source videos
- normalize frame dimensions and color format consistently
- support grouping videos into fixed-size temporal windows
- implement the paper’s group-of-frames logic with a configurable default of `49 frames`
- store preprocessing outputs in a reusable format for training and evaluation

### FR3. Absolute Difference Module

The system must:

- compute pixel-wise absolute difference between consecutive frames
- support thresholding or binarization if used in the final implementation
- make this step configurable for ablation studies
- expose outputs suitable as model input

### FR4. 3DCNN Model

The system must:

- implement a 3D CNN classifier aligned with the paper description
- support 3 classes:
  - authentic
  - insertion forgery
  - deletion forgery
- use a modular architecture so layer shapes and hyperparameters can be inspected and adjusted
- support training from scratch

### FR5. Training Pipeline

The system must:

- support training on prepared grouped video samples
- support validation during training
- log:
  - loss
  - accuracy
  - precision
  - recall
  - F1-score where appropriate
- save checkpoints
- save final model artifacts
- allow deterministic or near-deterministic reruns where feasible

### FR6. Inference Pipeline

The system must:

- load trained checkpoints
- run prediction on unseen video clips
- output predicted class labels and confidence scores
- support batch evaluation across test sets

### FR7. MS-SSIM Localization Module

The system must:

- compute MS-SSIM between consecutive frames
- support threshold-based temporal anomaly detection
- use a configurable threshold with paper-aligned default `0.8`
- generate localization outputs indicating suspected forged frame regions
- support visual outputs for insertion and deletion boundary analysis

### FR8. Evaluation and Metrics

The system must:

- compute classification metrics:
  - accuracy
  - precision
  - recall
  - F1-score
- produce confusion matrices
- support ablation comparison:
  - with absolute difference
  - without absolute difference
- generate a result summary aligned with the paper’s reported tables

### FR9. Reproducibility Report

The project must include a report documenting:

- implementation assumptions
- deviations from the paper
- dataset decisions
- hyperparameter choices
- achieved results
- comparison with paper-reported results
- observed failure cases and limitations

## 10. Non-Functional Requirements

### Reproducibility

- all experiments must be configurable via files or CLI arguments
- random seeds must be tracked
- dataset splits must be saved
- model versions and experiment outputs must be logged

### Modularity

- preprocessing, modeling, localization, and evaluation must be separated cleanly
- code should be suitable for future Phase 2 extensions without major restructuring

### Research Transparency

- all paper ambiguities must be explicitly documented
- no hidden preprocessing or undocumented tuning should be introduced

### Maintainability

- code must be readable and organized by responsibility
- configuration should be centralized

## 11. Assumptions

Because the paper does not provide complete implementation details, the following assumptions are allowed and must be documented:

- exact frame resizing dimensions may need to be chosen empirically
- exact Conv3D channel counts, kernel sizes, and dense dimensions may need to be inferred
- exact optimizer settings beyond what is stated may require reasonable defaults
- exact train/validation/test splits may need to be reconstructed
- some dataset preparation details for forged sample generation may need engineering interpretation
- VIFFD availability and structure may differ from the paper’s description

Every assumption must be recorded in the reproducibility report.

## 12. Constraints

- no original codebase is available
- paper details are incomplete
- dataset access may require adaptation depending on current availability
- reported performance may depend on hidden preprocessing decisions not disclosed in the paper

## 13. Risks

### Risk 1. Paper Ambiguity

The paper omits details required for exact reproduction.

Mitigation:

- document inferred choices
- isolate assumptions in configuration
- run ablations where uncertainty is high

### Risk 2. Dataset Mismatch

Current versions of UCF101/VIFFD or derived subsets may not match the paper exactly.

Mitigation:

- version datasets explicitly
- preserve split manifests
- compare under both paper-like and practical settings

### Risk 3. Performance Gap

Exact paper accuracy may not be reproducible.

Mitigation:

- measure gap transparently
- isolate likely causes
- validate whether relative behavior matches the paper even when absolute performance differs

### Risk 4. Localization Evaluation Ambiguity

The paper describes localization behavior but may not fully specify quantitative localization evaluation.

Mitigation:

- provide qualitative visualizations
- define an explicit localization evaluation protocol if needed

## 14. Proposed Deliverables

1. Dataset preparation pipeline
2. Frame extraction and preprocessing pipeline
3. Absolute difference module
4. 3DCNN training and inference pipeline
5. MS-SSIM localization module
6. Evaluation scripts and metrics
7. Reproducibility report comparing our results with the paper

## 15. Recommended Project Structure

```text
forensic_vision/
  configs/
  data/
    raw/
    processed/
    manifests/
  docs/
  notebooks/
  src/
    datasets/
    preprocessing/
    models/
    training/
    inference/
    localization/
    evaluation/
    utils/
  scripts/
  experiments/
  outputs/
    checkpoints/
    logs/
    reports/
    figures/
  tests/
```

## 16. Milestones

### Milestone 1. Paper Reconstruction Plan

- extract all implementable details from the paper
- list ambiguities and engineering assumptions
- finalize baseline design spec

### Milestone 2. Dataset Pipeline

- acquire datasets
- generate forged samples
- produce manifests and splits

### Milestone 3. Preprocessing and Model Baseline

- implement frame extraction
- implement absolute difference
- implement 3DCNN
- verify sample tensors and shapes

### Milestone 4. Training and Evaluation

- train baseline model
- evaluate on test data
- compute metrics
- run ablation with and without difference module

### Milestone 5. Localization

- implement MS-SSIM localization
- generate qualitative plots
- compare insertion vs deletion patterns

### Milestone 6. Reproducibility Report

- summarize assumptions
- summarize final results
- compare against paper
- list gaps and next steps for Phase 2

## 17. Acceptance Criteria

Phase 1 is complete when:

- the project can be run end-to-end from dataset preparation through evaluation
- model training and inference are reproducible from documented commands
- classification results are reported on the defined test data
- localization outputs are generated for insertion and deletion cases
- assumptions and deviations are documented clearly
- a final reproduction report is produced

## 18. Phase Boundary

After Phase 1 is complete and validated, only then should work proceed to:

- random frame insertion/deletion detection
- frame-wise localization improvements
- advanced temporal modeling
- intra-frame/object-level forgery research

No Phase 2 or Phase 3 work should begin until the Phase 1 baseline is stable and benchmarked.

## 19. Final Statement

This PRD is intended to ensure that the first implementation effort remains disciplined, research-faithful, and auditable. The priority is not novelty at this stage. The priority is to reconstruct the original paper as accurately as possible, establish a trustworthy baseline, and create a foundation for future extensions.
