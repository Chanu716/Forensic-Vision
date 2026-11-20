# VIFFD External Evaluation

This protocol tests a model trained only on UCF-101 against the public
[VIFFD test folder](https://data.mendeley.com/datasets/r3ss3v53sj/6) (version 6, CC BY 4.0).

## Commands

```powershell
python scripts/download_viffd.py --dry-run
python scripts/download_viffd.py
python scripts/evaluate_viffd.py --method ssim
python scripts/evaluate_viffd.py --method model --checkpoint outputs/checkpoints_enhanced/best.pt
```

Files are stored under `data/viffd/`, and reports under `outputs/reports_viffd/`.

## Selection

| Class | Files in folder | Evaluated |
| --- | ---: | ---: |
| Authentic | 30 | 29 |
| Frame insertion | 30 | 29 |
| Frame deletion | 30 | 30 |
| Frame duplication | 30 | 0 (out of scope) |

Labels come from file names because VIFFD has no label file. Two evaluated-class files are
byte-identical copies of other files and are removed. `viffd_test_exclusions.csv` records
every exclusion.

## Rules

- Do not train on VIFFD or tune any threshold, checkpoint or clip setting with it.
- Score every video with the same sliding window. Edit positions are never used to pick clips.
- Report `pipeline` as the primary decision and `clip_mean` as a secondary decision.
- Report by-video and source-clustered bootstrap intervals. Clusters join videos that share a
  recording, directly or through an insertion.

## Limitations

- Authentic files are small, while many forged AVI files are very large. Report the AVI and MP4
  results alongside the overall result, because a model could respond to encoding differences.
- Insertion positions are not recorded in file names, so insertion localization is not scored.
- Deletion localization assumes the numbers in file names are one-based frame indices. The
  "delete ... to black" file is excluded from localization.
- Results are not directly comparable with Gowda and Pawar's clip-group results (318 groups),
  because that paper does not give enough detail to reproduce its grouping.
