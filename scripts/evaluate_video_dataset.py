"""Evaluate trained model and localization on full uncropped test videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.datasets.preparation import sliding_windows
from forensic_vision.localization.msssim import compute_frame_msssim_scores
from forensic_vision.models.three_d_cnn import build_model
from forensic_vision.preprocessing.video_io import read_video_frames
from forensic_vision.utils.repro import set_seed


def prepare_clip_tensor(clips: list[np.ndarray]) -> torch.Tensor:
    clip_array = np.stack(clips, axis=0)
    return torch.from_numpy(clip_array).permute(0, 4, 1, 2, 3).float() / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video-level classification and localization.")
    parser.add_argument("--config", default="configs/enhanced.yaml", help="Path to configuration file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.values["experiment"]["seed"])

    data_cfg = config.values["data"]
    training_cfg = config.values["training"]
    model_cfg = config.values["model"]
    localization_cfg = config.values["localization"]
    class_names = list(data_cfg["class_names"])

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "outputs/checkpoints_enhanced"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else checkpoint_dir / "best.pt"
    reports_dir = Path(training_cfg.get("reports_dir", "outputs/reports_enhanced"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = model_cfg.get("arch", "dual_stream_r2plus1d")
    model = build_model(
        arch=arch,
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        dropout=model_cfg["dropout"],
        use_frame_difference=model_cfg.get("use_frame_difference", True),
        diff_threshold=model_cfg.get("diff_threshold", None),
        use_cbam=model_cfg.get("use_cbam", True),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest_path = Path(data_cfg["manifests_dir"]) / "clips_manifest.csv"
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Extract distinct test videos with ground truth
    test_rows = [r for r in reader if r["split"] == "test"]
    video_gt: dict[str, dict] = {}
    for r in test_rows:
        label = r["label"]
        if label == "authentic":
            vpath = Path(r["source_video"])
            key = vpath.stem
            if key not in video_gt:
                video_gt[key] = {
                    "video_path": vpath,
                    "label": "authentic",
                    "gt_start": None,
                    "gt_end": None,
                }
        elif label == "frame_insertion":
            vpath = Path("data/interim/test/frame_insertion") / f"{Path(r['source_video']).stem}_insert.mp4"
            key = vpath.stem
            if key not in video_gt and vpath.exists():
                video_gt[key] = {
                    "video_path": vpath,
                    "label": "frame_insertion",
                    "gt_start": float(r["forgery_start"]) if r["forgery_start"] else None,
                    "gt_end": float(r["forgery_end"]) if r["forgery_end"] else None,
                }
        elif label == "frame_deletion":
            vpath = Path("data/interim/test/frame_deletion") / f"{Path(r['source_video']).stem}_delete.mp4"
            key = vpath.stem
            if key not in video_gt and vpath.exists():
                video_gt[key] = {
                    "video_path": vpath,
                    "label": "frame_deletion",
                    "gt_start": float(r["forgery_start"]) if r["forgery_start"] else None,
                    "gt_end": float(r["forgery_end"]) if r["forgery_end"] else None,
                }

    print(f"Loaded {len(video_gt)} unique test videos from test manifest.")
    print("Evaluating video-level classification and temporal localization...")

    image_size = tuple(data_cfg["image_size"])
    clip_length = int(data_cfg["clip_length"])
    clip_stride = int(data_cfg["clip_stride"])

    results = []
    correct_count = 0
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    localization_errors = []

    for vid_key, info in video_gt.items():
        vpath = info["video_path"]
        gt_label = info["label"]
        gt_idx = label_to_idx[gt_label]

        try:
            frames, fps = read_video_frames(vpath, image_size=image_size)
        except Exception as exc:
            print(f"Skipping unreadable video {vpath}: {exc}")
            continue

        clips = sliding_windows(frames, clip_length=clip_length, stride=clip_stride)
        if not clips:
            continue

        clip_tensor = prepare_clip_tensor(clips).to(device)
        with torch.no_grad():
            logits = model(clip_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        mean_probs = probabilities.mean(axis=0)
        naive_pred_idx = int(mean_probs.argmax())
        naive_label = class_names[naive_pred_idx]

        localization = compute_frame_msssim_scores(
            frames=frames,
            threshold=float(localization_cfg.get("threshold", 0.8)),
            use_adaptive_threshold=bool(localization_cfg.get("use_adaptive_threshold", True)),
            adaptive_sensitivity=float(localization_cfg.get("adaptive_sensitivity", 2.5)),
            use_multiscale=bool(localization_cfg.get("use_multiscale", True)),
        )

        max_insertion = float(probabilities[:, 1].max()) if len(probabilities) > 0 else 0.0
        max_deletion = float(probabilities[:, 2].max()) if len(probabilities) > 0 else 0.0

        if localization.forgery_type == "frame_insertion" and (max_insertion > 0.3 or len(localization.suspicious_indices) >= 2):
            pred_label = "frame_insertion"
        elif localization.forgery_type == "frame_deletion" and (max_deletion > 0.3 or len(localization.suspicious_indices) >= 1):
            pred_label = "frame_deletion"
        elif max_insertion > 0.7:
            pred_label = "frame_insertion"
        elif max_deletion > 0.7:
            pred_label = "frame_deletion"
        else:
            pred_label = naive_label

        pred_idx = label_to_idx[pred_label]
        is_correct = (pred_idx == gt_idx)
        if is_correct:
            correct_count += 1
        confusion_matrix[gt_idx, pred_idx] += 1

        loc_err = None
        if gt_label == "frame_insertion" and localization.start_frame is not None and info["gt_start"] is not None:
            err_start = abs(localization.start_frame - info["gt_start"])
            err_end = abs(localization.end_frame - info["gt_end"]) if (localization.end_frame and info["gt_end"]) else 0.0
            loc_err = float((err_start + err_end) / 2.0)
            localization_errors.append(loc_err)

        results.append({
            "video": vid_key,
            "ground_truth": gt_label,
            "predicted": pred_label,
            "naive_mean_predicted": naive_label,
            "is_correct": is_correct,
            "localized_type": localization.forgery_type,
            "detected_start": localization.start_frame,
            "detected_end": localization.end_frame,
            "gt_start": info["gt_start"],
            "gt_end": info["gt_end"],
            "loc_error_frames": loc_err,
        })

    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    mean_loc_err = float(np.mean(localization_errors)) if localization_errors else 0.0

    print("\n========================================================")
    print(f"Video-Level Evaluation Summary ({total} videos):")
    print(f"  Overall Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total})")
    print(f"  Confusion Matrix (rows: GT, cols: Pred):")
    for i, row in enumerate(confusion_matrix):
        print(f"    {class_names[i]:<16}: {row.tolist()}")
    if localization_errors:
        print(f"  Mean Localization Boundary Error: {mean_loc_err:.2f} frames")
    print("========================================================\n")

    summary = {
        "num_videos": total,
        "correct": correct_count,
        "video_level_accuracy": accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": class_names,
        "mean_localization_error_frames": mean_loc_err,
        "detailed_results": results,
    }
    out_path = reports_dir / "video_level_test_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved video-level evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
