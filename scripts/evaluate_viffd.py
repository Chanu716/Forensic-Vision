"""Evaluate a UCF-101-trained model or standalone SSIM on the VIFFD test videos.

Every clip of every video is scored with the same sliding window, so ground-truth
edit positions never influence clip selection. Decision thresholds stay fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.datasets.preparation import sliding_windows
from forensic_vision.evaluation.video_level import decide_video_label, summarize_video_predictions
from forensic_vision.localization.msssim import compute_frame_msssim_scores
from forensic_vision.models.three_d_cnn import build_model
from forensic_vision.preprocessing.video_io import read_video_frames
from forensic_vision.utils.repro import set_seed

CLASS_NAMES = ["authentic", "frame_insertion", "frame_deletion"]
ClipPredictor = Callable[[list[np.ndarray]], np.ndarray]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no videos: {path}")
    for row in rows:
        row["cluster_id"] = int(row["cluster_id"])
        row["gt_start"] = int(row["gt_start"]) if row["gt_start"] else None
        row["gt_end"] = int(row["gt_end"]) if row["gt_end"] else None
        row["localization_eligible"] = row["localization_eligible"] == "True"
        row["video_path"] = path.parent / row["local_path"]
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_clip_predictor(model: torch.nn.Module, device: torch.device, batch_size: int) -> ClipPredictor:
    def predict(clips: list[np.ndarray]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                batch = np.stack(clips[start : start + batch_size], axis=0)
                tensor = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).float().div(255.0).to(device)
                outputs.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    return predict


def evaluate_videos(
    videos: list[dict[str, Any]],
    method: str,
    image_size: tuple[int, int],
    clip_length: int,
    clip_stride: int,
    localization_cfg: dict[str, Any],
    predictor: ClipPredictor | None = None,
) -> list[dict[str, Any]]:
    if method == "model" and predictor is None:
        raise ValueError("The model method requires a clip predictor.")

    rows: list[dict[str, Any]] = []
    for position, video in enumerate(videos, start=1):
        row: dict[str, Any] = {
            "video_id": video["video_id"],
            "filename": video["filename"],
            "ground_truth": video["label"],
            "cluster_id": video["cluster_id"],
            "status": "evaluated",
            "num_frames": None,
            "num_clips": None,
            "predicted_pipeline": None,
            "predicted_clip_mean": None,
            "score_authentic": None,
            "score_insertion": None,
            "score_deletion": None,
            "localized_type": None,
            "detected_start": None,
            "detected_end": None,
            "gt_start": video["gt_start"],
            "deletion_error_frames": None,
        }
        try:
            frames, _ = read_video_frames(video["video_path"], image_size=image_size)
        except Exception as exc:
            row["status"] = f"unreadable: {exc}"
            rows.append(row)
            continue

        row["num_frames"] = len(frames)
        localization = compute_frame_msssim_scores(
            frames=frames,
            threshold=float(localization_cfg.get("threshold", 0.85)),
            use_adaptive_threshold=bool(localization_cfg.get("use_adaptive_threshold", True)),
            adaptive_sensitivity=float(localization_cfg.get("adaptive_sensitivity", 3.5)),
            use_multiscale=bool(localization_cfg.get("use_multiscale", False)),
        )
        row["localized_type"] = localization.forgery_type
        row["detected_start"] = localization.start_frame
        row["detected_end"] = localization.end_frame
        if video["localization_eligible"] and localization.start_frame is not None:
            row["deletion_error_frames"] = abs(localization.start_frame - video["gt_start"])

        if method == "ssim":
            row["predicted_pipeline"] = localization.forgery_type
        else:
            clips = sliding_windows(frames, clip_length=clip_length, stride=clip_stride)
            if not clips:
                row["status"] = "too_short"
                rows.append(row)
                continue
            probabilities = predictor(clips)  # type: ignore[misc]
            mean_probabilities = probabilities.mean(axis=0)
            row["num_clips"] = len(clips)
            row["predicted_pipeline"] = decide_video_label(probabilities, localization.forgery_type)
            row["predicted_clip_mean"] = CLASS_NAMES[int(mean_probabilities.argmax())]
            row["score_authentic"] = float(mean_probabilities[0])
            row["score_insertion"] = float(mean_probabilities[1])
            row["score_deletion"] = float(mean_probabilities[2])

        rows.append(row)
        print(f"[{position}/{len(videos)}] {video['label']} -> {row['predicted_pipeline']}", flush=True)
    return rows


def build_report(rows: list[dict[str, Any]], method: str, seed: int) -> dict[str, Any]:
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    y_true = [row["ground_truth"] for row in evaluated]
    clusters = [row["cluster_id"] for row in evaluated]

    scores = None
    if method == "model":
        scores = np.array(
            [[row["score_authentic"], row["score_insertion"], row["score_deletion"]] for row in evaluated],
            dtype=float,
        )

    decisions = {
        "pipeline": summarize_video_predictions(
            y_true,
            [row["predicted_pipeline"] for row in evaluated],
            CLASS_NAMES,
            clusters,
            scores=scores,
            seed=seed,
        )
    }
    if method == "model":
        decisions["clip_mean"] = summarize_video_predictions(
            y_true,
            [row["predicted_clip_mean"] for row in evaluated],
            CLASS_NAMES,
            clusters,
            scores=scores,
            seed=seed,
        )

    # Authentic and forged files differ strongly in size, so container results are kept separate.
    by_container: dict[str, dict[str, Any]] = {}
    for suffix in sorted({Path(row["filename"].strip()).suffix.lower() for row in evaluated}):
        subset = [row for row in evaluated if Path(row["filename"].strip()).suffix.lower() == suffix]
        by_container[suffix.lstrip(".")] = summarize_video_predictions(
            [row["ground_truth"] for row in subset],
            [row["predicted_pipeline"] for row in subset],
            CLASS_NAMES,
            [row["cluster_id"] for row in subset],
            seed=seed,
        )

    errors = [row["deletion_error_frames"] for row in rows if row["deletion_error_frames"] is not None]
    return {
        "method": method,
        "pipeline_by_container": by_container,
        "num_manifest_videos": len(rows),
        "num_evaluated_videos": len(evaluated),
        "not_evaluated": [
            {"filename": row["filename"], "status": row["status"]}
            for row in rows
            if row["status"] != "evaluated"
        ],
        "decisions": decisions,
        "deletion_localization": {
            "num_detected": len(errors),
            "median_error_frames": float(np.median(errors)) if errors else None,
            "mean_error_frames": float(np.mean(errors)) if errors else None,
            "note": "VIFFD file names do not record insertion positions, so insertions are not scored.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("model", "ssim"), default="model")
    parser.add_argument("--config", default="configs/enhanced.yaml")
    parser.add_argument("--checkpoint", default=None, help="Defaults to training.checkpoint_dir/best.pt.")
    parser.add_argument("--manifest", default="data/viffd/viffd_test_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports_viffd")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N videos (smoke test).")
    parser.add_argument(
        "--arch",
        default=None,
        help="Build this architecture with library defaults, as the ablation study did (e.g. variant_a, variant_e).",
    )
    args = parser.parse_args()

    config = load_config(args.config, warn=False).values
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    data_cfg = config["data"]
    model_cfg = config["model"]
    if list(data_cfg["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Expected class order {CLASS_NAMES}, got {data_cfg['class_names']}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/download_viffd.py first.")
    videos = load_manifest(manifest_path)
    if args.limit is not None:
        videos = videos[: args.limit]
    missing = [video["filename"] for video in videos if not video["video_path"].exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest videos are missing, for example {missing[0]!r}.")

    predictor: ClipPredictor | None = None
    checkpoint_path: Path | None = None
    if args.method == "model":
        checkpoint_dir = Path(config["training"].get("checkpoint_dir", "outputs/checkpoints_enhanced"))
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train on UCF-101 first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.arch:
            model = build_model(arch=args.arch).to(device)
        else:
            model = build_model(
                arch=model_cfg.get("arch", "dual_stream_r2plus1d"),
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
        predictor = make_clip_predictor(model, device, args.batch_size)

    rows = evaluate_videos(
        videos=videos,
        method=args.method,
        image_size=tuple(data_cfg["image_size"]),
        clip_length=int(data_cfg["clip_length"]),
        clip_stride=int(data_cfg["clip_stride"]),
        localization_cfg=config.get("localization", {}),
        predictor=predictor,
    )
    report = build_report(rows, args.method, seed)
    report["provenance"] = {
        "config": args.config,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path else None,
        "arch": args.arch,
        "limit": args.limit,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"viffd_{args.method}_metrics.json"
    predictions_path = output_dir / f"viffd_{args.method}_predictions.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pipeline = report["decisions"]["pipeline"]
    print(f"Accuracy: {pipeline['accuracy']:.4f} ({pipeline['correct']}/{pipeline['num_videos']} videos)")
    print(f"Macro-F1: {pipeline['macro_f1']:.4f}")
    print(f"Saved: {report_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()
