from __future__ import annotations

import argparse
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
from forensic_vision.models.three_d_cnn import Forgery3DCNN
from forensic_vision.preprocessing.video_io import read_video_frames
from forensic_vision.utils.repro import set_seed


def prepare_clip_tensor(clips: list[np.ndarray]) -> torch.Tensor:
    clip_array = np.stack(clips, axis=0)
    return torch.from_numpy(clip_array).permute(0, 4, 1, 2, 3).float() / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-video inference and localization for the Phase 1 baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Defaults to training.checkpoint_dir/best.pt.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Optional override for inference clip stride. Defaults to data.clip_stride.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.values["experiment"]["seed"]
    set_seed(seed)

    data_cfg = config.values["data"]
    training_cfg = config.values["training"]
    model_cfg = config.values["model"]
    localization_cfg = config.values["localization"]
    class_names = list(data_cfg["class_names"])

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first or pass --checkpoint."
        )

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    reports_dir = Path(training_cfg.get("reports_dir", "outputs/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    device_name = training_cfg.get("device")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = Forgery3DCNN(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        dropout=model_cfg["dropout"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frames, fps = read_video_frames(video_path, image_size=tuple(data_cfg["image_size"]))
    clip_length = int(data_cfg["clip_length"])
    clip_stride = int(args.stride if args.stride is not None else data_cfg["clip_stride"])
    clips = sliding_windows(frames, clip_length=clip_length, stride=clip_stride)
    if not clips:
        raise ValueError(
            f"Video has {len(frames)} frames after resize, which is fewer than clip_length={clip_length}."
        )

    clip_tensor = prepare_clip_tensor(clips).to(device)
    with torch.no_grad():
        logits = model(clip_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probabilities = probabilities.mean(axis=0)
    predicted_index = int(mean_probabilities.argmax())
    predicted_label = class_names[predicted_index]

    localization = compute_frame_msssim_scores(
        frames=frames,
        threshold=float(localization_cfg["threshold"]),
    )

    clip_predictions = []
    for clip_index, probs in enumerate(probabilities):
        start_frame = clip_index * clip_stride
        end_frame = start_frame + clip_length - 1
        clip_predictions.append(
            {
                "clip_index": clip_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "predicted_label": class_names[int(probs.argmax())],
                "confidence": float(probs.max()),
                "probabilities": {
                    class_names[class_index]: float(probability)
                    for class_index, probability in enumerate(probs.tolist())
                },
            }
        )

    report = {
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "fps": fps,
        "num_frames": int(len(frames)),
        "clip_length": clip_length,
        "clip_stride": clip_stride,
        "num_clips": len(clips),
        "predicted_label": predicted_label,
        "predicted_confidence": float(mean_probabilities[predicted_index]),
        "mean_probabilities": {
            class_names[class_index]: float(probability)
            for class_index, probability in enumerate(mean_probabilities.tolist())
        },
        "localization_threshold": float(localization_cfg["threshold"]),
        "suspicious_frame_indices": localization.suspicious_indices,
        "localization_scores": localization.scores,
        "clip_predictions": clip_predictions,
    }

    report_path = reports_dir / f"{video_path.stem}_inference.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Video: {video_path}")
    print(f"Frames: {len(frames)} | Clips: {len(clips)} | FPS: {fps:.2f}")
    print(
        f"Predicted label: {predicted_label} "
        f"(confidence={float(mean_probabilities[predicted_index]):.4f})"
    )
    print(
        f"Suspicious frame transitions below threshold {float(localization_cfg['threshold']):.2f}: "
        f"{len(localization.suspicious_indices)}"
    )
    print(f"Saved inference report to: {report_path}")


if __name__ == "__main__":
    main()
