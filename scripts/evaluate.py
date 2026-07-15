from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.models.three_d_cnn import Forgery3DCNN
from forensic_vision.training import build_dataloader, collect_predictions
from forensic_vision.utils.repro import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Phase 1 3DCNN checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the evaluation configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Defaults to training.checkpoint_dir/best.pt.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate. Defaults to 'test'.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.values["experiment"]["seed"]
    set_seed(seed)

    data_cfg = config.values["data"]
    training_cfg = config.values["training"]
    model_cfg = config.values["model"]
    class_names = list(data_cfg["class_names"])
    manifest_path = Path(data_cfg["manifests_dir"]) / "clips_manifest.csv"

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first or pass --checkpoint."
        )

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

    dataloader = build_dataloader(
        manifest_path=manifest_path,
        split=args.split,
        class_names=class_names,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    loss, metrics, prediction_rows = collect_predictions(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    for row in prediction_rows:
        row["target_label"] = class_names[int(row["target_index"])]
        row["prediction_label"] = class_names[int(row["prediction_index"])]

    metrics_payload = {
        "split": args.split,
        "checkpoint_path": str(checkpoint_path),
        "loss": loss,
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "confusion_matrix": metrics.confusion.tolist(),
        "class_names": class_names,
        "num_samples": len(prediction_rows),
    }

    metrics_path = reports_dir / f"{args.split}_metrics.json"
    predictions_path = reports_dir / f"{args.split}_predictions.csv"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluating split: {args.split}")
    print(f"Samples: {len(prediction_rows)}")
    print(
        f"loss={loss:.4f} accuracy={metrics.accuracy:.4f} "
        f"precision_macro={metrics.precision_macro:.4f} "
        f"recall_macro={metrics.recall_macro:.4f} f1_macro={metrics.f1_macro:.4f}"
    )
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
