from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.evaluation.metrics import ClassificationMetrics
from forensic_vision.models.three_d_cnn import Forgery3DCNN
from forensic_vision.training import build_dataloader, run_epoch
from forensic_vision.utils.repro import set_seed


def compute_class_weights(dataset: object, class_names: list[str]) -> torch.Tensor:
    records = getattr(dataset, "records", None)
    if records is None:
        raise AttributeError("Training dataset does not expose records for class weighting.")

    label_counts = Counter(record["label"] for record in records)
    total_samples = sum(label_counts.values())
    num_classes = len(class_names)
    weights = []

    for class_name in class_names:
        count = label_counts.get(class_name, 0)
        if count == 0:
            raise ValueError(
                f"Cannot compute class weight because class '{class_name}' has no training samples."
            )
        weights.append(total_samples / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Adam,
    train_loss: float,
    val_loss: float,
    val_metrics: ClassificationMetrics,
    class_names: list[str],
    config_path: str,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_metrics.accuracy,
        "val_precision_macro": val_metrics.precision_macro,
        "val_recall_macro": val_metrics.recall_macro,
        "val_f1_macro": val_metrics.f1_macro,
        "val_confusion": val_metrics.confusion.tolist(),
        "class_names": class_names,
        "config_path": config_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 1 3DCNN baseline.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.values["experiment"]["seed"]
    set_seed(seed)

    data_cfg = config.values["data"]
    training_cfg = config.values["training"]
    model_cfg = config.values["model"]
    class_names = list(data_cfg["class_names"])
    manifests_dir = Path(data_cfg["manifests_dir"])
    manifest_path = manifests_dir / "clips_manifest.csv"
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = build_dataloader(
        manifest_path=manifest_path,
        split="train",
        class_names=class_names,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        manifest_path=manifest_path,
        split="val",
        class_names=class_names,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=False,
    )

    class_weights = compute_class_weights(train_loader.dataset, class_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    history: list[dict[str, float]] = []
    best_val_accuracy = float("-inf")

    print(f"Loaded configuration: {args.config}")
    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics.accuracy,
            "train_f1_macro": train_metrics.f1_macro,
            "val_loss": val_loss,
            "val_accuracy": val_metrics.accuracy,
            "val_f1_macro": val_metrics.f1_macro,
        }
        history.append(summary)

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.f1_macro:.4f}"
        )

        latest_payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            class_names=class_names,
            config_path=args.config,
        )
        torch.save(latest_payload, checkpoint_dir / "latest.pt")

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            torch.save(latest_payload, checkpoint_dir / "best.pt")

    history_path = checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoints to: {checkpoint_dir}")
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
