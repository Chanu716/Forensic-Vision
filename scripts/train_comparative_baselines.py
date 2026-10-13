"""Train and Evaluate Comparative Neural Baselines for Forensic-Vision.

Implements and evaluates:
  Experiment C: ResNet-18 with Temporal Pooling (2D spatial CNN + temporal mean/max pooling)
  Experiment D: ResNet-18 + Bidirectional GRU (2D spatial CNN + 2-layer BiGRU temporal modeling)

Strictly trained from scratch without pretrained weights on the leak-free UCF-101 split.
Saves checkpoints, training history, predictions, and standardized evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.evaluation.metrics import compute_standardized_metrics
from forensic_vision.models.resnet_baselines import ResNet18BiGRU, ResNet18TemporalPooling
from forensic_vision.training import FocalLoss, build_dataloader, run_epoch
from forensic_vision.utils.repro import set_seed


def compute_class_weights(dataset: Any, class_names: list[str]) -> torch.Tensor:
    from collections import Counter

    label_counts = Counter(r["label"] for r in dataset.records)
    total_samples = len(dataset.records)
    num_classes = len(class_names)
    weights = [
        total_samples / (num_classes * max(label_counts.get(c, 1), 1))
        for c in class_names
    ]
    return torch.tensor(weights, dtype=torch.float32)


def train_and_eval_baseline(
    model_name: str,
    model: nn.Module,
    manifest_path: Path,
    class_names: list[str],
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"TRAINING {model_name.upper()} (From Scratch, Seed {seed})")
    print("=" * 70)
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        manifest_path=manifest_path,
        split="train",
        class_names=class_names,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        cache_in_memory=True,
    )
    val_loader = build_dataloader(
        manifest_path=manifest_path,
        split="val",
        class_names=class_names,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        cache_in_memory=True,
    )
    test_loader = build_dataloader(
        manifest_path=manifest_path,
        split="test",
        class_names=class_names,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        cache_in_memory=True,
    )

    class_weights = compute_class_weights(train_loader.dataset, class_names).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable parameters: {param_count:,}")
    print(f"[*] Training schedule: {epochs} epochs, initial lr={lr}, batch_size={batch_size}")

    best_val_f1 = float("-inf")
    best_val_loss = float("inf")
    best_chk_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_loss, train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
        )
        scheduler.step()
        ep_duration = time.time() - t_ep

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_metrics.accuracy*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics.accuracy*100:.2f}%, F1: {val_metrics.f1_macro:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics.accuracy,
            "train_f1_macro": train_metrics.f1_macro,
            "val_loss": val_loss,
            "val_accuracy": val_metrics.accuracy,
            "val_f1_macro": val_metrics.f1_macro,
            "duration_seconds": ep_duration,
        })

        # Save checkpoint based solely on validation performance
        if (val_metrics.f1_macro > best_val_f1) or (
            abs(val_metrics.f1_macro - best_val_f1) < 1e-4 and val_loss < best_val_loss
        ):
            best_val_f1 = val_metrics.f1_macro
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_metrics.accuracy,
                    "val_f1_macro": val_metrics.f1_macro,
                    "class_names": class_names,
                    "model_name": model_name,
                },
                best_chk_path,
            )

    train_time = time.time() - t_start
    print(f"[*] Training complete in {train_time/60:.2f} min. Best Val F1: {best_val_f1:.4f}")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Load best checkpoint for test evaluation
    print(f"[*] Loading best checkpoint for test evaluation: {best_chk_path}")
    checkpoint = torch.load(best_chk_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []
    all_sample_ids: list[str] = []

    t_infer_start = time.time()
    with torch.no_grad():
        for clips, labels, sample_ids in test_loader:
            clips = clips.to(device, non_blocking=True)
            with torch.amp.autocast(device.type):
                logits = model(clips)
                probs = F.softmax(logits, dim=1)
            all_targets.extend(labels.tolist())
            all_probs.append(probs.cpu().numpy())
            all_sample_ids.extend(sample_ids)

    infer_time = time.time() - t_infer_start
    y_true = np.array(all_targets, dtype=int)
    y_prob = np.concatenate(all_probs, axis=0)

    # Compute standardized publication-ready metrics
    report = compute_standardized_metrics(y_true, y_prob, class_names, num_bootstrap_samples=1000, seed=seed)
    (output_dir / "test_metrics.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # Save test predictions CSV with probabilities
    pred_csv = output_dir / "test_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id", "target_index", "target_label",
                "pred_index", "pred_label", "confidence",
                "prob_auth", "prob_ins", "prob_del",
            ],
        )
        writer.writeheader()
        preds = np.argmax(y_prob, axis=1)
        confs = np.max(y_prob, axis=1)
        for i, sid in enumerate(all_sample_ids):
            writer.writerow({
                "sample_id": sid,
                "target_index": int(y_true[i]),
                "target_label": class_names[y_true[i]],
                "pred_index": int(preds[i]),
                "pred_label": class_names[preds[i]],
                "confidence": float(confs[i]),
                "prob_auth": float(y_prob[i, 0]),
                "prob_ins": float(y_prob[i, 1]),
                "prob_del": float(y_prob[i, 2]),
            })

    print(
        f"[+] {model_name} Results on Test Set (N={len(y_true)}):\n"
        f"    Clip Accuracy: {report.accuracy*100:.2f}%\n"
        f"    Macro F1: {report.macro_f1:.4f}\n"
        f"    OvR Macro ROC-AUC: {report.macro_roc_auc_ovr:.4f}\n"
        f"    Deletion ROC-AUC: {report.per_class['frame_deletion']['roc_auc']:.4f}\n"
        f"    Insertion ROC-AUC: {report.per_class['frame_insertion']['roc_auc']:.4f}\n"
        f"    Inference time: {infer_time:.2f}s ({infer_time/len(y_true)*1000:.1f} ms/clip)"
    )

    summary_entry = {
        "model_name": model_name,
        "parameters": param_count,
        "epochs_trained": epochs,
        "best_epoch": checkpoint["epoch"],
        "training_time_seconds": train_time,
        "inference_time_ms_per_clip": infer_time / len(y_true) * 1000,
        "accuracy": report.accuracy,
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "macro_f1": report.macro_f1,
        "macro_roc_auc_ovr": report.macro_roc_auc_ovr,
        "macro_roc_auc_interp": report.macro_roc_auc_interpolated,
        "authentic_roc_auc": report.per_class["authentic"]["roc_auc"],
        "insertion_roc_auc": report.per_class["frame_insertion"]["roc_auc"],
        "deletion_roc_auc": report.per_class["frame_deletion"]["roc_auc"],
        "confusion_matrix": report.confusion_matrix,
        "ci_95": report.bootstrap_ci_95,
    }
    return summary_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate comparative neural baselines.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["resnet18_temporal_pool", "resnet18_bigru"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running comparative baselines on device: {device}")

    manifest_path = REPO_ROOT / "data/manifests/clips_manifest.csv"
    class_names = ["authentic", "frame_insertion", "frame_deletion"]
    base_out = REPO_ROOT / "outputs/comparative_baselines"
    reports_out = REPO_ROOT / "outputs/reports_comparative"
    base_out.mkdir(parents=True, exist_ok=True)
    reports_out.mkdir(parents=True, exist_ok=True)

    summary_results: list[dict[str, Any]] = []

    if "resnet18_temporal_pool" in args.models:
        m1 = ResNet18TemporalPooling(num_classes=3).to(device)
        res1 = train_and_eval_baseline(
            model_name="ResNet-18 + Temporal Pooling",
            model=m1,
            manifest_path=manifest_path,
            class_names=class_names,
            output_dir=base_out / "resnet18_temporal_pool",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        summary_results.append(res1)

    if "resnet18_bigru" in args.models:
        m2 = ResNet18BiGRU(num_classes=3).to(device)
        res2 = train_and_eval_baseline(
            model_name="ResNet-18 + Bidirectional GRU",
            model=m2,
            manifest_path=manifest_path,
            class_names=class_names,
            output_dir=base_out / "resnet18_bigru",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        summary_results.append(res2)

    # Save summary JSON
    sum_json = reports_out / "neural_comparative_baselines.json"
    sum_json.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"\n[+] Saved summary results to: {sum_json}")


if __name__ == "__main__":
    main()
