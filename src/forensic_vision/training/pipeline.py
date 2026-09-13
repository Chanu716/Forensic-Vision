from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import torch.nn.functional as F
from forensic_vision.datasets import ForgeryClipDataset
from forensic_vision.evaluation.metrics import ClassificationMetrics, compute_classification_metrics


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing for focusing on hard tampering boundaries."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def collate_batch(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    clips = torch.stack([item.clip for item in batch], dim=0)
    labels = torch.tensor([item.label for item in batch], dtype=torch.long)
    sample_ids = [item.sample_id for item in batch]
    return clips, labels, sample_ids


def build_dataloader(
    manifest_path: Path,
    split: str,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ForgeryClipDataset(
        manifest_path=manifest_path,
        split=split,
        class_names=class_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, ClassificationMetrics]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_items = 0
    predictions: list[int] = []
    targets: list[int] = []

    use_amp = scaler is not None and device.type == "cuda"

    for clips, labels, _sample_ids in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(targets=targets, predictions=predictions)
    average_loss = total_loss / max(total_items, 1)
    return average_loss, metrics


def collect_predictions(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, ClassificationMetrics, list[dict[str, object]]]:
    model.train(mode=False)

    total_loss = 0.0
    total_items = 0
    predictions: list[int] = []
    targets: list[int] = []
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for clips, labels, sample_ids in dataloader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = probabilities.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            batch_predictions = predicted_labels.cpu().tolist()
            batch_targets = labels.cpu().tolist()
            batch_confidences = probabilities.max(dim=1).values.cpu().tolist()

            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

            for sample_id, target, prediction, confidence in zip(
                sample_ids,
                batch_targets,
                batch_predictions,
                batch_confidences,
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target_index": target,
                        "prediction_index": prediction,
                        "confidence": float(confidence),
                    }
                )

    metrics = compute_classification_metrics(targets=targets, predictions=predictions)
    average_loss = total_loss / max(total_items, 1)
    return average_loss, metrics, rows
