"""Video-level decisions and uncertainty estimates for full-video evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def decide_video_label(probabilities: np.ndarray, localization_type: str) -> str:
    """Apply the calibrated classifier-plus-SSIM video decision rule.

    ``probabilities`` has shape (clips, 3) in authentic, insertion, deletion order.
    The thresholds were set on UCF-101 and must not be tuned on external test data.
    """
    mean_authentic = float(probabilities[:, 0].mean())
    max_insertion = float(probabilities[:, 1].max())
    max_deletion = float(probabilities[:, 2].max())

    if max_insertion >= 0.65 and max_insertion > max_deletion:
        return "frame_insertion"
    if max_deletion >= 0.75 and max_deletion > max_insertion and localization_type != "authentic":
        return "frame_deletion"
    if localization_type == "frame_insertion" and max_insertion >= 0.20:
        return "frame_insertion"
    if localization_type == "frame_deletion" and max_deletion >= 0.10:
        return "frame_deletion"
    if max_deletion >= 0.50 and max_deletion > max_insertion and max_deletion > mean_authentic:
        return "frame_deletion"
    return "authentic"


def _accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    labels = list(range(n_classes))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(accuracy_score(y_true, y_pred)), float(macro_f1)


def _bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[np.ndarray],
    n_classes: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float] | int]:
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        if len(np.unique(y_true[indices])) < n_classes:
            continue
        accuracy, macro_f1 = _accuracy_and_f1(y_true[indices], y_pred[indices], n_classes)
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)

    if not accuracies:
        return {"accuracy": [], "macro_f1": [], "valid_resamples": 0}
    return {
        "accuracy": np.percentile(accuracies, [2.5, 97.5]).tolist(),
        "macro_f1": np.percentile(macro_f1s, [2.5, 97.5]).tolist(),
        "valid_resamples": len(accuracies),
    }


def summarize_video_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
    cluster_ids: Sequence[int],
    scores: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize predictions with per-video and source-clustered bootstrap intervals."""
    if not y_true:
        raise ValueError("No evaluated videos to summarize.")
    if not len(y_true) == len(y_pred) == len(cluster_ids):
        raise ValueError("Predictions, targets and cluster ids must have equal lengths.")

    names = list(class_names)
    index = {name: position for position, name in enumerate(names)}
    targets = np.array([index[label] for label in y_true], dtype=int)
    predictions = np.array([index[label] for label in y_pred], dtype=int)
    clusters = np.asarray(cluster_ids)
    labels = list(range(len(names)))

    accuracy, macro_f1 = _accuracy_and_f1(targets, predictions, len(names))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    per_class: dict[str, dict[str, float | int | None]] = {
        name: {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(support[position]),
            "roc_auc": None,
        }
        for position, name in enumerate(names)
    }

    macro_auc: float | None = None
    if scores is not None:
        class_aucs: list[float] = []
        for position, name in enumerate(names):
            binary_target = targets == position
            if binary_target.any() and not binary_target.all():
                value = float(roc_auc_score(binary_target, scores[:, position]))
                per_class[name]["roc_auc"] = value
                class_aucs.append(value)
        if len(class_aucs) == len(names):
            macro_auc = float(np.mean(class_aucs))

    video_groups = [np.array([position]) for position in range(len(targets))]
    cluster_groups = [np.flatnonzero(clusters == value) for value in np.unique(clusters)]

    return {
        "num_videos": len(targets),
        "num_source_clusters": len(cluster_groups),
        "correct": int((targets == predictions).sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_roc_auc_ovr": macro_auc,
        "per_class": per_class,
        "class_names": names,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "bootstrap_ci_95_by_video": _bootstrap_interval(
            targets, predictions, video_groups, len(names), n_bootstrap, seed
        ),
        "bootstrap_ci_95_by_source_cluster": _bootstrap_interval(
            targets, predictions, cluster_groups, len(names), n_bootstrap, seed
        ),
    }
