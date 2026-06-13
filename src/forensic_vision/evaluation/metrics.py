from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray


def compute_classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
) -> ClassificationMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(targets, predictions)
    confusion = confusion_matrix(targets, predictions)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        confusion=confusion,
    )
