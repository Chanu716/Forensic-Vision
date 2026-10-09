from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray


@dataclass
class StandardizedEvaluationReport:
    """Comprehensive evaluation metrics adhering to documented statistical standards."""
    num_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_roc_auc_ovr: float
    macro_roc_auc_interpolated: float
    macro_average_precision: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    class_names: list[str]
    bootstrap_ci_95: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def compute_standardized_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray,
    class_names: list[str],
    num_bootstrap_samples: int = 1000,
    seed: int = 42,
) -> StandardizedEvaluationReport:
    """Computes standardized, publication-ready metrics across classification models.

    Includes both standard scikit-learn One-vs-Rest macro ROC-AUC and interpolated ROC curve AUC.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)
    n_samples = len(y_t)
    y_onehot = np.eye(n_classes)[y_t]
    preds = np.argmax(y_p, axis=1)

    acc = float(accuracy_score(y_t, preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_t, preds).tolist()

    # Continuous OvR Macro ROC-AUC (scikit-learn standard)
    roc_auc_ovr = float(roc_auc_score(y_onehot, y_p, average="macro", multi_class="ovr"))
    ap_macro = float(average_precision_score(y_onehot, y_p, average="macro"))

    # Per-class metrics & Interpolated ROC curves
    per_class: dict[str, dict[str, float | int]] = {}
    fpr_dict: dict[int, np.ndarray] = {}
    tpr_dict: dict[int, np.ndarray] = {}

    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_t, preds, average=None, zero_division=0
    )

    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_p[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        c_auc = float(auc(fpr, tpr))
        c_ap = float(average_precision_score(y_onehot[:, i], y_p[:, i]))
        per_class[cname] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(s_per[i]),
            "roc_auc": c_auc,
            "average_precision": c_ap,
        }

    # Interpolated Macro ROC-AUC (trapezoidal integration over union of FPR points)
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    roc_auc_interp = float(auc(all_fpr, mean_tpr))

    # Bootstrap 95% Confidence Intervals
    rng = np.random.default_rng(seed)
    boot_accs: list[float] = []
    boot_f1s: list[float] = []
    boot_aucs_ovr: list[float] = []
    boot_aucs_interp: list[float] = []

    for _ in range(num_bootstrap_samples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        b_t = y_t[indices]
        if len(np.unique(b_t)) < n_classes:
            continue
        b_p = y_p[indices]
        b_preds = preds[indices]
        b_onehot = y_onehot[indices]

        boot_accs.append(float(accuracy_score(b_t, b_preds)))
        _, _, b_f1, _ = precision_recall_fscore_support(
            b_t, b_preds, average="macro", zero_division=0
        )
        boot_f1s.append(float(b_f1))

        try:
            b_ovr = float(roc_auc_score(b_onehot, b_p, average="macro", multi_class="ovr"))
            boot_aucs_ovr.append(b_ovr)
        except Exception:
            pass

        try:
            b_fpr_dict = {}
            b_tpr_dict = {}
            for i in range(n_classes):
                bfpr, btpr, _ = roc_curve(b_onehot[:, i], b_p[:, i])
                b_fpr_dict[i] = bfpr
                b_tpr_dict[i] = btpr
            b_all_fpr = np.unique(np.concatenate([b_fpr_dict[i] for i in range(n_classes)]))
            b_mean_tpr = np.zeros_like(b_all_fpr)
            for i in range(n_classes):
                b_mean_tpr += np.interp(b_all_fpr, b_fpr_dict[i], b_tpr_dict[i])
            b_mean_tpr /= n_classes
            boot_aucs_interp.append(float(auc(b_all_fpr, b_mean_tpr)))
        except Exception:
            pass

    ci_dict = {
        "accuracy": [
            float(np.percentile(boot_accs, 2.5)),
            float(np.percentile(boot_accs, 97.5)),
        ] if boot_accs else [acc, acc],
        "macro_f1": [
            float(np.percentile(boot_f1s, 2.5)),
            float(np.percentile(boot_f1s, 97.5)),
        ] if boot_f1s else [float(f1_macro), float(f1_macro)],
        "macro_roc_auc_ovr": [
            float(np.percentile(boot_aucs_ovr, 2.5)),
            float(np.percentile(boot_aucs_ovr, 97.5)),
        ] if boot_aucs_ovr else [roc_auc_ovr, roc_auc_ovr],
        "macro_roc_auc_interpolated": [
            float(np.percentile(boot_aucs_interp, 2.5)),
            float(np.percentile(boot_aucs_interp, 97.5)),
        ] if boot_aucs_interp else [roc_auc_interp, roc_auc_interp],
    }

    return StandardizedEvaluationReport(
        num_samples=n_samples,
        accuracy=acc,
        macro_precision=float(p_macro),
        macro_recall=float(r_macro),
        macro_f1=float(f1_macro),
        macro_roc_auc_ovr=roc_auc_ovr,
        macro_roc_auc_interpolated=roc_auc_interp,
        macro_average_precision=ap_macro,
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
        bootstrap_ci_95=ci_dict,
    )

