"""
Script to compare Baseline 3D-CNN (Gowda & Pawar, 2023) against Proposed Enhanced Dual-Stream.
Computes head-to-head ROC/AUC curves, McNemar's statistical significance test,
and generates comparative publication-quality visualizations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.models.three_d_cnn import build_model
from forensic_vision.training import build_dataloader


def get_model_probabilities(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads a model and extracts test set probabilities."""
    config = load_config(config_path)
    data_cfg = config.values["data"]
    model_cfg = config.values["model"]
    class_names = list(data_cfg["class_names"])
    manifest_path = Path(data_cfg["manifests_dir"]) / "clips_manifest.csv"

    arch = model_cfg.get("arch", "3dcnn")
    model = build_model(
        arch=arch,
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        dropout=model_cfg.get("dropout", 0.2),
        use_frame_difference=model_cfg.get("use_frame_difference", True),
        diff_threshold=model_cfg.get("diff_threshold", None),
        use_cbam=model_cfg.get("use_cbam", True),
    ).to(device)

    chk = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    dataloader = build_dataloader(
        manifest_path=manifest_path,
        split="test",
        class_names=class_names,
        batch_size=16,
        num_workers=0,
        shuffle=False,
    )

    all_targets = []
    all_probs = []
    all_sample_ids = []

    with torch.no_grad():
        for clips, labels, sample_ids in dataloader:
            clips = clips.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(clips)
                probs = F.softmax(logits, dim=1)
            all_targets.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
            all_sample_ids.extend(sample_ids)

    return np.array(all_targets), np.concatenate(all_probs, axis=0), all_sample_ids


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Dict[str, Any]:
    """Computes McNemar's test for paired classification accuracy."""
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    # Contingency table
    # a_correct & b_correct, a_correct & b_wrong
    # a_wrong & b_correct,   a_wrong & b_wrong
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B correct
    n10 = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    n11 = int(np.sum(correct_a & correct_b))

    b = n10
    c = n01
    # Continuity corrected McNemar statistic
    stat = (abs(b - c) - 1.0) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_value = 1.0 - stats.chi2.cdf(stat, df=1) if stat > 0 else 1.0

    return {
        "contingency_table": [[n11, n10], [n01, n00]],
        "statistic": float(stat),
        "p_value": float(p_value),
        "statistically_significant_005": bool(p_value < 0.05),
        "statistically_significant_001": bool(p_value < 0.01),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running comparison on device: {device}")

    base_config = "configs/base.yaml"
    base_checkpoint = "outputs/checkpoints/best.pt"
    enhanced_config = "configs/enhanced.yaml"
    enhanced_checkpoint = "outputs/checkpoints_enhanced/best.pt"

    print("[*] Evaluating Baseline 3D-CNN (Gowda & Pawar, 2023)...")
    y_true_base, base_probs, _ = get_model_probabilities(base_config, base_checkpoint, device)

    print("[*] Evaluating Proposed Enhanced Dual-Stream R(2+1)D...")
    y_true_enh, enh_probs, _ = get_model_probabilities(enhanced_config, enhanced_checkpoint, device)

    assert np.array_equal(y_true_base, y_true_enh), "Target labels must match across evaluations!"
    y_true = y_true_base
    n_classes = 3
    y_onehot = np.eye(n_classes)[y_true]
    class_names = ["authentic", "frame_insertion", "frame_deletion"]

    base_preds = np.argmax(base_probs, axis=1)
    enh_preds = np.argmax(enh_probs, axis=1)

    # Compute high-level metrics
    base_acc = float(np.mean(base_preds == y_true))
    enh_acc = float(np.mean(enh_preds == y_true))

    base_f1 = float(f1_score(y_true, base_preds, average="macro"))
    enh_f1 = float(f1_score(y_true, enh_preds, average="macro"))

    # ROC AUC scores
    base_macro_auc = float(roc_auc_score(y_onehot, base_probs, average="macro", multi_class="ovr"))
    enh_macro_auc = float(roc_auc_score(y_onehot, enh_probs, average="macro", multi_class="ovr"))

    # McNemar's test
    mcnemar_res = mcnemar_test(y_true, base_preds, enh_preds)

    print(f"  [+] Baseline 3D-CNN Accuracy: {base_acc*100:.2f}%, F1: {base_f1:.4f}, Macro AUC: {base_macro_auc:.4f}")
    print(f"  [+] Proposed Model Accuracy: {enh_acc*100:.2f}%, F1: {enh_f1:.4f}, Macro AUC: {enh_macro_auc:.4f}")
    print(f"  [+] McNemar Stat: {mcnemar_res['statistic']:.4f}, p-value: {mcnemar_res['p_value']:.4e} (Significant: {mcnemar_res['statistically_significant_001']})")

    # Generate Comparative ROC Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Plot 1: Macro-average ROC Comparison
    ax1 = axes[0]
    # Baseline macro ROC
    all_fpr_base = np.linspace(0, 1, 200)
    mean_tpr_base = np.zeros_like(all_fpr_base)
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_onehot[:, c], base_probs[:, c])
        mean_tpr_base += np.interp(all_fpr_base, fpr_c, tpr_c)
    mean_tpr_base /= n_classes

    # Enhanced macro ROC
    all_fpr_enh = np.linspace(0, 1, 200)
    mean_tpr_enh = np.zeros_like(all_fpr_enh)
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_onehot[:, c], enh_probs[:, c])
        mean_tpr_enh += np.interp(all_fpr_enh, fpr_c, tpr_c)
    mean_tpr_enh /= n_classes

    ax1.plot(
        all_fpr_enh,
        mean_tpr_enh,
        color="#1f77b4",
        linewidth=2.8,
        label=f"Proposed Enhanced Dual-Stream (AUC = {enh_macro_auc:.4f})",
    )
    ax1.plot(
        all_fpr_base,
        mean_tpr_base,
        color="#d62728",
        linestyle="--",
        linewidth=2.4,
        label=f"Baseline 3D-CNN Gowda & Pawar (AUC = {base_macro_auc:.4f})",
    )
    ax1.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title("Overall Macro-Average ROC Comparison", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=10.5, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Frame Deletion ROC Comparison (highlighting TP-Pool impact)
    ax2 = axes[1]
    fpr_del_base, tpr_del_base, _ = roc_curve(y_onehot[:, 2], base_probs[:, 2])
    auc_del_base = auc(fpr_del_base, tpr_del_base)

    fpr_del_enh, tpr_del_enh, _ = roc_curve(y_onehot[:, 2], enh_probs[:, 2])
    auc_del_enh = auc(fpr_del_enh, tpr_del_enh)

    ax2.plot(
        fpr_del_enh,
        tpr_del_enh,
        color="#2ca02c",
        linewidth=2.8,
        label=f"Proposed with TP-Pool (AUC = {auc_del_enh:.4f})",
    )
    ax2.plot(
        fpr_del_base,
        tpr_del_base,
        color="#e377c2",
        linestyle="--",
        linewidth=2.4,
        label=f"Baseline 3D-CNN AvgPool (AUC = {auc_del_base:.4f})",
    )
    ax2.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax2.set_title("Frame Deletion Detection ROC Comparison\n(Impact of Temporal Peak Pooling on 1-Frame Discontinuities)", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=10.5, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    rep_plot = Path("outputs/reports_enhanced/roc_comparison_baseline_vs_proposed.png")
    doc_plot = Path("docs/figures/roc_comparison_baseline_vs_proposed.png")
    plt.savefig(rep_plot, dpi=300)
    plt.savefig(doc_plot, dpi=300)
    plt.close()
    print(f"[+] Saved Comparison ROC plots to {rep_plot} and {doc_plot}")

    # Save comparison metrics JSON
    comp_metrics = {
        "baseline_3dcnn": {
            "accuracy": base_acc,
            "macro_f1": base_f1,
            "macro_auc": base_macro_auc,
            "deletion_auc": float(auc_del_base),
            "confusion_matrix": confusion_matrix(y_true, base_preds).tolist(),
        },
        "proposed_enhanced": {
            "accuracy": enh_acc,
            "macro_f1": enh_f1,
            "macro_auc": enh_macro_auc,
            "deletion_auc": float(auc_del_enh),
            "confusion_matrix": confusion_matrix(y_true, enh_preds).tolist(),
        },
        "mcnemar_statistical_test": mcnemar_res,
    }
    comp_json_path = Path("outputs/reports_enhanced/baseline_comparison_metrics.json")
    comp_json_path.write_text(json.dumps(comp_metrics, indent=2), encoding="utf-8")
    print(f"[+] Saved Comparison JSON to {comp_json_path}")


if __name__ == "__main__":
    main()
