"""Classical Forensic Baselines Evaluation Script for Forensic-Vision.

Implements and evaluates:
1. Experiment A: Standalone SSIM-only Baseline (Video-level detection and localization on 43 test videos)
2. Experiment B: Classical Frame-Difference Energy Baseline (Clip-level detection and 3-class classification)

Strictly uses validation data for any threshold selection.
Outputs machine-readable JSON and CSV reports in outputs/reports_comparative/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.localization.msssim import compute_frame_msssim_scores
from forensic_vision.preprocessing.video_io import read_video_frames


def evaluate_standalone_ssim(
    manifest_path: Path,
    image_size: tuple[int, int] = (112, 112),
    threshold: float = 0.85,
    use_adaptive_threshold: bool = True,
    use_multiscale: bool = False,
) -> dict[str, Any]:
    """Evaluates raw-frame SSIM transition clustering independently of the neural network.

    Evaluated on the 43 full uncropped test videos from UCF-101.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: STANDALONE SSIM-ONLY VIDEO BASELINE")
    print("=" * 70)
    print(f"[*] Protocol: single-scale SSIM, threshold={threshold}, adaptive={use_adaptive_threshold}, multiscale={use_multiscale}")

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    test_rows = [r for r in reader if r["split"] == "test"]
    video_gt: dict[str, dict[str, Any]] = {}
    for r in test_rows:
        label = r["label"]
        if label == "authentic":
            vpath = Path(r["source_video"])
            key = vpath.stem
            if key not in video_gt:
                video_gt[key] = {"video_path": vpath, "label": "authentic", "gt_start": None, "gt_end": None}
        elif label == "frame_insertion":
            vpath = Path("data/interim/test/frame_insertion") / f"{Path(r['source_video']).stem}_insert.mp4"
            key = vpath.stem
            if key not in video_gt and vpath.exists():
                video_gt[key] = {
                    "video_path": vpath,
                    "label": "frame_insertion",
                    "gt_start": float(r["forgery_start"]) if r["forgery_start"] else None,
                    "gt_end": float(r["forgery_end"]) if r["forgery_end"] else None,
                }
        elif label == "frame_deletion":
            vpath = Path("data/interim/test/frame_deletion") / f"{Path(r['source_video']).stem}_delete.mp4"
            key = vpath.stem
            if key not in video_gt and vpath.exists():
                video_gt[key] = {
                    "video_path": vpath,
                    "label": "frame_deletion",
                    "gt_start": float(r["forgery_start"]) if r["forgery_start"] else None,
                    "gt_end": float(r["forgery_end"]) if r["forgery_end"] else None,
                }

    print(f"[*] Loaded {len(video_gt)} unique full test videos.")
    class_names = ["authentic", "frame_insertion", "frame_deletion"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    y_true: list[str] = []
    y_pred: list[str] = []
    video_records: list[dict[str, Any]] = []

    localization_errors: list[float] = []
    detected_tampered_count = 0
    total_tampered_count = 0
    missed_tampered_videos: list[str] = []

    for vid_key, info in video_gt.items():
        vpath = info["video_path"]
        gt_label = info["label"]
        gt_start = info["gt_start"]
        gt_end = info["gt_end"]
        is_tampered = gt_label != "authentic"

        if is_tampered:
            total_tampered_count += 1

        try:
            frames, fps = read_video_frames(vpath, image_size=image_size)
        except Exception as e:
            print(f"[!] Error loading {vpath}: {e}")
            continue

        loc_result = compute_frame_msssim_scores(
            frames=frames,
            threshold=threshold,
            use_adaptive_threshold=use_adaptive_threshold,
            use_multiscale=use_multiscale,
            win_size=7,
        )

        pred_label = loc_result.forgery_type
        y_true.append(gt_label)
        y_pred.append(pred_label)

        # Localization error calculation conditional on detected seams
        loc_err: float | None = None
        if is_tampered:
            if loc_result.start_frame is not None and gt_start is not None:
                detected_tampered_count += 1
                if gt_label == "frame_deletion":
                    loc_err = float(abs(loc_result.start_frame - gt_start))
                else:  # insertion
                    err_start = abs(loc_result.start_frame - gt_start)
                    err_end = abs(loc_result.end_frame - gt_end) if loc_result.end_frame is not None and gt_end is not None else err_start
                    loc_err = float((err_start + err_end) / 2.0)
                localization_errors.append(loc_err)
            else:
                missed_tampered_videos.append(vid_key)

        video_records.append({
            "video_key": vid_key,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "correct": gt_label == pred_label,
            "num_clusters": len(loc_result.suspicious_indices),
            "pred_start": loc_result.start_frame,
            "pred_end": loc_result.end_frame,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "localization_error": loc_err,
            "effective_threshold": loc_result.dynamic_threshold,
        })

    # Metrics
    y_true_idx = [label_to_idx[l] for l in y_true]
    y_pred_idx = [label_to_idx[l] for l in y_pred]
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average=None, zero_division=0
    )

    auth_spec = float(r_per[0]) if len(r_per) > 0 else 0.0
    ins_rec = float(r_per[1]) if len(r_per) > 1 else 0.0
    del_rec = float(r_per[2]) if len(r_per) > 2 else 0.0

    # Binary detection (authentic vs tampered)
    bin_true = [0 if l == "authentic" else 1 for l in y_true]
    bin_pred = [0 if l == "authentic" else 1 for l in y_pred]
    bin_acc = float(accuracy_score(bin_true, bin_pred))
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        bin_true, bin_pred, average="binary", zero_division=0
    )

    seam_recall = float(detected_tampered_count / total_tampered_count) if total_tampered_count > 0 else 0.0
    med_loc_err = float(np.median(localization_errors)) if localization_errors else None
    mean_loc_err = float(np.mean(localization_errors)) if localization_errors else None

    print(f"\n[+] Standalone SSIM 3-Class Video Accuracy: {acc * 100:.2f}% ({sum(r['correct'] for r in video_records)}/{len(video_records)})")
    print(f"[+] Macro F1: {f1_macro:.4f} | Authentic Specificity: {auth_spec * 100:.2f}% | Deletion Recall: {del_rec * 100:.2f}% | Insertion Recall: {ins_rec * 100:.2f}%")
    print(f"[+] Binary Video Tampering Detection Acc: {bin_acc * 100:.2f}%, F1: {bin_f1:.4f}")
    print(f"[+] Seam Detection Recall: {detected_tampered_count}/{total_tampered_count} ({seam_recall * 100:.2f}%)")
    print(f"[+] Conditional Median Localization Error (N={len(localization_errors)}): {med_loc_err:.2f} frames")
    print(f"[+] Conditional Mean Localization Error (N={len(localization_errors)}): {mean_loc_err:.2f} frames")
    print(f"[+] Missed Tampered Videos ({len(missed_tampered_videos)}): {missed_tampered_videos}")

    report = {
        "method": "Standalone SSIM Baseline (Single-Scale, Unassisted)",
        "num_videos": len(video_records),
        "video_level_accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "authentic_specificity": auth_spec,
        "insertion_recall": ins_rec,
        "deletion_recall": del_rec,
        "binary_detection": {
            "accuracy": bin_acc,
            "precision": float(bin_p),
            "recall": float(bin_r),
            "f1": float(bin_f1),
        },
        "seam_detection": {
            "total_tampered_videos": total_tampered_count,
            "detected_tampered_videos": detected_tampered_count,
            "missed_tampered_videos": missed_tampered_videos,
            "seam_recall": seam_recall,
            "conditional_median_error": med_loc_err,
            "conditional_mean_error": mean_loc_err,
            "denominator_detected": len(localization_errors),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "video_details": video_records,
    }
    return report


def extract_clip_frame_diff_features(clip: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Computes consecutive frame difference energy trajectory and statistics for a clip.

    clip: np.ndarray of shape (T, H, W, C) in [0, 255] or [0, 1].
    Returns: (diff_trajectory, median_diff, mad_diff, peak_zscore)
    """
    if len(clip) < 2:
        return np.zeros(0, dtype=np.float32), 0.0, 0.0, 0.0

    if clip.dtype != np.float32 and clip.dtype != np.float64:
        clip = clip.astype(np.float32) / 255.0
    elif np.max(clip) > 1.0:
        clip = clip / 255.0

    # Frame differences across time (T-1, H, W, C)
    diffs = np.abs(clip[1:] - clip[:-1])
    # Scalar energy per transition step t
    energy_t = np.mean(diffs, axis=(1, 2, 3))  # shape (T-1,)

    med = float(np.median(energy_t))
    mad = float(np.median(np.abs(energy_t - med)))
    peak_val = float(np.max(energy_t))
    peak_z = float((peak_val - med) / (mad + 1e-5))

    return energy_t, med, mad, peak_z



def evaluate_classical_frame_diff(
    manifest_path: Path,
) -> dict[str, Any]:
    """Implements and evaluates the classical frame-difference energy baseline.

    Uses training/validation clips only to fit anomaly detection thresholds.
    Evaluates on held-out test clips (278 clips).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: CLASSICAL FRAME-DIFFERENCE ENERGY BASELINE")
    print("=" * 70)

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    class_names = ["authentic", "frame_insertion", "frame_deletion"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    val_records = [r for r in reader if r["split"] == "val"]
    test_records = [r for r in reader if r["split"] == "test"]

    print(f"[*] Extracting frame-diff energy for {len(val_records)} validation clips...")
    val_data = []
    for r in val_records:
        clip = np.load(r["clip_path"])
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        val_data.append({
            "sample_id": r["sample_id"],
            "label": r["label"],
            "energy_t": energy_t,
            "median": med,
            "mad": mad,
            "peak_z": peak_z,
        })

    print(f"[*] Extracting frame-diff energy for {len(test_records)} test clips...")
    test_data = []
    for r in test_records:
        clip = np.load(r["clip_path"])
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        test_data.append({
            "sample_id": r["sample_id"],
            "label": r["label"],
            "energy_t": energy_t,
            "median": med,
            "mad": mad,
            "peak_z": peak_z,
        })

    # Threshold calibration on Validation set only
    # Decision rule: count peaks where z_t > tau_z and energy_t > tau_abs
    # 0 peaks -> Authentic
    # 1 peak -> Deletion
    # >= 2 peaks -> Insertion
    print("[*] Calibrating thresholds on validation set (grid search over tau_z and tau_abs)...")
    best_val_f1 = -1.0
    best_tau_z = 3.0
    best_tau_abs = 0.02

    tau_z_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    tau_abs_candidates = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

    for tz in tau_z_candidates:
        for tabs in tau_abs_candidates:
            v_preds = []
            v_targets = [label_to_idx[item["label"]] for item in val_data]
            for item in val_data:
                e = item["energy_t"]
                med = item["median"]
                mad = item["mad"]
                z = (e - med) / (mad + 1e-5)
                # identify peaks above threshold
                spike_idx = np.where((z > tz) & (e > tabs))[0]
                # cluster spikes within distance <= 2
                clusters = 0
                if len(spike_idx) > 0:
                    clusters = 1
                    for idx_i in range(1, len(spike_idx)):
                        if spike_idx[idx_i] - spike_idx[idx_i - 1] > 2:
                            clusters += 1
                if clusters == 0:
                    v_preds.append(label_to_idx["authentic"])
                elif clusters == 1:
                    v_preds.append(label_to_idx["frame_deletion"])
                else:
                    v_preds.append(label_to_idx["frame_insertion"])

            _, _, vf1, _ = precision_recall_fscore_support(v_targets, v_preds, average="macro", zero_division=0)
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_tau_z = tz
                best_tau_abs = tabs

    print(f"[+] Optimal Validation Thresholds: tau_z = {best_tau_z:.2f}, tau_abs = {best_tau_abs:.3f} (Val Macro F1 = {best_val_f1:.4f})")

    # Evaluate on held-out test split using the frozen thresholds
    test_targets = [label_to_idx[item["label"]] for item in test_data]
    test_preds = []
    test_peak_z_scores = []
    test_cluster_counts = []

    for item in test_data:
        e = item["energy_t"]
        med = item["median"]
        mad = item["mad"]
        z = (e - med) / (mad + 1e-5)
        spike_idx = np.where((z > best_tau_z) & (e > best_tau_abs))[0]
        clusters = 0
        if len(spike_idx) > 0:
            clusters = 1
            for idx_i in range(1, len(spike_idx)):
                if spike_idx[idx_i] - spike_idx[idx_i - 1] > 2:
                    clusters += 1
        test_cluster_counts.append(clusters)
        test_peak_z_scores.append(item["peak_z"])

        if clusters == 0:
            test_preds.append(label_to_idx["authentic"])
        elif clusters == 1:
            test_preds.append(label_to_idx["frame_deletion"])
        else:
            test_preds.append(label_to_idx["frame_insertion"])

    # 3-Class Test Metrics
    cm3 = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])
    acc3 = float(accuracy_score(test_targets, test_preds))
    p3, r3, f1_3, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)

    # Binary Detection Task: Authentic (0) vs Tampered (1 or 2)
    bin_targets = [0 if t == 0 else 1 for t in test_targets]
    bin_preds = [0 if p == 0 else 1 for p in test_preds]
    cm_bin = confusion_matrix(bin_targets, bin_preds, labels=[0, 1])
    bin_acc = float(accuracy_score(bin_targets, bin_preds))
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(bin_targets, bin_preds, average="binary", zero_division=0)
    bin_auc = float(roc_auc_score(bin_targets, test_peak_z_scores))

    print(f"\n[+] 3-Class Clip Accuracy: {acc3 * 100:.2f}% ({sum(np.array(test_targets) == np.array(test_preds))}/{len(test_targets)})")
    print(f"[+] 3-Class Macro F1: {f1_3:.4f} | Precision: {p3:.4f} | Recall: {r3:.4f}")
    print(f"    - Authentic F1: {f1_per[0]:.4f} (Rec: {r_per[0]*100:.2f}%)")
    print(f"    - Frame Insertion F1: {f1_per[1]:.4f} (Rec: {r_per[1]*100:.2f}%)")
    print(f"    - Frame Deletion F1: {f1_per[2]:.4f} (Rec: {r_per[2]*100:.2f}%)")
    print(f"[+] Binary Tampering Detection (Authentic vs Tampered):")
    print(f"    - Binary Accuracy: {bin_acc * 100:.2f}% | F1: {bin_f1:.4f} | ROC-AUC: {bin_auc:.4f}")

    report = {
        "method": "Classical Frame-Difference Energy Baseline",
        "thresholds_selected": {
            "tau_z": best_tau_z,
            "tau_abs": best_tau_abs,
            "validation_macro_f1": best_val_f1,
        },
        "num_test_clips": len(test_targets),
        "three_class_evaluation": {
            "accuracy": acc3,
            "macro_precision": float(p3),
            "macro_recall": float(r3),
            "macro_f1": float(f1_3),
            "per_class": {
                class_names[i]: {
                    "precision": float(p_per[i]),
                    "recall": float(r_per[i]),
                    "f1": float(f1_per[i]),
                    "support": int(s_per[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": cm3.tolist(),
        },
        "binary_evaluation": {
            "task": "Authentic vs Manipulated (Binary)",
            "accuracy": bin_acc,
            "precision": float(bin_p),
            "recall": float(bin_r),
            "f1": float(bin_f1),
            "roc_auc": bin_auc,
            "confusion_matrix": cm_bin.tolist(),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Classical Forensic Baselines.")
    parser.add_argument("--manifest", default="data/manifests/clips_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports_comparative")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Standalone SSIM Baseline (Video-level)
    ssim_report = evaluate_standalone_ssim(manifest_path=manifest_path)
    ssim_json = output_dir / "standalone_ssim_results.json"
    ssim_json.write_text(json.dumps(ssim_report, indent=2), encoding="utf-8")
    print(f"\n[+] Saved Standalone SSIM results to: {ssim_json}")

    # Save detailed video CSV
    ssim_csv = output_dir / "standalone_ssim_video_predictions.csv"
    with ssim_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ssim_report["video_details"][0].keys()))
        writer.writeheader()
        writer.writerows(ssim_report["video_details"])
    print(f"[+] Saved Standalone SSIM video CSV to: {ssim_csv}")

    # 2. Evaluate Classical Frame-Difference Energy Baseline (Clip-level)
    fd_report = evaluate_classical_frame_diff(manifest_path=manifest_path)
    fd_json = output_dir / "classical_frame_diff_results.json"
    fd_json.write_text(json.dumps(fd_report, indent=2), encoding="utf-8")
    print(f"\n[+] Saved Classical Frame-Diff results to: {fd_json}")

    print("\n" + "=" * 70)
    print("CLASSICAL BASELINES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
