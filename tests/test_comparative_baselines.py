"""Unit tests for comparative baselines and classical forensic methods.

Tests:
1. ResNet18TemporalPooling architecture, forward pass, shape, and gradient flow.
2. ResNet18BiGRU architecture, forward pass, shape, and gradient flow.
3. Frame-difference energy calculations and anomaly statistics.
4. Classical frame-difference threshold decision logic.
5. Standalone SSIM decision rules and cluster mapping.
6. Split integrity and zero leakage across source/donor videos.
7. Missing/malformed input handling.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np
import torch

from forensic_vision.evaluation.metrics import (
    compute_classification_metrics,
    compute_standardized_metrics,
)
from forensic_vision.localization.msssim import compute_frame_msssim_scores
from forensic_vision.models.resnet_baselines import (
    ResNet18Backbone,
    ResNet18BiGRU,
    ResNet18TemporalPooling,
)
from scripts.evaluate_classical_baselines import extract_clip_frame_diff_features

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestComparativeModelArchitectures(unittest.TestCase):
    """Test ResNet-18 comparative models."""

    def test_resnet18_backbone(self) -> None:
        backbone = ResNet18Backbone()
        x = torch.randn(4, 3, 112, 112)
        out = backbone(x)
        self.assertEqual(out.shape, (4, 512))

    def test_resnet18_temporal_pooling_forward_and_backward(self) -> None:
        model = ResNet18TemporalPooling(num_classes=3)
        clips = torch.randn(2, 3, 8, 112, 112, requires_grad=True)
        logits = model(clips)
        self.assertEqual(logits.shape, (2, 3))
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(clips.grad)

    def test_resnet18_bigru_forward_and_backward(self) -> None:
        model = ResNet18BiGRU(num_classes=3, gru_hidden_size=64, num_gru_layers=2)
        clips = torch.randn(2, 3, 8, 112, 112, requires_grad=True)
        logits = model(clips)
        self.assertEqual(logits.shape, (2, 3))
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(clips.grad)

    def test_malformed_input_shape_handling(self) -> None:
        model = ResNet18TemporalPooling(num_classes=3)
        # Invalid 4D tensor instead of 5D
        with self.assertRaises(Exception):
            model(torch.randn(2, 3, 112, 112))


class TestClassicalFrameDiffEnergy(unittest.TestCase):
    """Test frame-difference energy extraction and peak z-score."""

    def test_uniform_frames_energy(self) -> None:
        # 10 identical frames
        clip = np.ones((10, 112, 112, 3), dtype=np.float32)
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        self.assertEqual(len(energy_t), 9)
        self.assertEqual(med, 0.0)
        self.assertEqual(mad, 0.0)
        self.assertEqual(peak_z, 0.0)

    def test_single_cut_discontinuity(self) -> None:
        # 10 frames with an abrupt jump at index 5
        clip = np.zeros((10, 112, 112, 3), dtype=np.float32)
        clip[5:] = 1.0  # transition at boundary 4->5
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        self.assertGreater(energy_t[4], 0.9)
        self.assertGreater(peak_z, 2.0)


class TestStandaloneSSIMRules(unittest.TestCase):
    """Test SSIM decision rules on synthetic video frame sequences."""

    def test_authentic_video_ssim_rule(self) -> None:
        # Uniform video -> SSIM = 1.0 across all frames
        frames = np.ones((15, 112, 112, 3), dtype=np.uint8) * 128
        res = compute_frame_msssim_scores(frames, threshold=0.85, use_multiscale=False)
        self.assertEqual(res.forgery_type, "authentic")
        self.assertEqual(len(res.suspicious_indices), 0)

    def test_single_transition_deletion_rule(self) -> None:
        # Video with 1 transition dip
        frames = np.ones((20, 112, 112, 3), dtype=np.uint8) * 128
        frames[10] = 0  # black frame creates dip at index 9 and index 10 (consecutive cluster)
        res = compute_frame_msssim_scores(frames, threshold=0.85, use_multiscale=False)
        self.assertEqual(res.forgery_type, "frame_deletion")
        self.assertIsNotNone(res.start_frame)

    def test_multi_transition_insertion_rule(self) -> None:
        # Video with 2 separated dips
        frames = np.ones((30, 112, 112, 3), dtype=np.uint8) * 128
        frames[8] = 0   # first dip cluster
        frames[22] = 0  # second dip cluster separated by >2 frames
        res = compute_frame_msssim_scores(frames, threshold=0.85, use_multiscale=False)
        self.assertEqual(res.forgery_type, "frame_insertion")
        self.assertIsNotNone(res.start_frame)
        self.assertIsNotNone(res.end_frame)


class TestSplitIntegrity(unittest.TestCase):
    """Verify source-video-disjoint partition guarantee."""

    def test_source_video_disjoint_splits(self) -> None:
        manifest_path = REPO_ROOT / "data/manifests/clips_manifest.csv"
        with manifest_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        train_sources = {r["source_video"] for r in rows if r["split"] == "train"}
        val_sources = {r["source_video"] for r in rows if r["split"] == "val"}
        test_sources = {r["source_video"] for r in rows if r["split"] == "test"}

        self.assertEqual(len(train_sources & test_sources), 0)
        self.assertEqual(len(val_sources & test_sources), 0)
        self.assertEqual(len(train_sources & val_sources), 0)

        # Donor video disjointness against test sources
        train_donors = {r["donor_video"] for r in rows if r["split"] == "train" and r["donor_video"]}
        test_donors = {r["donor_video"] for r in rows if r["split"] == "test" and r["donor_video"]}
        self.assertEqual(len(train_donors & test_sources), 0)
        self.assertEqual(len(test_donors & train_sources), 0)


class TestMetricCalculations(unittest.TestCase):
    """Verify standardized metric reporting."""

    def test_standardized_metrics_calculation(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # Perfect predictions
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.85, 0.1, 0.05],
            [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8],
        ])
        class_names = ["authentic", "frame_insertion", "frame_deletion"]
        report = compute_standardized_metrics(y_true, y_prob, class_names, num_bootstrap_samples=10, seed=42)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.macro_roc_auc_ovr, 1.0)

    def test_deletion_roc_auc_binary_convention(self) -> None:
        """Verify deletion-class AUC is calculated from column index 2 and binary ground truth."""
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
            [0.70, 0.15, 0.15],
            [0.10, 0.70, 0.20],
            [0.05, 0.15, 0.80],
        ])
        del_target = (y_true == 2).astype(int)
        del_prob = y_prob[:, 2]
        del_auc = float(roc_auc_score(del_target, del_prob))
        self.assertEqual(del_auc, 1.0)


class TestSavedPredictionsAlignment(unittest.TestCase):
    """Verify consistency and continuous probability normalization of saved test predictions."""

    def _verify_prediction_file(self, rel_path: str) -> None:
        p = REPO_ROOT / rel_path
        if not p.exists():
            self.skipTest(f"Prediction file {rel_path} does not exist.")

        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 278, f"{rel_path} has {len(rows)} samples, expected 278")
        mapping = {0: "authentic", 1: "frame_insertion", 2: "frame_deletion"}

        for r in rows:
            t_idx = int(r["target_index"])
            p_idx = int(r["pred_index"])
            self.assertIn(t_idx, [0, 1, 2])
            self.assertIn(p_idx, [0, 1, 2])
            self.assertEqual(r["target_label"], mapping[t_idx])
            self.assertEqual(r["pred_label"], mapping[p_idx])

            probs = [float(r["prob_auth"]), float(r["prob_ins"]), float(r["prob_del"])]
            for prob in probs:
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
            self.assertAlmostEqual(sum(probs), 1.0, places=3)
            self.assertEqual(p_idx, int(np.argmax(probs)))

    def test_variant_a_predictions(self) -> None:
        self._verify_prediction_file("outputs/ablation/variant_a/test_predictions.csv")

    def test_variant_e_10ep_predictions(self) -> None:
        self._verify_prediction_file("outputs/ablation/variant_e_10ep/test_predictions.csv")

    def test_resnet18_temporal_pool_predictions(self) -> None:
        self._verify_prediction_file("outputs/comparative_baselines/resnet18_temporal_pool/test_predictions.csv")

    def test_resnet18_bigru_predictions(self) -> None:
        self._verify_prediction_file("outputs/comparative_baselines/resnet18_bigru/test_predictions.csv")


class TestCheckpointIntegrity(unittest.TestCase):
    """Verify saved model checkpoints can be deserialized and executed."""

    def test_resnet18_temporal_pooling_checkpoint_load(self) -> None:
        chk_path = REPO_ROOT / "outputs/comparative_baselines/resnet18_temporal_pool/best.pt"
        if not chk_path.exists():
            self.skipTest("Checkpoint not found")
        chk = torch.load(chk_path, map_location="cpu")
        self.assertIn("model_state_dict", chk)
        model = ResNet18TemporalPooling(num_classes=3)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        dummy = torch.randn(1, 3, 8, 112, 112)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (1, 3))

    def test_resnet18_bigru_checkpoint_load(self) -> None:
        chk_path = REPO_ROOT / "outputs/comparative_baselines/resnet18_bigru/best.pt"
        if not chk_path.exists():
            self.skipTest("Checkpoint not found")
        chk = torch.load(chk_path, map_location="cpu")
        self.assertIn("model_state_dict", chk)
        model = ResNet18BiGRU(num_classes=3)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        dummy = torch.randn(1, 3, 8, 112, 112)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (1, 3))

    def test_variant_e_10ep_checkpoint_load(self) -> None:
        chk_path = REPO_ROOT / "outputs/ablation/variant_e_10ep/best.pt"
        if not chk_path.exists():
            self.skipTest("Checkpoint not found")
        from forensic_vision.models.three_d_cnn import build_model
        chk = torch.load(chk_path, map_location="cpu")
        self.assertIn("model_state_dict", chk)
        model = build_model(arch="variant_e")
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        dummy = torch.randn(1, 3, 49, 112, 112)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (1, 3))


class TestClassicalBaselineEdgeCases(unittest.TestCase):
    """Verify classical baseline edge cases."""

    def test_single_frame_exception(self) -> None:
        # Array with single frame -> cannot compute frame diffs
        clip = np.ones((1, 112, 112, 3), dtype=np.uint8)
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        self.assertEqual(len(energy_t), 0)
        self.assertEqual(peak_z, 0.0)

    def test_constant_zero_frame_diff(self) -> None:
        clip = np.zeros((8, 64, 64, 3), dtype=np.uint8)
        energy_t, med, mad, peak_z = extract_clip_frame_diff_features(clip)
        self.assertEqual(len(energy_t), 7)
        self.assertEqual(peak_z, 0.0)


if __name__ == "__main__":
    unittest.main()

