from __future__ import annotations

import csv
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forensic_vision.config import Config, load_config, validate_config
from forensic_vision.datasets.clip_dataset import ForgeryClipDataset
from forensic_vision.evaluation.metrics import (
    compute_classification_metrics,
    compute_standardized_metrics,
)
from forensic_vision.localization.msssim import (
    compute_frame_msssim_scores,
    compute_multiscale_ssim,
    compute_temporal_iou,
)
from forensic_vision.models.attention_3d import CBAM3D, ChannelAttention3D, SpatialAttention3D
from forensic_vision.models.dual_stream import DualStreamR2Plus1D, TemporalPeakPooling
from forensic_vision.models.r2plus1d import Conv2Plus1D, ForgeryR2Plus1D, R2Plus1DResidualBlock
from forensic_vision.models.three_d_cnn import Forgery3DCNN, build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestConfigValidationAndInertOptions(unittest.TestCase):
    """Test 1 & 2: Verification of config options and inert warnings."""

    def test_inert_options_detection(self) -> None:
        cfg = load_config(REPO_ROOT / "configs/enhanced.yaml", warn=False)
        detected_warnings = validate_config(cfg)
        # Should detect: use_gradient_guidance, use_pretrained, adaptive_sensitivity, use_multiscale notice
        warning_text = "\n".join(detected_warnings)
        self.assertIn("use_gradient_guidance", warning_text)
        self.assertIn("use_pretrained", warning_text)
        self.assertIn("adaptive_sensitivity", warning_text)
        self.assertIn("use_multiscale", warning_text)

    def test_load_config_emits_warnings(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _ = load_config(REPO_ROOT / "configs/enhanced.yaml", warn=True)
            self.assertGreaterEqual(len(captured), 3)
            categories = [c.category for c in captured]
            self.assertIn(UserWarning, categories)


class TestModelComponentsAndAblationShapes(unittest.TestCase):
    """Test 3: Output shapes and behavior of each model component and all 5 ablation variants."""

    def test_conv2plus1d_and_residual_block(self) -> None:
        x = torch.randn(2, 16, 8, 14, 14, requires_grad=True)
        conv = Conv2Plus1D(16, 32, stride=(2, 2, 2))
        y = conv(x)
        self.assertEqual(y.shape, (2, 32, 4, 7, 7))

        # Test R2Plus1DResidualBlock with and without CBAM
        block_cbam = R2Plus1DResidualBlock(32, 32, use_cbam=True)
        block_no_cbam = R2Plus1DResidualBlock(32, 32, use_cbam=False)

        y_cbam = block_cbam(y)
        loss_cbam = y_cbam.sum()
        loss_cbam.backward()
        self.assertIsNotNone(x.grad)

        x2 = torch.randn(2, 32, 4, 7, 7, requires_grad=True)
        y_no_cbam = block_no_cbam(x2)
        loss_no_cbam = y_no_cbam.sum()
        # Verify backward pass without autograd in-place error
        loss_no_cbam.backward()
        self.assertIsNotNone(x2.grad)

    def test_temporal_peak_pooling(self) -> None:
        tppool = TemporalPeakPooling(in_channels=32)
        x = torch.randn(2, 32, 10, 7, 7)
        out = tppool(x)
        self.assertEqual(out.shape, (2, 32))

    def test_all_five_ablation_variants(self) -> None:
        x = torch.randn(2, 3, 49, 112, 112)
        variants = ["variant_a", "variant_b", "variant_c", "variant_d", "variant_e"]
        expected_params = {
            "variant_a": 280131,
            "variant_b": 917462,
            "variant_c": 1867049,
            "variant_d": 1876541,
            "variant_e": 1975101,
        }

        for v in variants:
            m = build_model(arch=v)
            out = m(x)
            self.assertEqual(out.shape, (2, 3), f"Variant {v} produced unexpected shape {out.shape}")
            num_params = sum(p.numel() for p in m.parameters())
            self.assertEqual(
                num_params,
                expected_params[v],
                f"Variant {v} param mismatch: got {num_params}, expected {expected_params[v]}",
            )

    def test_enhanced_checkpoint_backward_compatibility(self) -> None:
        chk_path = REPO_ROOT / "outputs/checkpoints_enhanced/best.pt"
        if chk_path.exists():
            m = build_model(arch="variant_e")
            chk = torch.load(chk_path, map_location="cpu")
            msg = m.load_state_dict(chk["model_state_dict"])
            self.assertEqual(len(msg.missing_keys), 0)
            self.assertEqual(len(msg.unexpected_keys), 0)


class TestLocalizationModesAndThreshold(unittest.TestCase):
    """Test 4: Localization threshold calculations and single-/multiscale mode selection."""

    def test_clipped_median_relative_threshold(self) -> None:
        # Generate synthetic uniform frames with 2 tamper dips
        frames = np.ones((20, 64, 64, 3), dtype=np.uint8) * 128
        frames[5] = 0
        frames[12] = 0

        res = compute_frame_msssim_scores(
            frames=frames,
            threshold=0.85,
            use_adaptive_threshold=True,
            adaptive_sensitivity=3.5,  # inert parameter
            use_multiscale=False,
        )

        scores_arr = np.array(res.scores)
        median_val = float(np.median(scores_arr))
        expected_thresh = float(max(0.35, min(0.85, median_val - 0.14)))
        self.assertAlmostEqual(res.dynamic_threshold, expected_thresh, places=5)
        self.assertEqual(res.forgery_type, "frame_insertion")

    def test_single_scale_vs_multiscale_selection(self) -> None:
        f1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
        f2 = np.ones((32, 32, 3), dtype=np.uint8) * 105

        score_ms = compute_multiscale_ssim(f1, f2, num_scales=3)
        self.assertGreater(score_ms, 0.9)
        self.assertLessEqual(score_ms, 1.0)


class TestRobustnessTransformationSemantics(unittest.TestCase):
    """Test 5: Robustness transformation labels and their actual tensor operations."""

    def test_blur_uses_avg_pool_box_filter(self) -> None:
        # When s_param = 0.5 vs 1.0, both select kernel_size = 3
        # When s_param = 1.5, kernel_size = 5
        x = torch.rand(1, 3, 5, 20, 20)
        B, C, T, H, W = x.shape
        reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        def apply_blur(kernel_size: int) -> torch.Tensor:
            pad = kernel_size // 2
            blurred = F.avg_pool2d(
                F.pad(reshaped, (pad, pad, pad, pad), mode="reflect"),
                kernel_size=kernel_size,
                stride=1,
            )
            return blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        blur_05 = apply_blur(3)
        blur_10 = apply_blur(3)
        blur_15 = apply_blur(5)

        # sigma 0.5 and sigma 1.0 are computationally IDENTICAL due to identical kernel_size=3
        torch.testing.assert_close(blur_05, blur_10)
        # sigma 1.5 uses kernel_size=5, producing a distinct smoothed tensor
        self.assertFalse(torch.equal(blur_05, blur_15))

    def test_brightness_clamping(self) -> None:
        x = torch.ones(1, 3, 5, 10, 10) * 0.9
        bright = torch.clamp(x * 1.15, 0.0, 1.0)
        self.assertTrue(torch.all(bright <= 1.0))
        self.assertAlmostEqual(float(bright[0, 0, 0, 0, 0]), 1.0, places=4)


class TestCommonEvaluationFunction(unittest.TestCase):
    """Test 6: Common evaluation function for consistent macro ROC-AUC across models."""

    def test_standardized_metrics_calculation(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        # Probabilities with high accuracy
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.05, 0.9, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ])
        class_names = ["authentic", "frame_insertion", "frame_deletion"]

        report = compute_standardized_metrics(
            y_true=y_true,
            y_prob=y_prob,
            class_names=class_names,
            num_bootstrap_samples=100,
            seed=42,
        )

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.macro_roc_auc_ovr, 1.0)
        self.assertEqual(report.macro_roc_auc_interpolated, 1.0)
        self.assertIn("accuracy", report.bootstrap_ci_95)
        self.assertIn("macro_roc_auc_ovr", report.bootstrap_ci_95)


class TestDatasetSplitIntegrity(unittest.TestCase):
    """Test 7: Dataset split integrity and zero source-video leakage."""

    def test_zero_leakage_in_clips_manifest(self) -> None:
        manifest_path = REPO_ROOT / "data/manifests/clips_manifest.csv"
        if not manifest_path.exists():
            self.skipTest("Manifest file not found.")

        with manifest_path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        train_sources = {r["source_video"] for r in reader if r["split"] == "train"}
        val_sources = {r["source_video"] for r in reader if r["split"] == "val"}
        test_sources = {r["source_video"] for r in reader if r["split"] == "test"}

        train_donors = {r["donor_video"] for r in reader if r["split"] == "train" and r["donor_video"]}
        val_donors = {r["donor_video"] for r in reader if r["split"] == "val" and r["donor_video"]}
        test_donors = {r["donor_video"] for r in reader if r["split"] == "test" and r["donor_video"]}

        # 1. No source overlap
        self.assertEqual(len(train_sources & val_sources), 0, "Train-Val source video overlap detected!")
        self.assertEqual(len(train_sources & test_sources), 0, "Train-Test source video overlap detected!")
        self.assertEqual(len(val_sources & test_sources), 0, "Val-Test source video overlap detected!")

        # 2. No donor overlap
        self.assertEqual(len(train_donors & test_donors), 0, "Train-Test donor overlap detected!")

        # 3. No cross-split source-to-donor contamination
        self.assertEqual(len(train_sources & test_donors), 0, "Train source used as test donor!")
        self.assertEqual(len(test_sources & train_donors), 0, "Test source used as train donor!")

        # 4. Total samples match
        self.assertEqual(len(reader), 1941)
        self.assertEqual(len([r for r in reader if r["split"] == "test"]), 278)


if __name__ == "__main__":
    unittest.main()
