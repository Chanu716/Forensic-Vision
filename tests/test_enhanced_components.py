from __future__ import annotations

import sys
import unittest
from pathlib import Path
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.models.attention_3d import CBAM3D, ChannelAttention3D, SpatialAttention3D
from forensic_vision.models.r2plus1d import ForgeryR2Plus1D
from forensic_vision.models.three_d_cnn import build_model
from forensic_vision.preprocessing.frame_diff import AbsoluteFrameDifference
from forensic_vision.localization.msssim import (
    compute_frame_msssim_scores,
    compute_multiscale_ssim,
    compute_temporal_iou,
)


def test_attention_3d_forward() -> None:
    x = torch.randn(2, 32, 16, 28, 28)
    cbam = CBAM3D(in_channels=32)
    out = cbam(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


def test_r2plus1d_forward() -> None:
    x = torch.randn(2, 3, 16, 56, 56)
    model = ForgeryR2Plus1D(in_channels=3, num_classes=3, stage_channels=(16, 32, 64), use_cbam=True)
    out = model(x)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"


def test_build_model_factory() -> None:
    x = torch.randn(2, 3, 16, 56, 56)
    model_baseline = build_model(arch="3dcnn", conv_channels=(16, 32, 64))
    model_enhanced = build_model(arch="r2plus1d_cbam", conv_channels=(16, 32, 64))

    out1 = model_baseline(x)
    out2 = model_enhanced(x)

    assert out1.shape == (2, 3)
    assert out2.shape == (2, 3)


def test_frame_diff_enhanced() -> None:
    clips = torch.randn(2, 3, 10, 32, 32)
    diff_module = AbsoluteFrameDifference(use_gradient_guidance=True)
    out = diff_module(clips)
    assert out.shape == (2, 3, 9, 32, 32), f"Expected shape (2, 3, 9, 32, 32), got {out.shape}"


def test_msssim_localization() -> None:
    frames = np.random.randint(0, 256, size=(20, 64, 64, 3), dtype=np.uint8)
    # Inject synthetic insertion dip at indices 5 and 12
    frames[5] = 0
    frames[12] = 0

    res = compute_frame_msssim_scores(
        frames,
        threshold=0.8,
        use_adaptive_threshold=True,
        use_multiscale=False,  # Fast test
    )
    assert isinstance(res.scores, list)
    assert len(res.scores) == 19
    assert res.forgery_type in ("authentic", "frame_insertion", "frame_deletion")


def test_temporal_iou() -> None:
    iou1 = compute_temporal_iou(10, 20, 10, 20)
    assert iou1 == 1.0, f"Expected 1.0, got {iou1}"

    iou2 = compute_temporal_iou(10, 20, 15, 25)
    assert 0.0 < iou2 < 1.0, f"Expected fractional IoU, got {iou2}"

    iou3 = compute_temporal_iou(10, 20, 30, 40)
    assert iou3 == 0.0, f"Expected 0.0, got {iou3}"


class TestEnhancedComponents(unittest.TestCase):
    def test_attention_3d(self) -> None:
        test_attention_3d_forward()

    def test_r2plus1d(self) -> None:
        test_r2plus1d_forward()

    def test_build_model(self) -> None:
        test_build_model_factory()

    def test_frame_diff(self) -> None:
        test_frame_diff_enhanced()

    def test_msssim(self) -> None:
        test_msssim_localization()

    def test_temporal_iou(self) -> None:
        test_temporal_iou()


if __name__ == "__main__":
    unittest.main()

