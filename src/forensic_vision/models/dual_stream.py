from __future__ import annotations

import torch
from torch import nn
from forensic_vision.models.r2plus1d import Conv2Plus1D, R2Plus1DResidualBlock


class TemporalPeakPooling(nn.Module):
    """Temporal Peak-Preserving Pooling (TP-Pool).

    Pools spatially over (H, W) -> (1, 1), then computes temporal max, temporal mean,
    and temporal anomaly contrast to preserve single-frame tampering spikes.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        x_sp = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # (B, C, T)
        t_mean = torch.mean(x_sp, dim=-1)  # (B, C)
        t_max, _ = torch.max(x_sp, dim=-1)  # (B, C)
        t_contrast = t_max - t_mean  # (B, C) - isolated spike detector
        combined = torch.cat([t_mean, t_max, t_contrast], dim=-1)  # (B, 3*C)
        return self.proj(combined)  # (B, C)


class DualStreamR2Plus1D(nn.Module):
    """Dual-Stream R(2+1)D with CBAM Attention and TP-Pool for High-Accuracy Video Forgery Detection.

    Stream 1 (RGB Appearance): Factorized (2+1)D residual blocks with CBAM attention on raw RGB frames.
    Stream 2 (Motion Discontinuity): Factorized (2+1)D residual blocks with CBAM attention on frame differences.
    Temporal Peak-Preserving Pooling: Preserves isolated 1-frame boundary spikes from global averaging.
    Cross-Attention Fusion: Gated sigmoid fusion dynamically weighting appearance vs motion cues.
    """

    def __init__(
        self,
        num_classes: int = 3,
        stage_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.3,
        use_cbam: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3 = stage_channels

        def build_backbone() -> nn.Sequential:
            return nn.Sequential(
                Conv2Plus1D(3, c1, stride=(2, 2, 2)),
                R2Plus1DResidualBlock(c1, c1, stride=(1, 1, 1), use_cbam=use_cbam),
                R2Plus1DResidualBlock(c1, c2, stride=(2, 2, 2), use_cbam=use_cbam),
                R2Plus1DResidualBlock(c2, c3, stride=(2, 2, 2), use_cbam=use_cbam),
            )

        self.rgb_backbone = build_backbone()
        self.rgb_pool = TemporalPeakPooling(c3)

        self.diff_backbone = build_backbone()
        self.diff_pool = TemporalPeakPooling(c3)

        # Gated Cross-Stream Fusion
        self.gate = nn.Sequential(
            nn.Linear(c3 * 2, c3),
            nn.Sigmoid(),
        )

        # High-Capacity Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(c3 * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Forward pass for dual-stream video forgery detection.

        Args:
            clips: Tensor of shape (B, C, T, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        if clips.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {tuple(clips.shape)}")

        # Stream 1: RGB Appearance
        f_rgb = self.rgb_pool(self.rgb_backbone(clips))

        # Stream 2: Consecutive Frame Differences
        diffs = torch.abs(clips[:, :, 1:, :, :] - clips[:, :, :-1, :, :])
        f_diff = self.diff_pool(self.diff_backbone(diffs))

        # Gated Attention Fusion
        concat = torch.cat([f_rgb, f_diff], dim=1)
        g = self.gate(concat)
        f_fused = g * f_rgb + (1.0 - g) * f_diff

        features = torch.cat([f_fused, f_diff], dim=1)
        return self.classifier(features)
