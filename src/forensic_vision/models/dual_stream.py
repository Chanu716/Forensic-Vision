from __future__ import annotations

import torch
from torch import nn
from forensic_vision.models.r2plus1d import Conv2Plus1D, R2Plus1DResidualBlock


class DualStreamR2Plus1D(nn.Module):
    """Dual-Stream R(2+1)D with CBAM Attention for High-Accuracy Video Forgery Detection.

    Stream 1 (RGB Appearance): Factorized (2+1)D residual blocks with CBAM attention on raw RGB frames.
    Stream 2 (Motion Discontinuity): Factorized (2+1)D residual blocks with CBAM attention on frame differences.
    Cross-Attention Fusion: Gated sigmoid fusion mechanism dynamically weighting appearance vs motion tampering cues.
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

        def build_stream() -> nn.Sequential:
            return nn.Sequential(
                Conv2Plus1D(3, c1, stride=(2, 2, 2)),
                R2Plus1DResidualBlock(c1, c1, stride=(1, 1, 1), use_cbam=use_cbam),
                R2Plus1DResidualBlock(c1, c2, stride=(2, 2, 2), use_cbam=use_cbam),
                R2Plus1DResidualBlock(c2, c3, stride=(2, 2, 2), use_cbam=use_cbam),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
            )

        self.rgb_stream = build_stream()
        self.diff_stream = build_stream()

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
        f_rgb = self.rgb_stream(clips)

        # Stream 2: Consecutive Frame Differences
        diffs = torch.abs(clips[:, :, 1:, :, :] - clips[:, :, :-1, :, :])
        f_diff = self.diff_stream(diffs)

        # Gated Attention Fusion
        concat = torch.cat([f_rgb, f_diff], dim=1)
        g = self.gate(concat)
        f_fused = g * f_rgb + (1.0 - g) * f_diff

        features = torch.cat([f_fused, f_diff], dim=1)
        return self.classifier(features)
