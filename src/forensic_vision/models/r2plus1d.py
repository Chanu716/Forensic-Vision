from __future__ import annotations

import torch
from torch import nn

from forensic_vision.models.attention_3d import CBAM3D


class Conv2Plus1D(nn.Module):
    """Factorized (2+1)D convolution block: 2D Spatial Conv + 1D Temporal Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if mid_channels is None:
            # Formula from R(2+1)D paper: preserves parameter budget
            mid_channels = max(1, (in_channels * out_channels * 3 * 3 * 3) // (in_channels * 3 * 3 + out_channels * 3))

        t_stride, h_stride, w_stride = stride
        t_pad, h_pad, w_pad = padding

        # Spatial Conv: (1, 3, 3)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, h_stride, w_stride),
            padding=(0, h_pad, w_pad),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm3d(mid_channels)
        self.relu_spatial = nn.ReLU(inplace=True)

        # Temporal Conv: (3, 1, 1)
        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            stride=(t_stride, 1, 1),
            padding=(t_pad, 0, 0),
            bias=False,
        )
        self.bn_temporal = nn.BatchNorm3d(out_channels)
        self.relu_temporal = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = self.relu_spatial(x)

        x = self.temporal_conv(x)
        x = self.bn_temporal(x)
        x = self.relu_temporal(x)
        return x


class R2Plus1DResidualBlock(nn.Module):
    """Residual Block using Conv2Plus1D with optional CBAM3D attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        use_cbam: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2Plus1D(in_channels, out_channels, stride=stride)
        self.conv2 = Conv2Plus1D(out_channels, out_channels, stride=(1, 1, 1))

        self.cbam = CBAM3D(out_channels) if use_cbam else None

        if in_channels != out_channels or stride != (1, 1, 1):
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class ForgeryR2Plus1D(nn.Module):
    """High-accuracy R(2+1)D video forgery classifier with optional attention."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        stage_channels: tuple[int, int, int] = (32, 64, 128),
        use_cbam: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2, c3 = stage_channels

        # Stem block
        self.stem = Conv2Plus1D(in_channels, c1, stride=(1, 2, 2))

        # Stages
        self.stage1 = R2Plus1DResidualBlock(c1, c1, stride=(1, 1, 1), use_cbam=use_cbam)
        self.stage2 = R2Plus1DResidualBlock(c1, c2, stride=(2, 2, 2), use_cbam=use_cbam)
        self.stage3 = R2Plus1DResidualBlock(c2, c3, stride=(2, 2, 2), use_cbam=use_cbam)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.fc(x)
