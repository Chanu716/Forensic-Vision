from __future__ import annotations

import torch
from torch import nn


class ChannelAttention3D(nn.Module):
    """Channel attention for 3D tensor inputs (B, C, T, H, W)."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention3D(nn.Module):
    """Spatial-Temporal attention for 3D tensor inputs (B, C, T, H, W)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            2, 1, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(combined))
        return x * scale


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module combining Channel and Spatial attention."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention3D(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention3D(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
