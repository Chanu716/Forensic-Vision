"""ResNet-18 Based Comparative Baselines for Forensic-Vision.

Implements:
1. ResNet18TemporalPooling (Experiment C):
   2D frame-level ResNet-18 backbone with temporal average and max pooling.
2. ResNet18BiGRU (Experiment D):
   2D frame-level ResNet-18 backbone with a 2-layer Bidirectional GRU temporal model.

Both models are trained from scratch without pretrained weights to match the repository protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor that outputs 512-dim spatial embeddings."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=None)
        # Retain all layers up to avgpool
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (N, 512)


class ResNet18TemporalPooling(nn.Module):
    """Experiment C: ResNet-18 2D spatial backbone with temporal pooling.

    Answers: Can a standard 2D CNN with temporal pooling match 3D spatio-temporal convolutions?
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet18Backbone()

        # Concatenation of temporal mean (512) and temporal max (512) -> 1024
        in_dim = 512 * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (B, C, T, H, W)
        b, c, t, h, w = clips.shape

        # Permute and fold time into batch dimension: (B * T, C, H, W)
        frames = clips.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)

        # Extract per-frame features
        feat = self.backbone(frames)  # (B * T, 512)

        # Unfold time dimension: (B, T, 512)
        feat_t = feat.view(b, t, -1)

        # Temporal pooling
        f_mean = torch.mean(feat_t, dim=1)  # (B, 512)
        f_max = torch.max(feat_t, dim=1).values  # (B, 512)

        pooled = torch.cat([f_mean, f_max], dim=1)  # (B, 1024)
        return self.classifier(pooled)


class ResNet18BiGRU(nn.Module):
    """Experiment D: ResNet-18 spatial feature extractor + 2-layer Bidirectional GRU.

    Answers: How does explicit recurrent temporal modeling compare against 3D convolutions?
    """

    def __init__(
        self,
        num_classes: int = 3,
        gru_hidden_size: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet18Backbone()

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # Bidirectional GRU output per frame is 2 * gru_hidden_size (256)
        gru_dim = gru_hidden_size * 2
        # Temporal mean (256) + temporal max (256) -> 512
        in_dim = gru_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (B, C, T, H, W)
        b, c, t, h, w = clips.shape

        frames = clips.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        feat = self.backbone(frames)  # (B * T, 512)
        feat_t = feat.view(b, t, -1)  # (B, T, 512)

        # Recurrent temporal sequence modeling
        gru_out, _ = self.gru(feat_t)  # (B, T, gru_dim)

        # Aggregate temporal representations
        f_mean = torch.mean(gru_out, dim=1)  # (B, gru_dim)
        f_max = torch.max(gru_out, dim=1).values  # (B, gru_dim)

        pooled = torch.cat([f_mean, f_max], dim=1)  # (B, 2 * gru_dim)
        return self.classifier(pooled)
