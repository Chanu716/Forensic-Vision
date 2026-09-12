from __future__ import annotations

import torch
from torch import nn

from forensic_vision.preprocessing.frame_diff import AbsoluteFrameDifference


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Forgery3DCNN(nn.Module):
    """Baseline 3DCNN aligned to the paper description."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.2,
        use_frame_difference: bool = True,
        diff_threshold: float | None = None,
    ) -> None:
        super().__init__()

        self.use_frame_difference = use_frame_difference
        self.frame_difference = AbsoluteFrameDifference(threshold=diff_threshold)

        c1, c2, c3 = conv_channels
        self.features = nn.Sequential(
            Conv3DBlock(in_channels, c1),
            Conv3DBlock(c1, c2),
            Conv3DBlock(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = self.frame_difference(clips) if self.use_frame_difference else clips
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    arch: str = "3dcnn",
    in_channels: int = 3,
    num_classes: int = 3,
    conv_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.2,
    use_frame_difference: bool = True,
    diff_threshold: float | None = None,
    use_cbam: bool = True,
) -> nn.Module:
    """Build a forgery classifier model by name."""
    if arch.lower() in ("3dcnn", "baseline"):
        return Forgery3DCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=tuple(conv_channels[:3]),  # type: ignore
            dropout=dropout,
            use_frame_difference=use_frame_difference,
            diff_threshold=diff_threshold,
        )
    elif arch.lower() in ("r2plus1d", "r2plus1d_cbam", "enhanced"):
        from forensic_vision.models.r2plus1d import ForgeryR2Plus1D

        class WrappedR2Plus1D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_frame_difference = use_frame_difference
                self.frame_difference = AbsoluteFrameDifference(threshold=diff_threshold)
                self.backbone = ForgeryR2Plus1D(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    stage_channels=tuple(conv_channels[:3]),  # type: ignore
                    use_cbam=use_cbam if "cbam" in arch.lower() or arch.lower() == "enhanced" else False,
                    dropout=dropout,
                )

            def forward(self, clips: torch.Tensor) -> torch.Tensor:
                x = self.frame_difference(clips) if self.use_frame_difference else clips
                return self.backbone(x)

        return WrappedR2Plus1D()
    else:
        raise ValueError(f"Unknown architecture type: {arch}")

