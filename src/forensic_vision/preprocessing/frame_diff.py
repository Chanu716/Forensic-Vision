from __future__ import annotations

import torch
from torch import nn


class AbsoluteFrameDifference(nn.Module):
    """Compute absolute frame differences across adjacent frames.

    Expected input shape: (batch, channels, time, height, width).
    Output shape: (batch, channels, time - 1, height, width).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(
                f"Expected 5D clip tensor (B, C, T, H, W), got shape {tuple(clips.shape)}"
            )

        diffs = torch.abs(clips[:, :, 1:, :, :] - clips[:, :, :-1, :, :])
        if self.threshold is None:
            return diffs

        return (diffs > self.threshold).to(diffs.dtype)
