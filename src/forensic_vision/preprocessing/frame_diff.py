from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AbsoluteFrameDifference(nn.Module):
    """Compute absolute frame differences across adjacent frames with optional multi-step & gradient guidance.

    Expected input shape: (batch, channels, time, height, width).
    Output shape: (batch, channels, time - 1, height, width).
    """

    def __init__(
        self,
        threshold: float | None = None,
        multi_step: bool = False,
        use_gradient_guidance: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.multi_step = multi_step
        self.use_gradient_guidance = use_gradient_guidance

        # Sobel filters for 2D spatial gradients
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _spatial_gradients(self, tensor_4d: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude for a 4D tensor (B*T, C, H, W)."""
        gray = tensor_4d.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(gray.device, gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(gray.device, gray.dtype), padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"Expected 5D clip tensor (B, C, T, H, W), got shape {tuple(clips.shape)}")

        diffs = torch.abs(clips[:, :, 1:, :, :] - clips[:, :, :-1, :, :])

        if self.use_gradient_guidance:
            b, c, t_diff, h, w = diffs.shape
            diffs_flat = diffs.reshape(b * t_diff, c, h, w)
            grad_mag = self._spatial_gradients(diffs_flat).reshape(b, 1, t_diff, h, w)
            diffs = diffs * (1.0 + grad_mag)

        if self.threshold is not None:
            diffs = (diffs > self.threshold).to(diffs.dtype)

        return diffs
