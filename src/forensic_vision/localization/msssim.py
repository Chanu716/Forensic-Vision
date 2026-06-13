from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity


@dataclass(slots=True)
class LocalizationResult:
    scores: list[float]
    suspicious_indices: list[int]


def compute_frame_msssim_scores(
    frames: np.ndarray,
    threshold: float = 0.8,
    win_size: int = 7,
    multichannel: bool = True,
) -> LocalizationResult:
    """Approximate paper localization using consecutive-frame SSIM scores.

    The paper specifies MS-SSIM, but practical reproduction starts with
    consecutive-frame similarity scoring and a paper-aligned default threshold.
    This function expects frames as (time, height, width, channels).
    """

    if frames.ndim != 4:
        raise ValueError(
            f"Expected frame tensor with shape (T, H, W, C), got {tuple(frames.shape)}"
        )

    scores: list[float] = []
    suspicious_indices: list[int] = []
    channel_axis = -1 if multichannel else None

    for index in range(len(frames) - 1):
        score = structural_similarity(
            frames[index],
            frames[index + 1],
            win_size=win_size,
            channel_axis=channel_axis,
            data_range=1.0 if np.issubdtype(frames.dtype, np.floating) else 255,
        )
        scores.append(float(score))
        if score < threshold:
            suspicious_indices.append(index)

    return LocalizationResult(scores=scores, suspicious_indices=suspicious_indices)
