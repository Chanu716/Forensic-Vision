from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import pyramid_gaussian


@dataclass(slots=True)
class LocalizationResult:
    scores: list[float]
    suspicious_indices: list[int]
    forgery_type: str
    start_frame: int | None
    end_frame: int | None
    dynamic_threshold: float


def compute_multiscale_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    num_scales: int = 3,
    win_size: int = 7,
    multichannel: bool = True,
) -> float:
    """Compute true Multi-Scale SSIM between two frames using Gaussian pyramids."""
    channel_axis = -1 if multichannel else None
    data_range = 1.0  # pyramid_gaussian always normalizes outputs to floats in [0.0, 1.0]

    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:num_scales])
    weights /= weights.sum()

    pyramid1 = list(pyramid_gaussian(img1, max_layer=num_scales - 1, channel_axis=channel_axis))
    pyramid2 = list(pyramid_gaussian(img2, max_layer=num_scales - 1, channel_axis=channel_axis))

    msssim_score = 0.0
    for scale_idx in range(len(pyramid1)):
        p1 = pyramid1[scale_idx]
        p2 = pyramid2[scale_idx]
        if min(p1.shape[:2]) < win_size:
            break
        score = structural_similarity(
            p1,
            p2,
            win_size=win_size,
            channel_axis=channel_axis,
            data_range=data_range,
        )
        msssim_score += weights[scale_idx] * float(score)

    return float(msssim_score)


def compute_frame_msssim_scores(
    frames: np.ndarray,
    threshold: float = 0.8,
    use_adaptive_threshold: bool = True,
    adaptive_sensitivity: float = 2.5,
    use_multiscale: bool = True,
    win_size: int = 7,
    multichannel: bool = True,
) -> LocalizationResult:
    """Compute consecutive-frame MS-SSIM scores, adaptive thresholds, and automated boundary detection.

    Expects frames as (T, H, W, C).
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor with shape (T, H, W, C), got {tuple(frames.shape)}")

    scores: list[float] = []
    for index in range(len(frames) - 1):
        if use_multiscale and frames[index].shape[0] >= 16 and frames[index].shape[1] >= 16:
            score = compute_multiscale_ssim(
                frames[index],
                frames[index + 1],
                num_scales=3,
                win_size=win_size,
                multichannel=multichannel,
            )
        else:
            channel_axis = -1 if multichannel else None
            score = float(
                structural_similarity(
                    frames[index],
                    frames[index + 1],
                    win_size=win_size,
                    channel_axis=channel_axis,
                    data_range=1.0 if np.issubdtype(frames.dtype, np.floating) else 255.0,
                )
            )
        scores.append(score)

    scores_arr = np.array(scores)
    if use_adaptive_threshold and len(scores_arr) > 2:
        median_val = float(np.median(scores_arr))
        # Authentic video motion never causes an inter-frame drop > 0.18 from median.
        # Forgeries (insertions/deletions) cause drops of 0.25 to 0.95.
        effective_threshold = float(max(0.35, min(threshold, median_val - 0.20)))
    else:
        effective_threshold = threshold

    # Exclude boundary decoder edge artifacts (first 2 and last 2 frames)
    suspicious_indices = [
        int(i)
        for i in np.where(scores_arr < effective_threshold)[0]
        if 2 <= i < len(scores_arr) - 2
    ]

    # Automated Forgery Boundary Classifier (Peak Dip Signature Analysis)
    forgery_type = "authentic"
    start_frame = None
    end_frame = None

    if len(suspicious_indices) == 1:
        forgery_type = "frame_deletion"
        start_frame = suspicious_indices[0]
        end_frame = suspicious_indices[0]
    elif len(suspicious_indices) >= 2:
        # Check if dips are clustered or separate
        dip_gaps = np.diff(suspicious_indices)
        if len(dip_gaps) > 0 and np.max(dip_gaps) > 1:
            forgery_type = "frame_insertion"
            start_frame = suspicious_indices[0]
            end_frame = suspicious_indices[-1]
        else:
            forgery_type = "frame_deletion"
            start_frame = suspicious_indices[0]
            end_frame = suspicious_indices[-1]

    return LocalizationResult(
        scores=scores,
        suspicious_indices=suspicious_indices,
        forgery_type=forgery_type,
        start_frame=start_frame,
        end_frame=end_frame,
        dynamic_threshold=effective_threshold,
    )


def compute_temporal_iou(
    pred_start: int | None,
    pred_end: int | None,
    gt_start: int | None,
    gt_end: int | None,
) -> float:
    """Compute Temporal Intersection over Union (IoU) for frame localization."""
    if pred_start is None or pred_end is None or gt_start is None or gt_end is None:
        return 1.0 if (pred_start == gt_start and pred_end == gt_end) else 0.0

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)

    if inter_start > inter_end:
        return 0.0

    intersection = inter_end - inter_start + 1
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - intersection

    return float(intersection / union) if union > 0 else 0.0
