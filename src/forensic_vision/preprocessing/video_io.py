from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VIDEO_CODEC_FALLBACKS = ("mp4v", "XVID", "MJPG")


def read_video_frames(
    video_path: str | Path,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[np.ndarray] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image_size is not None:
            width, height = image_size
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    capture.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    return np.asarray(frames), float(fps)


def write_video_frames(
    frames: np.ndarray,
    output_path: str | Path,
    fps: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, H, W, C), got {frames.shape}")

    height, width = frames.shape[1], frames.shape[2]
    rgb_frames = frames.astype(np.uint8)

    last_error: Exception | None = None
    for codec in VIDEO_CODEC_FALLBACKS:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps if fps > 0 else 25.0,
            (width, height),
        )
        if not writer.isOpened():
            continue
        try:
            for frame in rgb_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # pragma: no cover - defensive cleanup
            last_error = exc
        finally:
            writer.release()
        if last_error is None:
            return

    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to open a video writer for {output_path}")
