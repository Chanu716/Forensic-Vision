from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class VideoSample:
    sample_id: str
    source_video: str
    split: str
    label: str
    clip_path: str
    donor_video: str | None = None
    forgery_start: int | None = None
    forgery_end: int | None = None
    codec: str | None = None
    fps: float | None = None


def save_manifest(samples: list[VideoSample], output_path: str | Path) -> None:
    frame = pd.DataFrame(asdict(sample) for sample in samples)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
