from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class ClipSample:
    clip: torch.Tensor
    label: int
    sample_id: str


class ForgeryClipDataset(Dataset[ClipSample]):
    """Load clip tensors described by the generated CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        class_names: list[str],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.class_names = class_names
        self.label_to_index = {label: index for index, label in enumerate(class_names)}

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. Run scripts/prepare_dataset.py first."
            )

        frame = pd.read_csv(self.manifest_path)
        split_frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if split_frame.empty:
            raise ValueError(f"No samples found for split '{split}' in {self.manifest_path}")

        unknown_labels = sorted(set(split_frame["label"]) - set(self.label_to_index))
        if unknown_labels:
            raise ValueError(
                f"Manifest contains labels not present in config data.class_names: {unknown_labels}"
            )

        self.records = split_frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ClipSample:
        record = self.records[index]
        clip_path = Path(record["clip_path"])
        clip = np.load(clip_path)
        if clip.ndim != 4:
            raise ValueError(f"Expected clip tensor with shape (T, H, W, C), got {clip.shape}")

        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        label = self.label_to_index[record["label"]]
        return ClipSample(
            clip=clip_tensor,
            label=label,
            sample_id=str(record["sample_id"]),
        )
