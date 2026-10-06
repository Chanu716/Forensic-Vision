"""Download a sample subset of UCF-101 videos for Forensic Vision dataset preparation."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download


def download_sample_dataset(
    output_dir: str | Path = "data/raw",
    target_count: int = 100,
    min_frames: int = 120,
    max_frames: int = 300,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Fetching UCF-101 subset archive from Hugging Face...")
    archive_path = hf_hub_download(
        repo_id="sayakpaul/ucf101-subset",
        filename="UCF101_subset.tar.gz",
        repo_type="dataset",
    )
    print(f"Archive available at: {archive_path}")

    saved_videos: list[Path] = []
    print(f"Extracting {target_count} valid sample videos to {output_path}...")

    with tarfile.open(archive_path, "r") as tar:
        avi_members = [
            m for m in tar.getmembers()
            if m.name.endswith(".avi") and not m.name.startswith(".")
        ]

        for member in avi_members:
            if len(saved_videos) >= target_count:
                break

            # Extract to a temp file to verify readability and frame count
            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue

            content = extracted_file.read()
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
                tmp.write(content)
                tmp_name = tmp.name

            try:
                cap = cv2.VideoCapture(tmp_name)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if min_frames <= frame_count <= max_frames:
                    # Keep category folder if present
                    rel_parts = Path(member.name).parts
                    # usually UCF101_subset / train / <Category> / <video>.avi
                    if len(rel_parts) >= 3:
                        category = rel_parts[-2]
                    else:
                        category = "general"

                    category_dir = output_path / category
                    category_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = category_dir / Path(member.name).name

                    dest_file.write_bytes(content)
                    saved_videos.append(dest_file)
                    print(f"Saved ({len(saved_videos)}/{target_count}): {dest_file} ({frame_count} frames)")
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)

    print(f"\nSuccessfully downloaded and saved {len(saved_videos)} sample videos to {output_path}.")
    return saved_videos


if __name__ == "__main__":
    download_sample_dataset()
