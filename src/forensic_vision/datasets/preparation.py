from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forensic_vision.datasets.manifest import VideoSample, save_manifest
from forensic_vision.preprocessing.video_io import read_video_frames, write_video_frames


@dataclass(slots=True)
class PreparedVideo:
    path: Path
    split: str
    sample_id: str
    label: str
    source_video: str
    donor_video: str | None = None
    forgery_start: int | None = None
    forgery_end: int | None = None
    fps: float | None = None


def discover_videos(
    raw_dir: str | Path,
    extensions: tuple[str, ...],
    max_videos_per_class: int | None = None,
    seed: int | None = None,
) -> list[Path]:
    root = Path(raw_dir)
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if max_videos_per_class is not None:
        by_class: dict[str, list[Path]] = {}
        for video in videos:
            try:
                class_name = video.relative_to(root).parts[0]
            except ValueError:
                class_name = video.parent.name
            by_class.setdefault(class_name, []).append(video)

        rng = random.Random(seed)
        selected: list[Path] = []
        for class_name in sorted(by_class):
            class_videos = list(by_class[class_name])
            rng.shuffle(class_videos)
            selected.extend(class_videos[:max_videos_per_class])

        return sorted(selected)
    return videos


def split_videos(
    videos: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled = list(videos)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def make_authentic_record(video_path: Path, split: str, fps: float) -> PreparedVideo:
    return PreparedVideo(
        path=video_path,
        split=split,
        sample_id=f"auth_{video_path.stem}",
        label="authentic",
        source_video=str(video_path),
        fps=fps,
    )


def choose_forgery_length(frame_count: int, lengths: list[int], rng: random.Random) -> int:
    valid = [length for length in lengths if 0 < length < frame_count]
    if not valid:
        raise ValueError(f"No valid forgery lengths for frame count {frame_count}")
    return rng.choice(valid)


def create_insertion_forgery(
    source_frames: np.ndarray,
    donor_frames: np.ndarray,
    lengths: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, int, int]:
    insert_len = choose_forgery_length(min(len(source_frames), len(donor_frames)), lengths, rng)
    source_start = rng.randint(1, max(1, len(source_frames) - insert_len - 1))
    donor_start = rng.randint(0, len(donor_frames) - insert_len)

    inserted = donor_frames[donor_start : donor_start + insert_len]
    forged = np.concatenate(
        [source_frames[:source_start], inserted, source_frames[source_start:]],
        axis=0,
    )
    return forged, source_start, source_start + insert_len - 1


def create_deletion_forgery(
    source_frames: np.ndarray,
    lengths: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, int, int]:
    delete_len = choose_forgery_length(len(source_frames), lengths, rng)
    delete_start = rng.randint(1, max(1, len(source_frames) - delete_len - 1))
    delete_end = delete_start + delete_len
    forged = np.concatenate(
        [source_frames[:delete_start], source_frames[delete_end:]],
        axis=0,
    )
    return forged, delete_start, delete_end - 1


def sliding_windows(
    frames: np.ndarray,
    clip_length: int,
    stride: int,
) -> list[np.ndarray]:
    if len(frames) < clip_length:
        return []
    windows: list[np.ndarray] = []
    for start in range(0, len(frames) - clip_length + 1, stride):
        windows.append(frames[start : start + clip_length])
    return windows


def _video_output_path(interim_dir: Path, split: str, label: str, stem: str) -> Path:
    return interim_dir / split / label / f"{stem}.mp4"


def _clip_output_path(processed_dir: Path, split: str, label: str, sample_id: str, index: int) -> Path:
    return processed_dir / split / label / f"{sample_id}_clip_{index:04d}.npy"


def prepare_dataset(
    *,
    raw_dir: str | Path,
    interim_dir: str | Path,
    processed_dir: str | Path,
    manifests_dir: str | Path,
    image_size: tuple[int, int],
    clip_length: int,
    clip_stride: int,
    video_extensions: tuple[str, ...],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    forgery_lengths: list[int],
    seed: int,
    max_videos_per_class: int | None = None,
    save_intermediate_videos: bool = True,
) -> list[VideoSample]:
    videos = discover_videos(
        raw_dir,
        video_extensions,
        max_videos_per_class=max_videos_per_class,
        seed=seed,
    )
    if len(videos) < 2:
        raise ValueError("At least two raw videos are required to generate insertion forgeries.")

    splits = split_videos(
        videos,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    rng = random.Random(seed)
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    manifests_dir = Path(manifests_dir)

    samples: list[VideoSample] = []

    for split_name, split_videos_list in splits.items():
        readable_videos: list[Path] = []
        for video_path in split_videos_list:
            try:
                # Probe readability without keeping every decoded video in memory.
                read_video_frames(video_path, image_size=image_size)
            except Exception as exc:
                warnings.warn(f"Skipping unreadable video {video_path}: {exc}", RuntimeWarning)
                continue
            readable_videos.append(video_path)

        if len(readable_videos) < 2:
            warnings.warn(
                f"Skipping split '{split_name}' because fewer than two readable videos were found.",
                RuntimeWarning,
            )
            continue

        for video_path in readable_videos:
            try:
                frames, fps = read_video_frames(video_path, image_size=image_size)
            except Exception as exc:
                warnings.warn(f"Skipping unreadable video {video_path}: {exc}", RuntimeWarning)
                continue

            authentic_record = make_authentic_record(video_path, split_name, fps)
            samples.extend(
                materialize_clips(
                    frames=frames,
                    record=authentic_record,
                    processed_dir=processed_dir,
                    clip_length=clip_length,
                    clip_stride=clip_stride,
                )
            )

            donor_paths = [path for path in readable_videos if path != video_path]
            rng.shuffle(donor_paths)

            donor_path: Path | None = None
            donor_frames: np.ndarray | None = None
            for candidate in donor_paths:
                try:
                    candidate_frames, _ = read_video_frames(candidate, image_size=image_size)
                except Exception as exc:
                    warnings.warn(f"Skipping unreadable donor video {candidate}: {exc}", RuntimeWarning)
                    continue
                donor_path = candidate
                donor_frames = candidate_frames
                break

            if donor_path is None or donor_frames is None:
                warnings.warn(
                    f"Skipping insertion forgery for {video_path} because no readable donor video was found.",
                    RuntimeWarning,
                )
                continue

            insertion_frames, insertion_start, insertion_end = create_insertion_forgery(
                frames,
                donor_frames,
                lengths=forgery_lengths,
                rng=rng,
            )
            insertion_record = PreparedVideo(
                path=_video_output_path(interim_dir, split_name, "frame_insertion", f"{video_path.stem}_insert"),
                split=split_name,
                sample_id=f"ins_{video_path.stem}",
                label="frame_insertion",
                source_video=str(video_path),
                donor_video=str(donor_path),
                forgery_start=insertion_start,
                forgery_end=insertion_end,
                fps=fps,
            )
            if save_intermediate_videos:
                write_video_frames(insertion_frames, insertion_record.path, fps)
            samples.extend(
                materialize_clips(
                    frames=insertion_frames,
                    record=insertion_record,
                    processed_dir=processed_dir,
                    clip_length=clip_length,
                    clip_stride=clip_stride,
                )
            )

            try:
                deletion_frames, deletion_start, deletion_end = create_deletion_forgery(
                    frames,
                    lengths=forgery_lengths,
                    rng=rng,
                )
                deletion_record = PreparedVideo(
                    path=_video_output_path(interim_dir, split_name, "frame_deletion", f"{video_path.stem}_delete"),
                    split=split_name,
                    sample_id=f"del_{video_path.stem}",
                    label="frame_deletion",
                    source_video=str(video_path),
                    forgery_start=deletion_start,
                    forgery_end=deletion_end,
                    fps=fps,
                )
                if save_intermediate_videos:
                    write_video_frames(deletion_frames, deletion_record.path, fps)
                samples.extend(
                    materialize_clips(
                        frames=deletion_frames,
                        record=deletion_record,
                        processed_dir=processed_dir,
                        clip_length=clip_length,
                        clip_stride=clip_stride,
                    )
                )
            except Exception as exc:
                warnings.warn(f"Skipping deletion forgery for {video_path}: {exc}", RuntimeWarning)

            del frames
            del donor_frames

    save_manifest(samples, manifests_dir / "clips_manifest.csv")
    return samples


def materialize_clips(
    *,
    frames: np.ndarray,
    record: PreparedVideo,
    processed_dir: Path,
    clip_length: int,
    clip_stride: int,
) -> list[VideoSample]:
    clips = sliding_windows(frames, clip_length=clip_length, stride=clip_stride)
    samples: list[VideoSample] = []
    for index, clip in enumerate(clips):
        output_path = _clip_output_path(
            processed_dir,
            record.split,
            record.label,
            record.sample_id,
            index,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, clip)
        samples.append(
            VideoSample(
                sample_id=f"{record.sample_id}_clip_{index:04d}",
                source_video=record.source_video,
                split=record.split,
                label=record.label,
                clip_path=str(output_path),
                donor_video=record.donor_video,
                forgery_start=record.forgery_start,
                forgery_end=record.forgery_end,
                fps=record.fps,
            )
        )
    return samples
