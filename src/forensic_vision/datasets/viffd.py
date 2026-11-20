"""Metadata parsing and test-set selection for the public VIFFD dataset.

VIFFD (Nguyen and Hu, Mendeley Data, doi:10.17632/r3ss3v53sj.6) ships no label file.
Labels are recovered from the file names in its "Testing Data" folder, so every rule
here is deliberately explicit and covered by tests. File names come from a remote
service and are never used as local paths.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any

VIFFD_DATASET_ID = "r3ss3v53sj"
VIFFD_VERSION = 6
VIFFD_METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{VIFFD_DATASET_ID}"
VIFFD_TEST_FOLDER_ID = "20f4a97f-4311-41cb-8dd8-83d415511f8b"
VIFFD_DOWNLOAD_PREFIX = "https://data.mendeley.com/public-files/datasets/"

ALLOWED_SUFFIXES = {".avi", ".mp4"}
EVALUATED_LABELS = ("authentic", "frame_insertion", "frame_deletion")

# Compact names of the recording sessions referenced by forged test videos.
SOURCE_TOKENS = (
    "anwarwalk2",
    "baby3",
    "finger1",
    "girlrun2",
    "haiyong2",
    "haiyongball2",
    "haiyongwalk2",
    "huabinwalk2",
    "nemoball4",
    "nemobicball1",
    "salmanexercise1",
    "salmanrun3",
    "salmanwalk7",
    "withoutlightputthecup1",
    "womenball7",
)
_SPELLING_FIXES = (
    ("wemen", "women"),
    ("dalman", "salman"),
    ("cannon", "canon"),
    ("salmanwalk8", "salmanwalk7"),
    ("womenball8", "womenball7"),
)
_DELETION_RANGE = re.compile(r"^delete\s+(\d+)\s*-\s*(\d+)")


@dataclass(slots=True)
class ViffdVideo:
    video_id: str
    filename: str
    label: str
    sha256: str
    size_bytes: int
    download_url: str
    local_path: str
    source_tokens: str
    cluster_id: int
    gt_start: int | None
    gt_end: int | None
    localization_eligible: bool
    flags: str


@dataclass(slots=True)
class ViffdExclusion:
    filename: str
    sha256: str
    label: str
    reason: str


def classify_filename(filename: str) -> str:
    """Map a VIFFD test file name to its forgery class."""
    name = filename.strip().lower()
    if name.startswith("insert"):
        return "frame_insertion"
    if name.startswith("delete"):
        return "frame_deletion"
    if name.startswith("copy"):
        return "frame_duplication"
    return "authentic"


def compact_name(filename: str) -> str:
    """Normalize spacing, re-encode suffixes and known spelling variants in a file name."""
    stem = PurePath(filename.strip()).stem.lower()
    stem = re.sub(r"_\d+", "", stem)
    compact = re.sub(r"[^a-z0-9]", "", stem)
    for wrong, right in _SPELLING_FIXES:
        compact = compact.replace(wrong, right)
    return compact


def find_source_tokens(filename: str) -> list[str]:
    """Return the recording sessions a file derives from.

    All Canon recordings share one token because their short names cannot be matched
    reliably. Merging them can only make clustered uncertainty estimates more cautious.
    """
    compact = compact_name(filename)
    tokens = [token for token in SOURCE_TOKENS if token in compact]
    if "canon" in compact:
        tokens.append("canon")
    return sorted(tokens) or [compact]


def parse_deletion_range(filename: str) -> tuple[int, int] | None:
    """Return the zero-based deleted range, assuming the name's numbers are one-based."""
    match = _DELETION_RANGE.match(filename.strip().lower())
    if match is None:
        return None
    first, last = int(match.group(1)), int(match.group(2))
    if first < 1 or last < first:
        return None
    return first - 1, last - 1


def _assign_clusters(token_sets: list[list[str]]) -> list[int]:
    """Group videos connected through any shared source recording."""
    parent: dict[str, str] = {}

    def find(token: str) -> str:
        parent.setdefault(token, token)
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    for tokens in token_sets:
        for token in tokens[1:]:
            parent[find(token)] = find(tokens[0])

    roots = sorted({find(tokens[0]) for tokens in token_sets})
    root_to_cluster = {root: index for index, root in enumerate(roots)}
    return [root_to_cluster[find(tokens[0])] for tokens in token_sets]


def select_test_videos(
    metadata: dict[str, Any],
) -> tuple[list[ViffdVideo], list[ViffdExclusion]]:
    """Select unique authentic, insertion and deletion videos from the test folder."""
    entries = [item for item in metadata["files"] if item.get("folder_id") == VIFFD_TEST_FOLDER_ID]
    if not entries:
        raise ValueError("VIFFD metadata contains no files in the test folder.")
    entries.sort(key=lambda item: (item["filename"].lower(), item["content_details"]["sha256_hash"]))

    kept: list[dict[str, Any]] = []
    excluded: list[ViffdExclusion] = []
    first_by_hash: dict[str, dict[str, Any]] = {}

    for item in entries:
        filename = item["filename"]
        details = item["content_details"]
        sha256 = str(details["sha256_hash"]).lower()
        label = classify_filename(filename)

        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Invalid SHA-256 for VIFFD file {filename!r}.")
        if label not in EVALUATED_LABELS:
            excluded.append(ViffdExclusion(filename, sha256, label, "class_out_of_scope"))
            continue

        original = first_by_hash.get(sha256)
        if original is not None:
            if classify_filename(original["filename"]) != label:
                raise ValueError(f"Identical VIFFD files have different labels: {filename!r}.")
            reason = f"exact_duplicate_of:{original['filename']}"
            excluded.append(ViffdExclusion(filename, sha256, label, reason))
            continue

        suffix = PurePath(filename.strip()).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported VIFFD file type: {filename!r}.")
        if not str(details["download_url"]).startswith(VIFFD_DOWNLOAD_PREFIX):
            raise ValueError(f"Unexpected download URL for VIFFD file {filename!r}.")

        first_by_hash[sha256] = item
        kept.append(item)

    token_sets = [find_source_tokens(item["filename"]) for item in kept]
    clusters = _assign_clusters(token_sets)

    videos: list[ViffdVideo] = []
    for item, tokens, cluster_id in zip(kept, token_sets, clusters):
        filename = item["filename"]
        details = item["content_details"]
        sha256 = str(details["sha256_hash"]).lower()
        label = classify_filename(filename)
        video_id = sha256[:16]
        suffix = PurePath(filename.strip()).suffix.lower()

        gt_start: int | None = None
        gt_end: int | None = None
        flags: list[str] = []
        if label == "frame_deletion":
            deleted_range = parse_deletion_range(filename)
            if deleted_range is None:
                flags.append("unparsed_deletion_range")
            else:
                gt_start, gt_end = deleted_range
            if " black" in filename.lower():
                flags.append("black_fill")
        elif label == "frame_insertion":
            flags.append("insertion_position_not_annotated")

        videos.append(
            ViffdVideo(
                video_id=video_id,
                filename=filename,
                label=label,
                sha256=sha256,
                size_bytes=int(item["size"]),
                download_url=str(details["download_url"]),
                local_path=f"videos/{video_id}{suffix}",
                source_tokens="|".join(tokens),
                cluster_id=cluster_id,
                gt_start=gt_start,
                gt_end=gt_end,
                localization_eligible=gt_start is not None and "black_fill" not in flags,
                flags="|".join(flags),
            )
        )
    return videos, excluded


def write_viffd_manifests(
    videos: list[ViffdVideo],
    excluded: list[ViffdExclusion],
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Write the evaluation manifest and the audit list of excluded files."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "viffd_test_manifest.csv"
    exclusions_path = root / "viffd_test_exclusions.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ViffdVideo.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(video) for video in videos)
    with exclusions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ViffdExclusion.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(item) for item in excluded)
    return manifest_path, exclusions_path
