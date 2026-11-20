"""Build the VIFFD test manifest and optionally download its selected videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.datasets.viffd import (
    VIFFD_METADATA_URL,
    VIFFD_VERSION,
    ViffdVideo,
    select_test_videos,
    write_viffd_manifests,
)

USER_AGENT = "forensic-vision-dataset-tool"
CHUNK_SIZE = 1024 * 1024


def fetch_metadata() -> dict:
    request = urllib.request.Request(
        VIFFD_METADATA_URL,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    if metadata.get("version") != VIFFD_VERSION:
        raise ValueError(f"Expected VIFFD version {VIFFD_VERSION}, got {metadata.get('version')}.")
    return metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_video(video: ViffdVideo, output_root: Path, retries: int = 3) -> str:
    target = output_root / video.local_path
    if target.exists() and file_sha256(target) == video.sha256:
        return "already_present"

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(video.download_url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != video.sha256:
                raise ValueError("SHA-256 mismatch")
            partial.replace(target)
            return "downloaded"
        except Exception as exc:  # Network and integrity failures are retried identically.
            last_error = exc
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {video.filename!r}: {last_error}")


def import_downloads(videos: list[ViffdVideo], output_root: Path, source_dir: Path) -> None:
    """Move browser-downloaded files into place when their SHA-256 matches the manifest."""
    by_hash = {video.sha256: video for video in videos}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".avi", ".mp4"}:
            continue
        video = by_hash.get(file_sha256(path))
        if video is None:
            continue
        target = output_root / video.local_path
        if target.exists() and file_sha256(target) == video.sha256:
            print(f"already present: {video.filename}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        print(f"imported: {video.filename}")

    missing = [video for video in videos if not (output_root / video.local_path).exists()]
    print(f"Videos still missing: {len(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/viffd", help="Directory for manifests and videos.")
    parser.add_argument("--metadata-file", default=None, help="Use a saved Mendeley metadata JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without writing files.")
    parser.add_argument("--manifest-only", action="store_true", help="Write manifests but skip videos.")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel downloads.")
    parser.add_argument(
        "--import-dir",
        default=None,
        help="Import matching videos downloaded elsewhere (for example a browser's Downloads folder).",
    )
    args = parser.parse_args()

    if args.metadata_file:
        metadata = json.loads(Path(args.metadata_file).read_text(encoding="utf-8"))
    else:
        metadata = fetch_metadata()

    videos, excluded = select_test_videos(metadata)
    total_bytes = sum(video.size_bytes for video in videos)
    print(f"Selected videos: {len(videos)} ({total_bytes / 1e9:.2f} GB)")
    print(f"Selected by class: {dict(sorted(Counter(video.label for video in videos).items()))}")
    print(f"Source clusters: {len({video.cluster_id for video in videos})}")
    print(f"Localization-eligible deletions: {sum(video.localization_eligible for video in videos)}")
    print(f"Excluded files: {len(excluded)}")
    for item in excluded:
        if item.reason.startswith("exact_duplicate_of:"):
            print(f"  duplicate: {item.filename!r} ({item.reason.split(':', 1)[1]!r})")
    if args.dry_run:
        return

    output_root = Path(args.output_root)
    manifest_path, exclusions_path = write_viffd_manifests(videos, excluded, output_root)
    (output_root / "mendeley_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Exclusions: {exclusions_path}")
    if args.manifest_only:
        return
    if args.import_dir:
        import_downloads(videos, output_root, Path(args.import_dir))
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(download_video, video, output_root): video for video in videos}
        for position, future in enumerate(as_completed(futures), start=1):
            video = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"failed ({exc})"
                failures.append(video.filename)
            print(f"[{position}/{len(videos)}] {status}: {video.filename}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} videos failed; rerun the command to retry them.")


if __name__ == "__main__":
    main()
