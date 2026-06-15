from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.datasets.preparation import prepare_dataset
from forensic_vision.utils.repro import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 1 video forgery datasets.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the dataset preparation configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config).values
    seed = config["experiment"]["seed"]
    set_seed(seed)

    data_cfg = config["data"]
    max_videos_per_class = data_cfg["max_videos_per_class"]
    samples = prepare_dataset(
        raw_dir=data_cfg["raw_dir"],
        interim_dir=data_cfg["interim_dir"],
        processed_dir=data_cfg["processed_dir"],
        manifests_dir=data_cfg["manifests_dir"],
        image_size=tuple(data_cfg["image_size"]),
        clip_length=data_cfg["clip_length"],
        clip_stride=data_cfg["clip_stride"],
        video_extensions=tuple(data_cfg["video_extensions"]),
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        forgery_lengths=list(data_cfg["forgery_lengths"]),
        seed=seed,
        max_videos_per_class=max_videos_per_class,
        save_intermediate_videos=bool(data_cfg["save_intermediate_videos"]),
    )

    print(f"Prepared {len(samples)} clips.")
    print(f"Manifest: {Path(data_cfg['manifests_dir']) / 'clips_manifest.csv'}")


if __name__ == "__main__":
    main()
