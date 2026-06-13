from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forensic_vision.config import load_config
from forensic_vision.models.three_d_cnn import Forgery3DCNN
from forensic_vision.utils.repro import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 1 3DCNN baseline.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.values["experiment"]["seed"]
    set_seed(seed)

    model_cfg = config.values["model"]
    model = Forgery3DCNN(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        dropout=model_cfg["dropout"],
    )

    print("Loaded configuration:", args.config)
    print("Initialized model:", model.__class__.__name__)
    print("Next step: wire datasets, dataloaders, optimizer, and training loop.")


if __name__ == "__main__":
    main()
