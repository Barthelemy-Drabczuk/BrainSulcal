"""Download Daly music and/or MUSIN-G datasets from OpenNeuro.

Usage:
    pixi run download-daly
    pixi run download-musing
    pixi run download-all

    # Or directly:
    python scripts/00_download_data.py --dataset daly
    python scripts/00_download_data.py --dataset musin_g
    python scripts/00_download_data.py --dataset all
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DATASETS = {
    "daly": {
        "openneuro_id": "ds002725",
        "target": "data/raw/daly_music",
        "description": "Daly et al. joint EEG+fMRI music dataset (21 subjects)",
    },
    "musin_g": {
        "openneuro_id": "ds003774",
        "target": "data/raw/musin_g",
        "description": "MUSIN-G EEG music genre dataset (20 subjects)",
    },
}


def download_dataset(name: str, root: Path) -> None:
    info = DATASETS[name]
    target = root / info["target"]
    target.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s → %s", info["description"], target)

    try:
        import openneuro  # openneuro-py
        openneuro.download(dataset=info["openneuro_id"], target_dir=str(target))
    except ImportError:
        logger.error(
            "openneuro-py not installed. Run `pixi install` first. "
            "Then: openneuro-py download --dataset %s --target %s",
            info["openneuro_id"], target,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BrainSulcal datasets.")
    parser.add_argument(
        "--dataset",
        choices=["daly", "musin_g", "all"],
        required=True,
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    datasets_to_download = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in datasets_to_download:
        download_dataset(name, root)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
