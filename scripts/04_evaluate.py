"""Full LOSO evaluation suite.

Usage:
    pixi run evaluate
    python scripts/04_evaluate.py --checkpoint runs/default/checkpoints/best.pt
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO evaluation for BrainSulcal.")
    parser.add_argument("--checkpoint", default="runs/default/checkpoints/best.pt")
    parser.add_argument("--config", default="configs/daly_music.yaml")
    parser.add_argument("--output-dir", default="runs/default/results")
    parser.add_argument("--ablation", action="store_true", help="Also run ablation study.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from brainsulcal.evaluation.cross_subject import LOSOEvaluator
    from brainsulcal.evaluation.ablation import AblationStudy

    # TODO: wire up build_dataset_fn, build_model_fn, train_fn, eval_fn
    # from a loaded config + checkpoint. This requires implementing the full
    # training pipeline first.

    logger.info("Evaluation script ready. Implement LOSO wiring after training pipeline is complete.")
    logger.info("Output will be saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
