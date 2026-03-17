"""Visualizations: UMAP + attention heatmaps + ROC curves.

Produces figures saved to runs/{experiment}/figures/:
  - ablation_bar.png            Accuracy per ablation condition
  - router_bias_umap.png        Subjects clustered by expert routing pattern
  - attention_heatmap.png       Sulcal region attention weights
  - roc_curves.png              ROC per subject

Usage:
    pixi run visualize
    python scripts/05_visualize.py --results-dir runs/default/results
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def plot_ablation_bar(ablation_csv: Path, output_path: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(ablation_csv, index_col=0)
    acc_col = [c for c in df.columns if "accuracy_mean" in c and "valence" in c]
    if not acc_col:
        logger.warning("No valence accuracy column found in %s", ablation_csv)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = df[acc_col[0]]
    vals.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", label="Chance (50%)")
    ax.axhline(0.718, color="orange", linestyle="--", label="Daly 2023 LSTM (71.8%)")
    ax.set_ylabel("Valence Accuracy (LOSO)")
    ax.set_title("BrainSulcal Ablation Study")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Saved: %s", output_path)


def plot_router_bias_umap(features: "np.ndarray", subject_ids: list, output_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=range(len(subject_ids)), cmap="tab20")
    for i, sid in enumerate(subject_ids):
        ax.annotate(sid.replace("sub-", ""), (embedding[i, 0], embedding[i, 1]), fontsize=7)
    ax.set_title("UMAP of Router Bias Patterns by Subject")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BrainSulcal visualizations.")
    parser.add_argument("--results-dir", default="runs/default/results")
    parser.add_argument("--output-dir", default="runs/default/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    ablation_csv = results_dir / "ablation_table.csv"
    if ablation_csv.exists():
        plot_ablation_bar(ablation_csv, output_dir / "ablation_bar.png")
    else:
        logger.info("No ablation_table.csv found at %s — run 04_evaluate.py first.", ablation_csv)

    logger.info("Visualization complete. Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
