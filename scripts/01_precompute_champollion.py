"""Pre-compute Champollion sulcal embeddings from T1 MRIs.

Runs champollion_pipeline (via its own Pixi environment) on Daly T1 MRIs,
then aggregates per-subject CSVs into numpy arrays cached to:
    data/processed/champollion/{subject_id}.npy        shape: (56, embedding_dim)
    data/processed/champollion/{subject_id}_mask.npy   shape: (56,) bool

Run ONCE before training. Results are deterministic and reused every epoch.

Usage:
    pixi run precompute-champollion

IMPORTANT: champollion_pipeline uses its own Pixi environment.
Do NOT run it from brainsulcal's environment directly.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

N_REGIONS = 56
CHAMPOLLION_PIPELINE_DIR = Path("external/champollion_pipeline")


def run_champollion_pipeline(subject_id: str, t1_path: Path, output_dir: Path) -> None:
    """Run champollion_pipeline via its own Pixi environment."""
    cmd = [
        "pixi", "run", "--manifest-path",
        str(CHAMPOLLION_PIPELINE_DIR / "pixi.toml"),
        "champollion",
        "--subject", subject_id,
        "--t1", str(t1_path),
        "--output", str(output_dir),
    ]
    logger.info("Running champollion_pipeline for %s", subject_id)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("champollion_pipeline failed for %s:\n%s", subject_id, result.stderr)
        raise RuntimeError(f"champollion_pipeline failed for {subject_id}")


def aggregate_champollion_csvs(
    subject_id: str,
    pipeline_output_dir: Path,
    region_file: Path,
    cache_dir: Path,
) -> None:
    """Read Champollion CSV outputs and save as (56, d) numpy array.

    Region matching is done by name using sulci_regions_champollion_V1.json,
    NOT by assuming a fixed ordering of CSV files.
    """
    import pandas as pd

    # Load region ordering from champollion_pipeline
    with open(region_file) as f:
        region_data = json.load(f)
    region_names = region_data.get("regions", [])

    if len(region_names) != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} regions in {region_file}, got {len(region_names)}"
        )

    embeddings = []
    mask = []

    subject_output = pipeline_output_dir / subject_id / "embeddings"

    for region_name in region_names:
        csv_path = subject_output / f"{region_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            emb = df.values.flatten().astype(np.float32)
            embeddings.append(emb)
            mask.append(True)
        else:
            logger.warning("Missing region %s for %s — using zeros.", region_name, subject_id)
            # Placeholder: filled with correct dim after first valid region found
            embeddings.append(None)
            mask.append(False)

    # Infer embedding_dim from first valid embedding
    embedding_dim = next((e.shape[0] for e in embeddings if e is not None), 1)
    embeddings_filled = [
        e if e is not None else np.zeros(embedding_dim, dtype=np.float32)
        for e in embeddings
    ]

    emb_array = np.stack(embeddings_filled)  # (56, embedding_dim)
    mask_array = np.array(mask, dtype=bool)  # (56,)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{subject_id}.npy", emb_array)
    np.save(cache_dir / f"{subject_id}_mask.npy", mask_array)
    logger.info(
        "Saved %s: shape=%s, valid_regions=%d/%d",
        subject_id, emb_array.shape, mask_array.sum(), N_REGIONS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute Champollion embeddings.")
    parser.add_argument("--raw-dir", default="data/raw/daly_music")
    parser.add_argument("--output-dir", default="data/processed/champollion")
    parser.add_argument("--pipeline-output", default="data/processed/champollion_pipeline")
    parser.add_argument("--region-file",
                        default="external/champollion_pipeline/share/sulci_regions_champollion_V1.json")
    parser.add_argument("--subjects", nargs="*", help="Subset of subjects (default: all)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip champollion_pipeline (assume CSVs already exist)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    cache_dir = Path(args.output_dir)
    pipeline_output = Path(args.pipeline_output)
    region_file = Path(args.region_file)

    # Find subjects
    if args.subjects:
        subject_ids = args.subjects
    else:
        subject_ids = sorted([d.name for d in raw_dir.iterdir() if d.name.startswith("sub-")])

    logger.info("Processing %d subjects: %s", len(subject_ids), subject_ids[:3])

    for subject_id in subject_ids:
        # Find T1 MRI
        anat_dir = raw_dir / subject_id / "anat"
        t1_files = list(anat_dir.glob("*T1w.nii*")) if anat_dir.exists() else []

        if not t1_files:
            logger.warning("No T1 MRI for %s — saving zero embeddings.", subject_id)
            emb = np.zeros((N_REGIONS, 1), dtype=np.float32)
            mask = np.zeros(N_REGIONS, dtype=bool)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / f"{subject_id}.npy", emb)
            np.save(cache_dir / f"{subject_id}_mask.npy", mask)
            continue

        if not args.skip_pipeline:
            run_champollion_pipeline(subject_id, t1_files[0], pipeline_output)

        aggregate_champollion_csvs(subject_id, pipeline_output, region_file, cache_dir)

    logger.info("Champollion pre-computation complete. Cache: %s", cache_dir)


if __name__ == "__main__":
    main()
