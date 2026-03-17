"""Pre-compute BrainOmni tokenizer outputs from preprocessed EEG.

BrainTokenizer is frozen and deterministic — re-running it every epoch wastes
~60% of forward pass time. Cache outputs once:
    data/processed/cache/{subject_id}_tokens.npy  shape: (n_epochs, n_tokens, d_model)

Run ONCE before training. Must have already run preprocessing
(either via DalyMusicDataset or manually).

Usage:
    pixi run precompute-brainomni
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def precompute_tokens(
    subject_id: str,
    eeg_cache_path: Path,
    output_dir: Path,
    brainomni_checkpoint: str,
    montage_info: dict,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Tokenize cached EEG epochs for one subject."""
    from brainsulcal.dynamics.brainomni_wrapper import BrainOmniWrapper

    out_path = output_dir / f"{subject_id}_tokens.npy"
    if out_path.exists():
        logger.info("Cache already exists for %s — skipping.", subject_id)
        return

    eeg_np = np.load(eeg_cache_path)  # (n_epochs, n_channels, n_samples)
    eeg_tensor = torch.from_numpy(eeg_np).float()

    brainomni = BrainOmniWrapper(checkpoint=brainomni_checkpoint, freeze=True).to(device)
    brainomni.eval()

    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(eeg_tensor), batch_size):
            batch = eeg_tensor[i:i + batch_size].to(device)
            tokens = brainomni(batch, montage_info)  # (B, n_tokens, d_model)
            all_tokens.append(tokens.cpu().numpy())

    all_tokens_np = np.concatenate(all_tokens, axis=0)  # (n_epochs, n_tokens, d_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, all_tokens_np)
    logger.info(
        "Saved tokenized EEG for %s: shape=%s", subject_id, all_tokens_np.shape
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute BrainOmni tokenizer outputs.")
    parser.add_argument("--eeg-cache-dir", default="data/processed/cache")
    parser.add_argument("--output-dir", default="data/processed/cache")
    parser.add_argument("--checkpoint", default="OpenTSLab/BrainOmni")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    eeg_cache_dir = Path(args.eeg_cache_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # Build a standard 10-20 montage for the 31 Daly channels
    montage_info = _build_daly_montage()

    if args.subjects:
        subject_ids = args.subjects
    else:
        eeg_files = sorted(eeg_cache_dir.glob("sub-*_epochs.npy"))
        subject_ids = [f.name.replace("_epochs.npy", "") for f in eeg_files]

    logger.info("Pre-computing BrainOmni tokens for %d subjects.", len(subject_ids))

    for subject_id in subject_ids:
        eeg_path = eeg_cache_dir / f"{subject_id}_epochs.npy"
        if not eeg_path.exists():
            logger.warning("EEG cache not found for %s: %s", subject_id, eeg_path)
            continue
        precompute_tokens(
            subject_id, eeg_path, output_dir,
            args.checkpoint, montage_info,
            args.batch_size, device,
        )

    logger.info("BrainOmni pre-computation complete.")


def _build_daly_montage() -> dict:
    """Build montage_info dict for the 31-channel Daly recording.

    Uses MNE standard_1020 positions. If BrainOmni requires a custom format,
    add a custom montage file to external/BrainOmni/share/custom_montages/.
    """
    try:
        import mne
        montage = mne.channels.make_standard_montage("standard_1020")
        # Return positions dict in format expected by BrainTokenizer
        # TODO: verify exact format required by BrainOmni after studying source
        return {
            "type": "eeg",
            "montage": "standard_1020",
            "n_channels": 31,
            "positions": montage.get_positions()["ch_pos"],
        }
    except ImportError:
        return {"type": "eeg", "montage": "standard_1020", "n_channels": 31}


if __name__ == "__main__":
    main()
