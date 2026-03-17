"""Dataset loader for the Daly et al. music EEG+fMRI dataset (OpenNeuro ds002725).

N = 21 subjects with paired EEG+fMRI, 114 subjects EEG-only.
Labels: continuous valence/arousal via FEELTRACE, binarized per subject.
EEG: 31 channels, originally 5000 Hz, preprocessed to 250 Hz.

Reference:
  Daly et al., Scientific Reports 2019, doi:10.1038/s41598-019-45105-2
  Daly et al., Scientific Data 2020, doi:10.1038/s41597-020-0507-6
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from brainsulcal.data.montage_utils import channel_pos_sensor_type
from brainsulcal.data.preprocessing import preprocess_eeg, epoch_music_segments
from brainsulcal.priors.champollion_wrapper import ChampollionWrapper

logger = logging.getLogger(__name__)

OPENNEURO_ID = "ds002725"
N_CHANNELS = 31
N_SUBJECTS_EEG_FMRI = 21


class DalyMusicDataset(Dataset):
    """PyTorch dataset for Daly music EEG epochs.

    Each item: one 4-second music epoch with binary valence/arousal labels.

    Args:
        subject_ids:         List of subject IDs (e.g. ["sub-01", "sub-02"]).
        raw_dir:             Path to downloaded OpenNeuro dataset root.
        champollion_wrapper: Pre-initialised wrapper for sulcal embeddings.
        cache_dir:           If provided, load pre-processed epochs from cache.
        config:              Optional preprocessing config dict.
    """

    def __init__(
        self,
        subject_ids: list[str],
        raw_dir: str | Path,
        champollion_wrapper: ChampollionWrapper | None = None,
        cache_dir: str | Path | None = None,
        config: dict | None = None,
    ):
        self.subject_ids = subject_ids
        self.raw_dir = Path(raw_dir)
        self.champollion_wrapper = champollion_wrapper
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.config = config or {}

        # Loaded data: list of (eeg_tensor, label, subject_id, pos, sensor_type)
        # pos: (C, 6) float32, sensor_type: (C,) int64 — or None if not available
        self._items: list[tuple[torch.Tensor, torch.Tensor, str,
                                torch.Tensor | None, torch.Tensor | None]] = []
        self._sulcal_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self._load_all_subjects()

    def _load_all_subjects(self) -> None:
        for subject_id in self.subject_ids:
            items = self._load_subject(subject_id)
            self._items.extend(items)
            logger.info(
                "Loaded %d epochs for %s (total: %d)",
                len(items), subject_id, len(self._items),
            )

        if self.champollion_wrapper is not None:
            for subject_id in self.subject_ids:
                emb, mask = self.champollion_wrapper.load_subject(subject_id)
                self._sulcal_cache[subject_id] = (emb, mask)

    def _load_subject(self, subject_id: str) -> list[tuple]:
        # Try cache first
        if self.cache_dir is not None:
            cache_path  = self.cache_dir / f"{subject_id}_epochs.npy"
            labels_path = self.cache_dir / f"{subject_id}_labels.npy"
            ch_path     = self.cache_dir / f"{subject_id}_channels.txt"
            if cache_path.exists() and labels_path.exists():
                eeg_np    = np.load(cache_path)     # (n_epochs, C, T)
                labels_np = np.load(labels_path)    # (n_epochs, 2)
                pos_t, st_t = None, None
                if ch_path.exists():
                    ch_names = ch_path.read_text().strip().splitlines()
                    pos_t, st_t = _pos_from_ch_names(ch_names)
                return self._arrays_to_items(eeg_np, labels_np, subject_id, pos_t, st_t)

        # Load and preprocess from raw BIDS data
        return self._preprocess_subject(subject_id)

    def _preprocess_subject(
        self, subject_id: str
    ) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        eeg_dir = self.raw_dir / subject_id / "eeg"
        if not eeg_dir.exists():
            logger.warning("EEG directory not found for %s: %s", subject_id, eeg_dir)
            return []

        # Find raw EEG file (BrainVision format in Daly dataset)
        vhdr_files = list(eeg_dir.glob("*task-music*_eeg.vhdr"))
        if not vhdr_files:
            logger.warning("No EEG file found for %s in %s", subject_id, eeg_dir)
            return []

        raw = preprocess_eeg(
            vhdr_files[0],
            bandpass_low=self.config.get("bandpass_low", 1.0),
            bandpass_high=self.config.get("bandpass_high", 40.0),
            target_sfreq=self.config.get("target_sfreq", 250.0),
            reject_threshold_uv=self.config.get("epoch_reject_threshold_uv", 150.0),
        )

        events, event_id, feeltrace_labels = self._load_events_and_labels(
            subject_id, raw
        )
        if events is None:
            return []

        epochs, labels_binary = epoch_music_segments(
            raw, events, event_id, feeltrace_labels,
            reject_threshold_uv=self.config.get("epoch_reject_threshold_uv", 150.0),
        )

        eeg_np = epochs.get_data()  # (n_epochs, n_channels, n_samples)

        # Extract pos / sensor_type from the preprocessed montage
        ch_names = raw.info["ch_names"]
        pos_t, st_t = channel_pos_sensor_type(raw.info)

        # Save to cache
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{subject_id}_epochs.npy", eeg_np)
            np.save(self.cache_dir / f"{subject_id}_labels.npy", labels_binary)
            (self.cache_dir / f"{subject_id}_channels.txt").write_text(
                "\n".join(ch_names)
            )

        return self._arrays_to_items(eeg_np, labels_binary, subject_id, pos_t, st_t)

    def _load_events_and_labels(
        self, subject_id: str, raw: "mne.io.Raw"
    ) -> tuple | tuple[None, None, None]:
        """Load FEELTRACE events and labels from BIDS sidecar files."""
        import mne

        eeg_dir = self.raw_dir / subject_id / "eeg"
        events_file = list(eeg_dir.glob("*task-music*_events.tsv"))
        if not events_file:
            logger.warning("No events file for %s", subject_id)
            return None, None, None

        import pandas as pd
        events_df = pd.read_csv(events_file[0], sep="\t")

        # Build MNE events array from BIDS events.tsv
        sfreq = raw.info["sfreq"]
        onset_samples = (events_df["onset"].values * sfreq).astype(int)
        event_codes = np.ones(len(onset_samples), dtype=int)  # All same event type
        events = np.stack([onset_samples, np.zeros_like(onset_samples), event_codes], axis=1)
        event_id = {"music_segment": 1}

        # FEELTRACE ratings (valence and arousal columns)
        val_col = "valence" if "valence" in events_df.columns else events_df.columns[-2]
        aro_col = "arousal" if "arousal" in events_df.columns else events_df.columns[-1]
        feeltrace_labels = events_df[[val_col, aro_col]].values.astype(float)

        return events, event_id, feeltrace_labels

    @staticmethod
    def _arrays_to_items(
        eeg_np: "np.ndarray",
        labels_np: "np.ndarray",
        subject_id: str,
        pos: "torch.Tensor | None" = None,
        sensor_type: "torch.Tensor | None" = None,
    ) -> list[tuple]:
        items = []
        for i in range(len(eeg_np)):
            eeg_t   = torch.from_numpy(eeg_np[i]).float()
            label_t = torch.from_numpy(labels_np[i]).long()
            items.append((eeg_t, label_t, subject_id, pos, sensor_type))
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:

        eeg, labels, subject_id, pos, sensor_type = self._items[idx]

        item: dict = {
            "eeg": eeg,                              # (C, T)
            "valence_label": labels[0],              # scalar int
            "arousal_label": labels[1],              # scalar int
            "subject_id": subject_id,
        }

        if pos is not None:
            item["pos"] = pos                        # (C, 6)
            item["sensor_type"] = sensor_type        # (C,)

        if self.champollion_wrapper is not None and subject_id in self._sulcal_cache:
            emb, mask = self._sulcal_cache[subject_id]
            item["sulcal_embeddings"] = emb          # (56, d)
            item["sulcal_mask"] = mask               # (56,)

        return item


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _pos_from_ch_names(
    ch_names: list[str],
    montage_name: str = "standard_1020",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BrainOmni-compatible pos/sensor_type from a list of EEG channel names.

    Creates a temporary MNE Info, sets the montage, then delegates to
    montage_utils.channel_pos_sensor_type.

    Args:
        ch_names:     List of EEG channel names.
        montage_name: MNE standard montage name.

    Returns:
        pos:         (C, 6) float32 tensor
        sensor_type: (C,)   int64  tensor (all zeros for EEG)
    """
    import mne
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg", verbose=False)
    montage = mne.channels.make_standard_montage(montage_name)
    with mne.utils.use_log_level("ERROR"):
        info.set_montage(montage, on_missing="ignore")
    return channel_pos_sensor_type(info)


def collate_fn(batch: list[dict]) -> dict:
    """DataLoader collate function for DalyMusicDataset.

    Stacks tensor fields and passes through string/optional fields.
    Optional fields (pos, sensor_type, sulcal_embeddings, sulcal_mask)
    are only included in the output if ALL items in the batch have them.

    Args:
        batch: List of dicts from DalyMusicDataset.__getitem__

    Returns:
        Batched dict with tensors stacked along dim=0.
    """
    out: dict = {}
    keys = batch[0].keys()

    for key in keys:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            out[key] = vals  # keep as list of strings
        # None values skipped (item without pos/sensor_type)

    # Include optional tensor fields only when all items provide them
    for opt_key in ("pos", "sensor_type", "sulcal_embeddings", "sulcal_mask"):
        if opt_key not in keys:
            vals = [item.get(opt_key) for item in batch]
            if all(v is not None for v in vals):
                out[opt_key] = torch.stack(vals, dim=0)

    return out
