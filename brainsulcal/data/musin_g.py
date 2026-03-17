"""Dataset loader for the MUSIN-G EEG music dataset (OpenNeuro ds003774).

N = 20 subjects, 128 channels, 250 Hz.
Labels: familiarity (1-5) and enjoyment (1-5) per song, 12 genres.
Used for pre-training the sulcal aggregator in a genre classification setting.

Reference:
  Miyapuram et al. 2022, OpenNeuro ds003774
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

OPENNEURO_ID = "ds003774"
N_SUBJECTS = 20
N_CHANNELS = 128
SFREQ = 250.0
N_GENRES = 12


class MusinGDataset(Dataset):
    """PyTorch dataset for MUSIN-G EEG genre classification.

    Each item: one EEG segment with genre label and subjective ratings.

    Args:
        subject_ids:  List of subject IDs.
        raw_dir:      Path to downloaded OpenNeuro dataset root.
        cache_dir:    If provided, load from pre-processed cache.
        task:         'genre' (default) or 'enjoyment' or 'familiarity'.
    """

    def __init__(
        self,
        subject_ids: list[str],
        raw_dir: str | Path,
        cache_dir: str | Path | None = None,
        task: str = "genre",
    ):
        self.subject_ids = subject_ids
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.task = task

        self._items: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        self._load_all_subjects()

    def _load_all_subjects(self) -> None:
        for subject_id in self.subject_ids:
            items = self._load_subject(subject_id)
            self._items.extend(items)
            logger.info("Loaded %d segments for %s", len(items), subject_id)

    def _load_subject(self, subject_id: str) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"musing_{subject_id}_eeg.npy"
            labels_path = self.cache_dir / f"musing_{subject_id}_labels.npy"
            if cache_path.exists() and labels_path.exists():
                eeg_np = np.load(cache_path)
                labels_np = np.load(labels_path)
                return [
                    (torch.from_numpy(eeg_np[i]).float(),
                     torch.tensor(labels_np[i]).long(),
                     subject_id)
                    for i in range(len(eeg_np))
                ]

        logger.warning(
            "No cache found for %s in MUSIN-G. Run scripts/00_download_data.py first.",
            subject_id,
        )
        return []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        eeg, label, subject_id = self._items[idx]
        return {
            "eeg": eeg,
            "label": label,
            "subject_id": subject_id,
        }
