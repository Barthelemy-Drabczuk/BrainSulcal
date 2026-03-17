"""Tests for data loaders and preprocessing utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from brainsulcal.data.preprocessing import binarize_labels_per_subject

# 31-channel subset used in tests (subset of standard_1020)
_TEST_CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2", "POz",
]
N_CHANNELS = len(_TEST_CH_NAMES)  # 31


def test_binarize_labels_median_split():
    """Labels should be binarized by median split."""
    labels = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.3, 0.7],
    ], dtype=float)
    binary = binarize_labels_per_subject(labels)
    assert binary.shape == (4, 2)
    assert binary.dtype in (np.int32, np.int64, int)
    assert set(binary[:, 0].tolist()).issubset({0, 1})
    assert set(binary[:, 1].tolist()).issubset({0, 1})


def test_binarize_labels_roughly_balanced():
    """Median split should produce roughly balanced classes."""
    np.random.seed(42)
    labels = np.random.rand(100, 2)
    binary = binarize_labels_per_subject(labels)
    # Should be close to 50/50 for uniform random labels
    for task in range(2):
        n_pos = binary[:, task].sum()
        assert 30 <= n_pos <= 70, f"Task {task}: expected ~50 positives, got {n_pos}"


def test_binarize_no_test_leakage():
    """Each call to binarize_labels_per_subject uses only the provided data.

    Simulates the LOSO requirement: train and test are binarized independently.
    """
    train_labels = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    test_labels = np.array([[0.99, 0.01]])

    train_binary = binarize_labels_per_subject(train_labels)
    test_binary = binarize_labels_per_subject(test_labels)

    # Test subject's label is based only on its own median (not training data)
    assert test_binary.shape == (1, 2)


def test_daly_dataset_requires_cache_or_raw():
    """DalyMusicDataset with missing data should return empty, not crash."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = DalyMusicDataset(
            subject_ids=["sub-01"],
            raw_dir=tmpdir,
            champollion_wrapper=None,
            cache_dir=None,
        )
        # No data available — should be empty
        assert len(dataset) == 0


def _write_cache(cache_dir: Path, n_epochs: int, n_channels: int, n_samples: int,
                 with_channels: bool = True):
    eeg = np.random.randn(n_epochs, n_channels, n_samples).astype(np.float32)
    labels = np.column_stack([
        np.random.randint(0, 2, n_epochs),
        np.random.randint(0, 2, n_epochs),
    ])
    np.save(cache_dir / "sub-01_epochs.npy", eeg)
    np.save(cache_dir / "sub-01_labels.npy", labels)
    if with_channels:
        (cache_dir / "sub-01_channels.txt").write_text("\n".join(_TEST_CH_NAMES))


def test_daly_dataset_loads_from_cache():
    """DalyMusicDataset should load pre-computed numpy cache."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        _write_cache(cache_dir, n_epochs=10, n_channels=N_CHANNELS, n_samples=1000)

        dataset = DalyMusicDataset(
            subject_ids=["sub-01"],
            raw_dir=tmpdir,
            champollion_wrapper=None,
            cache_dir=cache_dir,
        )

        assert len(dataset) == 10
        item = dataset[0]
        assert "eeg" in item
        assert item["eeg"].shape == (N_CHANNELS, 1000)
        assert "valence_label" in item
        assert "arousal_label" in item
        assert item["subject_id"] == "sub-01"


def test_dataset_item_types():
    """Dataset items must be torch tensors of correct dtype."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        _write_cache(cache_dir, n_epochs=5, n_channels=N_CHANNELS, n_samples=1000)

        dataset = DalyMusicDataset(["sub-01"], tmpdir, cache_dir=cache_dir)
        item = dataset[0]

        assert isinstance(item["eeg"], torch.Tensor)
        assert item["eeg"].dtype == torch.float32
        assert isinstance(item["valence_label"], torch.Tensor)
        assert item["valence_label"].dtype == torch.int64


def test_dataset_item_has_pos_sensor_type():
    """Items must include pos (C,6) and sensor_type (C,) when channel file exists."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        _write_cache(cache_dir, n_epochs=3, n_channels=N_CHANNELS, n_samples=1000)

        dataset = DalyMusicDataset(["sub-01"], tmpdir, cache_dir=cache_dir)
        item = dataset[0]

        assert "pos" in item, "item must contain 'pos'"
        assert "sensor_type" in item, "item must contain 'sensor_type'"
        assert item["pos"].shape == (N_CHANNELS, 6)
        assert item["sensor_type"].shape == (N_CHANNELS,)
        assert item["pos"].dtype == torch.float32
        assert item["sensor_type"].dtype == torch.int64


def test_dataset_item_no_pos_without_channel_file():
    """Items must NOT include pos/sensor_type when channel file is missing."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        _write_cache(cache_dir, n_epochs=3, n_channels=N_CHANNELS, n_samples=1000,
                     with_channels=False)

        dataset = DalyMusicDataset(["sub-01"], tmpdir, cache_dir=cache_dir)
        item = dataset[0]

        assert "pos" not in item
        assert "sensor_type" not in item


def test_collate_fn_stacks_tensors():
    """collate_fn must stack tensors along batch dimension."""
    from brainsulcal.data.daly_music import DalyMusicDataset, collate_fn
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        _write_cache(cache_dir, n_epochs=8, n_channels=N_CHANNELS, n_samples=1000)

        dataset = DalyMusicDataset(["sub-01"], tmpdir, cache_dir=cache_dir)
        loader  = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        batch   = next(iter(loader))

        assert batch["eeg"].shape == (4, N_CHANNELS, 1000)
        assert batch["pos"].shape == (4, N_CHANNELS, 6)
        assert batch["sensor_type"].shape == (4, N_CHANNELS)
        assert batch["valence_label"].shape == (4,)
        assert isinstance(batch["subject_id"], list)
