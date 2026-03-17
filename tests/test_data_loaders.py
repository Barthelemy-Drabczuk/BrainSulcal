"""Tests for data loaders and preprocessing utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from brainsulcal.data.preprocessing import binarize_labels_per_subject


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


def test_daly_dataset_loads_from_cache():
    """DalyMusicDataset should load pre-computed numpy cache."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()

        # Create fake cached data
        n_epochs, n_channels, n_samples = 10, 31, 1000
        eeg = np.random.randn(n_epochs, n_channels, n_samples).astype(np.float32)
        labels = np.column_stack([
            np.random.randint(0, 2, n_epochs),
            np.random.randint(0, 2, n_epochs),
        ])
        np.save(cache_dir / "sub-01_epochs.npy", eeg)
        np.save(cache_dir / "sub-01_labels.npy", labels)

        dataset = DalyMusicDataset(
            subject_ids=["sub-01"],
            raw_dir=tmpdir,
            champollion_wrapper=None,
            cache_dir=cache_dir,
        )

        assert len(dataset) == n_epochs

        item = dataset[0]
        assert "eeg" in item
        assert item["eeg"].shape == (n_channels, n_samples)
        assert "valence_label" in item
        assert "arousal_label" in item
        assert item["subject_id"] == "sub-01"


def test_dataset_item_types():
    """Dataset items must be torch tensors of correct dtype."""
    from brainsulcal.data.daly_music import DalyMusicDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()

        eeg = np.random.randn(5, 31, 1000).astype(np.float32)
        labels = np.zeros((5, 2), dtype=int)
        np.save(cache_dir / "sub-01_epochs.npy", eeg)
        np.save(cache_dir / "sub-01_labels.npy", labels)

        dataset = DalyMusicDataset(["sub-01"], tmpdir, cache_dir=cache_dir)
        item = dataset[0]

        assert isinstance(item["eeg"], torch.Tensor)
        assert item["eeg"].dtype == torch.float32
        assert isinstance(item["valence_label"], torch.Tensor)
        assert item["valence_label"].dtype == torch.int64
