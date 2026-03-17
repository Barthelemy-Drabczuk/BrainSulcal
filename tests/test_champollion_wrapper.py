"""Tests for ChampollionWrapper."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from brainsulcal.priors.champollion_wrapper import ChampollionWrapper

N_REGIONS = 56
EMBEDDING_DIM = 64


@pytest.fixture
def tmp_embeddings_dir():
    """Create a temporary directory with fake Champollion embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        for subject_id in ["sub-01", "sub-02"]:
            emb = np.random.randn(N_REGIONS, EMBEDDING_DIM).astype(np.float32)
            mask = np.ones(N_REGIONS, dtype=bool)
            np.save(path / f"{subject_id}.npy", emb)
            np.save(path / f"{subject_id}_mask.npy", mask)
        yield path


def test_load_subject_shape(tmp_embeddings_dir):
    wrapper = ChampollionWrapper(tmp_embeddings_dir)
    emb, mask = wrapper.load_subject("sub-01")
    assert emb.shape == (N_REGIONS, EMBEDDING_DIM)
    assert mask.shape == (N_REGIONS,)
    assert emb.dtype == torch.float32
    assert mask.dtype == torch.bool


def test_load_subject_all_valid(tmp_embeddings_dir):
    wrapper = ChampollionWrapper(tmp_embeddings_dir)
    _, mask = wrapper.load_subject("sub-01")
    assert mask.all(), "All regions should be valid for complete embedding"


def test_missing_subject_returns_zeros(tmp_embeddings_dir):
    wrapper = ChampollionWrapper(tmp_embeddings_dir)
    # Load a valid subject first to set embedding_dim
    wrapper.load_subject("sub-01")
    emb, mask = wrapper.load_subject("sub-NONEXISTENT")
    assert emb.shape == (N_REGIONS, EMBEDDING_DIM)
    assert not mask.any(), "All regions should be invalid for missing subject"
    assert torch.all(emb == 0), "Missing subject should return zero embeddings"


def test_masked_positions_are_zero(tmp_embeddings_dir):
    """Masked regions must be zeroed out to prevent NaN propagation."""
    # Create embedding with partial mask
    path = tmp_embeddings_dir
    emb = np.random.randn(N_REGIONS, EMBEDDING_DIM).astype(np.float32)
    mask = np.ones(N_REGIONS, dtype=bool)
    mask[5:10] = False  # Mark some regions as missing

    np.save(path / "sub-partial.npy", emb)
    np.save(path / "sub-partial_mask.npy", mask)

    wrapper = ChampollionWrapper(path)
    loaded_emb, loaded_mask = wrapper.load_subject("sub-partial")

    assert not loaded_mask[5:10].any()
    assert torch.all(loaded_emb[5:10] == 0), "Masked regions must be zero"


def test_load_batch(tmp_embeddings_dir):
    wrapper = ChampollionWrapper(tmp_embeddings_dir)
    embs, masks = wrapper.load_batch(["sub-01", "sub-02"])
    assert embs.shape == (2, N_REGIONS, EMBEDDING_DIM)
    assert masks.shape == (2, N_REGIONS)


def test_embedding_dim_property(tmp_embeddings_dir):
    wrapper = ChampollionWrapper(tmp_embeddings_dir)
    wrapper.load_subject("sub-01")
    assert wrapper.embedding_dim == EMBEDDING_DIM
