"""Tests for SulcalAggregator."""

import pytest
import torch

from brainsulcal.priors.sulcal_aggregator import SulcalAggregator

N_REGIONS = 56
INPUT_DIM = 64
HIDDEN_DIM = 256
BATCH_SIZE = 4


@pytest.fixture
def aggregator():
    return SulcalAggregator(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=4,
        n_layers=2,
    )


def test_output_shapes(aggregator):
    x = torch.randn(BATCH_SIZE, N_REGIONS, INPUT_DIM)
    z_sulcal, R_sulcal = aggregator(x)
    assert z_sulcal.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert R_sulcal.shape == (BATCH_SIZE, N_REGIONS, HIDDEN_DIM)


def test_output_shapes_with_mask(aggregator):
    x = torch.randn(BATCH_SIZE, N_REGIONS, INPUT_DIM)
    mask = torch.ones(BATCH_SIZE, N_REGIONS, dtype=torch.bool)
    mask[:, 50:] = False  # Mask out last 6 regions

    z_sulcal, R_sulcal = aggregator(x, mask)
    assert z_sulcal.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert R_sulcal.shape == (BATCH_SIZE, N_REGIONS, HIDDEN_DIM)


def test_no_nan_with_all_masked(aggregator):
    """All-masked input (no sulcal prior) must not produce NaN."""
    x = torch.zeros(BATCH_SIZE, N_REGIONS, INPUT_DIM)
    mask = torch.zeros(BATCH_SIZE, N_REGIONS, dtype=torch.bool)  # All masked

    z_sulcal, R_sulcal = aggregator(x, mask)
    assert not torch.isnan(z_sulcal).any(), "z_sulcal must not be NaN with all-zero mask"
    assert not torch.isnan(R_sulcal).any(), "R_sulcal must not be NaN with all-zero mask"


def test_no_nan_with_no_mask(aggregator):
    x = torch.randn(BATCH_SIZE, N_REGIONS, INPUT_DIM)
    z_sulcal, R_sulcal = aggregator(x)
    assert not torch.isnan(z_sulcal).any()
    assert not torch.isnan(R_sulcal).any()


def test_different_subjects_different_outputs(aggregator):
    """Different inputs must produce different outputs."""
    x1 = torch.randn(1, N_REGIONS, INPUT_DIM)
    x2 = torch.randn(1, N_REGIONS, INPUT_DIM)
    z1, _ = aggregator(x1)
    z2, _ = aggregator(x2)
    assert not torch.allclose(z1, z2), "Different subjects must have different z_sulcal"


def test_batch_consistency(aggregator):
    """Single-item and batched forward must produce identical results."""
    aggregator.eval()
    x = torch.randn(3, N_REGIONS, INPUT_DIM)

    with torch.no_grad():
        z_batch, _ = aggregator(x)
        z_single_0, _ = aggregator(x[0:1])
        z_single_1, _ = aggregator(x[1:2])

    assert torch.allclose(z_batch[0], z_single_0[0], atol=1e-5)
    assert torch.allclose(z_batch[1], z_single_1[0], atol=1e-5)


def test_gradient_flows(aggregator):
    """Gradients must flow through the aggregator."""
    x = torch.randn(BATCH_SIZE, N_REGIONS, INPUT_DIM, requires_grad=True)
    z_sulcal, _ = aggregator(x)
    loss = z_sulcal.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0), "Gradients should be non-zero"
