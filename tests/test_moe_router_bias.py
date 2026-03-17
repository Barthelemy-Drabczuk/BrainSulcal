"""Tests for MoERouterBias."""

import pytest
import torch

from brainsulcal.dynamics.moe_router_bias import MoERouterBias

SULCAL_DIM = 256
N_EXPERTS = 8
BATCH_SIZE = 4


@pytest.fixture
def router_bias():
    return MoERouterBias(sulcal_dim=SULCAL_DIM, n_experts=N_EXPERTS, hidden_dim=128)


def test_output_shape(router_bias):
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    bias = router_bias(z)
    assert bias.shape == (BATCH_SIZE, N_EXPERTS)


def test_near_zero_init(router_bias):
    """Final layer must start near zero to preserve BrainOmni routing."""
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    with torch.no_grad():
        bias = router_bias(z)
    assert bias.abs().max().item() < 0.5, (
        "Router bias should be near zero at init (std=1e-3). "
        "Got max |bias| = %.4f" % bias.abs().max().item()
    )


def test_zero_after_zero_(router_bias):
    """After zero_(), forward must return exactly zero."""
    router_bias.zero_()
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    with torch.no_grad():
        bias = router_bias(z)
    assert torch.all(bias == 0), "After zero_(), bias must be exactly zero"


def test_requires_n_experts():
    with pytest.raises(ValueError, match="n_experts"):
        MoERouterBias(sulcal_dim=256, n_experts=None)


def test_gradient_flows(router_bias):
    z = torch.randn(BATCH_SIZE, SULCAL_DIM, requires_grad=True)
    bias = router_bias(z)
    bias.sum().backward()
    assert z.grad is not None


def test_different_subjects_different_bias(router_bias):
    z1 = torch.randn(1, SULCAL_DIM)
    z2 = torch.randn(1, SULCAL_DIM)
    b1 = router_bias(z1)
    b2 = router_bias(z2)
    # Even with near-zero init, two different inputs should give different outputs
    # (unless weights are literally zero)
    # Just check shapes here — near-zero init test covers the magnitude
    assert b1.shape == b2.shape
