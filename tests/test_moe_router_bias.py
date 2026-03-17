"""Tests for MoERouterBias (sulcal channel embedding bias)."""

import pytest
import torch

from brainsulcal.dynamics.moe_router_bias import MoERouterBias

SULCAL_DIM = 256
N_DIM = 256   # BrainOmni tokenizer n_dim
BATCH_SIZE = 4


@pytest.fixture
def bias_net():
    return MoERouterBias(sulcal_dim=SULCAL_DIM, n_dim=N_DIM, hidden_dim=128)


def test_output_shape(bias_net):
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    out = bias_net(z)
    assert out.shape == (BATCH_SIZE, 1, 1, N_DIM), \
        f"Expected (B, 1, 1, n_dim), got {out.shape}"


def test_near_zero_init(bias_net):
    """Final layer must start near zero to preserve BrainOmni representations."""
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    with torch.no_grad():
        out = bias_net(z)
    assert out.abs().max().item() < 0.5, (
        "Channel bias should be near zero at init (std=1e-3). "
        "Got max |bias| = %.4f" % out.abs().max().item()
    )


def test_zero_after_zero_(bias_net):
    """After zero_(), forward must return exactly zero."""
    bias_net.zero_()
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    with torch.no_grad():
        out = bias_net(z)
    assert torch.all(out == 0), "After zero_(), bias must be exactly zero"


def test_requires_n_dim():
    with pytest.raises(ValueError, match="n_dim"):
        MoERouterBias(sulcal_dim=256, n_dim=None)


def test_gradient_flows(bias_net):
    z = torch.randn(BATCH_SIZE, SULCAL_DIM, requires_grad=True)
    out = bias_net(z)
    out.sum().backward()
    assert z.grad is not None


def test_broadcastable_over_channels(bias_net):
    """Output must broadcast correctly over (B, C, W, n_dim)."""
    z = torch.randn(BATCH_SIZE, SULCAL_DIM)
    bias = bias_net(z)  # (B, 1, 1, n_dim)

    C, W = 31, 8
    token_emb = torch.randn(BATCH_SIZE, C, W, N_DIM)
    result = token_emb + bias  # should broadcast without error
    assert result.shape == (BATCH_SIZE, C, W, N_DIM)
