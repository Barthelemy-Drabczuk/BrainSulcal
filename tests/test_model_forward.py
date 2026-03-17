"""Integration tests for BrainSulcal forward pass.

These tests use the BrainOmni stub (no real checkpoint needed) to verify
shapes, invariants, and the zero-bias identity property.
"""

import pytest
import torch

from brainsulcal.model import BrainSulcal, BrainSulcalConfig

# Test dimensions
BATCH_SIZE = 2
N_CHANNELS = 31
N_SAMPLES = 1000    # 4s at 250Hz
N_REGIONS = 56
CHAMPOLLION_DIM = 64
N_EXPERTS = 8
D_MODEL = 512


@pytest.fixture
def config():
    return BrainSulcalConfig(
        champollion_input_dim=CHAMPOLLION_DIM,
        sulcal_hidden_dim=256,
        sulcal_n_heads=4,
        sulcal_n_layers=2,
        n_experts=N_EXPERTS,
        d_model=D_MODEL,
        n_classes=2,
        n_tasks=2,
        use_sulcal_prior=True,
        use_router_bias=True,
        use_prefix_token=True,
    )


@pytest.fixture
def model(config):
    return BrainSulcal(config)


@pytest.fixture
def batch():
    return {
        "eeg": torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES),
        "sulcal_embeddings": torch.randn(BATCH_SIZE, N_REGIONS, CHAMPOLLION_DIM),
        "sulcal_mask": torch.ones(BATCH_SIZE, N_REGIONS, dtype=torch.bool),
        "montage_info": {"type": "eeg", "montage": "standard_1020"},
    }


def test_forward_output_keys(model, batch):
    out = model(
        batch["eeg"], batch["montage_info"],
        batch["sulcal_embeddings"], batch["sulcal_mask"],
    )
    assert "logits_valence" in out
    assert "logits_arousal" in out
    assert "z_sulcal" in out
    assert "z_eeg" in out
    assert "z_fused" in out


def test_output_shapes(model, batch, config):
    out = model(
        batch["eeg"], batch["montage_info"],
        batch["sulcal_embeddings"], batch["sulcal_mask"],
    )
    B = BATCH_SIZE
    assert out["logits_valence"].shape == (B, config.n_classes)
    assert out["logits_arousal"].shape == (B, config.n_classes)
    assert out["z_sulcal"].shape == (B, config.sulcal_hidden_dim)
    assert out["z_eeg"].shape == (B, config.d_model)
    assert out["z_fused"].shape == (B, config.fusion_hidden_dim)


def test_no_nan_output(model, batch):
    out = model(
        batch["eeg"], batch["montage_info"],
        batch["sulcal_embeddings"], batch["sulcal_mask"],
    )
    assert not torch.isnan(out["logits_valence"]).any()
    assert not torch.isnan(out["logits_arousal"]).any()


def test_missing_subject_no_nan(model, config):
    """Zero embeddings with all-False mask must not produce NaN."""
    B = BATCH_SIZE
    eeg = torch.randn(B, N_CHANNELS, N_SAMPLES)
    sulcal_emb = torch.zeros(B, N_REGIONS, CHAMPOLLION_DIM)
    sulcal_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)
    montage_info = {"type": "eeg", "montage": "standard_1020"}

    out = model(eeg, montage_info, sulcal_emb, sulcal_mask)
    assert not torch.isnan(out["logits_valence"]).any(), \
        "Missing subject (zero embeddings) must not produce NaN logits"


def test_brainomni_frozen(model):
    """Invariant: all BrainOmni parameters must be frozen."""
    assert not any(p.requires_grad for p in model.brainomni.parameters()), \
        "BrainOmni parameters must have requires_grad=False"


def test_trainable_params_non_empty(model):
    """There must be trainable parameters (sulcal aggregator etc.)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "Model must have trainable parameters"


def test_zero_bias_identity(config):
    """Zero router bias must produce numerically identical output to vanilla BrainOmni.

    This is a critical correctness invariant: if router bias is zeroed,
    the model must behave exactly like unmodified BrainOmni.
    """
    model = BrainSulcal(config)
    model.eval()

    eeg = torch.randn(1, N_CHANNELS, N_SAMPLES)
    sulcal_emb = torch.zeros(1, N_REGIONS, CHAMPOLLION_DIM)
    sulcal_mask = torch.zeros(1, N_REGIONS, dtype=torch.bool)
    montage_info = {"type": "eeg", "montage": "standard_1020"}

    # Zero out router bias
    if model.moe_router_bias is not None:
        model.moe_router_bias.zero_()

    with torch.no_grad():
        out_conditioned = model(eeg, montage_info, sulcal_emb, sulcal_mask)
        # With zero bias, BrainOmni should route identically to vanilla
        # We verify z_eeg matches — logits will differ due to fusion MLP weights
        # but the EEG representation itself should be identical
        z_eeg_conditioned = out_conditioned["z_eeg"]

        # Run BrainOmni directly without any bias
        token_repr_vanilla = model.brainomni(eeg, montage_info, router_bias=None)
        z_eeg_vanilla = token_repr_vanilla.mean(dim=1)

    assert torch.allclose(z_eeg_conditioned, z_eeg_vanilla, atol=1e-5), \
        "Zero router bias must produce identical EEG representation as vanilla BrainOmni"


def test_ablation_brainomni_only():
    """BrainOmni-only condition: z_sulcal must be None."""
    config = BrainSulcalConfig(
        champollion_input_dim=CHAMPOLLION_DIM,
        n_experts=N_EXPERTS,
        d_model=D_MODEL,
        use_sulcal_prior=False,
        use_router_bias=False,
        use_prefix_token=False,
    )
    model = BrainSulcal(config)

    eeg = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    sulcal_emb = torch.zeros(BATCH_SIZE, N_REGIONS, CHAMPOLLION_DIM)
    sulcal_mask = torch.zeros(BATCH_SIZE, N_REGIONS, dtype=torch.bool)
    montage_info = {"type": "eeg", "montage": "standard_1020"}

    out = model(eeg, montage_info, sulcal_emb, sulcal_mask)
    assert out["z_sulcal"] is None, "BrainOmni-only mode must have z_sulcal=None"
