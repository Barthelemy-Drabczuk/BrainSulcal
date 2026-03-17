"""Integration tests for BrainSulcal forward pass.

Uses BrainOmni stub (no checkpoint needed) to verify shapes, invariants,
and the zero-bias identity property.

BrainOmni API (from source):
  encode(x, pos, sensor_type) → (B, C, W, lm_dim)
  x:           (B, C, W*T) windowed EEG
  pos:         (B, C, 6)   electrode positions
  sensor_type: (B, C)      sensor type codes
"""

import pytest
import torch

from brainsulcal.model import BrainSulcal, BrainSulcalConfig

# Test dimensions (matching BrainOmni stub defaults)
BATCH_SIZE = 2
N_CHANNELS = 31
N_WINDOWS = 8
N_SAMPLES_PER_WINDOW = 125    # 0.5s at 250 Hz
N_REGIONS = 56
CHAMPOLLION_DIM = 64
N_DIM = 256      # BrainOmni tokenizer n_dim
LM_DIM = 512     # BrainOmni backbone lm_dim


def make_eeg_batch(B=BATCH_SIZE, C=N_CHANNELS):
    """Create windowed EEG in BrainOmni format."""
    return torch.randn(B, C, N_WINDOWS * N_SAMPLES_PER_WINDOW)


def make_pos(B=BATCH_SIZE, C=N_CHANNELS):
    return torch.randn(B, C, 6)


def make_sensor_type(B=BATCH_SIZE, C=N_CHANNELS):
    return torch.zeros(B, C, dtype=torch.long)


def make_sulcal(B=BATCH_SIZE):
    return (
        torch.randn(B, N_REGIONS, CHAMPOLLION_DIM),
        torch.ones(B, N_REGIONS, dtype=torch.bool),
    )


@pytest.fixture
def config():
    return BrainSulcalConfig(
        champollion_input_dim=CHAMPOLLION_DIM,
        sulcal_hidden_dim=256,
        sulcal_n_heads=4,
        sulcal_n_layers=2,
        n_dim=N_DIM,
        lm_dim=LM_DIM,
        n_classes=2,
        n_tasks=2,
        use_sulcal_prior=True,
        use_router_bias=True,
        use_prefix_token=True,
    )


@pytest.fixture
def model(config):
    return BrainSulcal(config)


def test_forward_output_keys(model):
    eeg, pos, st = make_eeg_batch(), make_pos(), make_sensor_type()
    sulcal_emb, sulcal_mask = make_sulcal()
    out = model(eeg, pos, st, sulcal_emb, sulcal_mask)
    assert "logits_valence" in out
    assert "logits_arousal" in out
    assert "z_sulcal" in out
    assert "z_eeg" in out
    assert "z_fused" in out


def test_output_shapes(model, config):
    B = BATCH_SIZE
    eeg, pos, st = make_eeg_batch(), make_pos(), make_sensor_type()
    sulcal_emb, sulcal_mask = make_sulcal()
    out = model(eeg, pos, st, sulcal_emb, sulcal_mask)

    assert out["logits_valence"].shape == (B, config.n_classes)
    assert out["logits_arousal"].shape == (B, config.n_classes)
    assert out["z_sulcal"].shape == (B, config.sulcal_hidden_dim)
    assert out["z_eeg"].shape == (B, config.lm_dim)
    assert out["z_fused"].shape == (B, config.fusion_hidden_dim)


def test_no_nan_output(model):
    eeg, pos, st = make_eeg_batch(), make_pos(), make_sensor_type()
    sulcal_emb, sulcal_mask = make_sulcal()
    out = model(eeg, pos, st, sulcal_emb, sulcal_mask)
    assert not torch.isnan(out["logits_valence"]).any()
    assert not torch.isnan(out["logits_arousal"]).any()
    assert not torch.isnan(out["z_eeg"]).any()


def test_missing_subject_no_nan(model, config):
    """Zero embeddings + all-False mask (no sulcal prior) must not produce NaN."""
    B = BATCH_SIZE
    eeg, pos, st = make_eeg_batch(), make_pos(), make_sensor_type()
    sulcal_emb = torch.zeros(B, N_REGIONS, CHAMPOLLION_DIM)
    sulcal_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool)  # all invalid

    out = model(eeg, pos, st, sulcal_emb, sulcal_mask)
    assert not torch.isnan(out["logits_valence"]).any(), \
        "Missing subject must not produce NaN logits"


def test_brainomni_frozen(model):
    """All BrainOmni parameters must have requires_grad=False."""
    assert not any(p.requires_grad for p in model.brainomni.parameters()), \
        "BrainOmni parameters must be frozen"


def test_trainable_params_non_empty(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_zero_bias_identity(config):
    """Zero channel bias → BrainOmniWrapper.encode() identical to vanilla.

    Tests the wrapper-level identity invariant: when channel_bias is all-zero,
    BrainOmniWrapper.encode() must be numerically identical to encode() with
    channel_bias=None.

    Note: the full model's z_eeg may differ due to the prefix virtual channel
    (which adds z_sulcal signal). The identity invariant applies specifically
    to the channel bias injection in the wrapper.
    """
    model = BrainSulcal(config)
    model.eval()

    eeg, pos, st = make_eeg_batch(1), make_pos(1), make_sensor_type(1)
    zero_bias = torch.zeros(1, 1, 1, config.n_dim)

    with torch.no_grad():
        feat_with_zero_bias = model.brainomni.encode(eeg, pos, st, channel_bias=zero_bias)
        feat_vanilla = model.brainomni.encode(eeg, pos, st, channel_bias=None)

    assert torch.allclose(feat_with_zero_bias, feat_vanilla, atol=1e-5), \
        "Zero channel_bias must produce identical encode() output as channel_bias=None"


def test_ablation_brainomni_only():
    """BrainOmni-only condition: z_sulcal must be None."""
    config = BrainSulcalConfig(
        champollion_input_dim=CHAMPOLLION_DIM,
        n_dim=N_DIM,
        lm_dim=LM_DIM,
        use_sulcal_prior=False,
        use_router_bias=False,
        use_prefix_token=False,
    )
    model = BrainSulcal(config)
    eeg, pos, st = make_eeg_batch(), make_pos(), make_sensor_type()
    sulcal_emb = torch.zeros(BATCH_SIZE, N_REGIONS, CHAMPOLLION_DIM)
    sulcal_mask = torch.zeros(BATCH_SIZE, N_REGIONS, dtype=torch.bool)

    out = model(eeg, pos, st, sulcal_emb, sulcal_mask)
    assert out["z_sulcal"] is None


def test_moe_router_bias_output_shape():
    """MoERouterBias must output (B, 1, 1, n_dim) for correct broadcast."""
    from brainsulcal.dynamics.moe_router_bias import MoERouterBias
    bias_net = MoERouterBias(sulcal_dim=256, n_dim=N_DIM)
    z = torch.randn(BATCH_SIZE, 256)
    out = bias_net(z)
    assert out.shape == (BATCH_SIZE, 1, 1, N_DIM), \
        f"Expected (B, 1, 1, n_dim), got {out.shape}"
