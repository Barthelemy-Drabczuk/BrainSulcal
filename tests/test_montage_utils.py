"""Tests for montage_utils — BrainOmni pos/sensor_type tensor generation."""

import numpy as np
import pytest
import torch

from brainsulcal.data.montage_utils import (
    SENSOR_TYPE_EEG,
    channel_pos_sensor_type,
    make_batch,
)

# 31-channel subset matching the Daly dataset (standard 10-20 BrainAmp MR)
DALY_CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    "POz",
]

N_CHANNELS = len(DALY_CH_NAMES)  # 31
BATCH_SIZE = 4


@pytest.fixture
def daly_info():
    """MNE Info for DALY_CH_NAMES with standard_1020 montage."""
    import mne
    info = mne.create_info(
        ch_names=DALY_CH_NAMES,
        sfreq=256,
        ch_types=["eeg"] * N_CHANNELS,
        verbose=False,
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    with mne.utils.use_log_level("ERROR"):
        info.set_montage(montage)
    return info


def test_pos_shape(daly_info):
    pos, st = channel_pos_sensor_type(daly_info)
    assert pos.shape == (N_CHANNELS, 6), f"Expected ({N_CHANNELS}, 6), got {pos.shape}"
    assert st.shape == (N_CHANNELS,), f"Expected ({N_CHANNELS},), got {st.shape}"


def test_sensor_type_all_eeg(daly_info):
    _, st = channel_pos_sensor_type(daly_info)
    assert (st == SENSOR_TYPE_EEG).all(), "All Daly channels must be EEG (type=0)"
    assert st.dtype == torch.int64


def test_pos_dtype(daly_info):
    pos, _ = channel_pos_sensor_type(daly_info)
    assert pos.dtype == torch.float32


def test_orientation_zeros_for_eeg(daly_info):
    """EEG channels must have zero orientation (last 3 columns of pos)."""
    pos, _ = channel_pos_sensor_type(daly_info)
    assert (pos[:, 3:] == 0).all(), "EEG orientation columns must be zero"


def test_pos_normalized_mean_approx_zero(daly_info):
    """After normalization, xyz centroid must be near zero."""
    pos, _ = channel_pos_sensor_type(daly_info)
    xyz_mean = pos[:, :3].mean(dim=0)
    assert xyz_mean.abs().max().item() < 1e-5, (
        f"xyz centroid not near zero: {xyz_mean}"
    )


def test_pos_no_nan(daly_info):
    pos, st = channel_pos_sensor_type(daly_info)
    assert not torch.isnan(pos).any()
    assert not torch.isnan(st.float()).any()


def test_make_batch_shapes(daly_info):
    pos, st = channel_pos_sensor_type(daly_info)
    pos_b, st_b = make_batch(pos, st, BATCH_SIZE)
    assert pos_b.shape == (BATCH_SIZE, N_CHANNELS, 6)
    assert st_b.shape  == (BATCH_SIZE, N_CHANNELS)


def test_make_batch_values_identical(daly_info):
    """All batch items must share the same electrode layout."""
    pos, st = channel_pos_sensor_type(daly_info)
    pos_b, st_b = make_batch(pos, st, BATCH_SIZE)
    for b in range(BATCH_SIZE):
        assert torch.equal(pos_b[b], pos), "Batch items must have identical pos"
        assert torch.equal(st_b[b], st),  "Batch items must have identical sensor_type"
