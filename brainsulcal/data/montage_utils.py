"""BrainOmni-compatible pos and sensor_type tensor generation.

Replicates the exact preprocessing from
  external/BrainOmni/factory/utils.py :: extract_pos_sensor_type + normalize_pos

so that pos and sensor_type fed to BrainOmniWrapper.encode() match the
distribution seen during BrainOmni pre-training.

For EEG (Daly dataset, 31 channels):
  pos:         (C, 6) float32 — xyz electrode positions (meters, centered and
               RMS-scaled) + three zeros (no orientation vector for EEG)
  sensor_type: (C,)   int64   — all zeros (EEG sensor type code = 0)

Normalization (copy of BrainOmni factory/utils.py::normalize_pos):
  mu    = mean(xyz over C channels)
  scale = sqrt(3 * mean(sum_per_channel(xyz_centered^2)))
  pos[:, :3] = (pos[:, :3] - mu) / scale

Usage:
    pos_t, st_t = channel_pos_sensor_type(raw.info)    # (C,6), (C,)
    pos_batch   = pos_t.unsqueeze(0).expand(B, -1, -1) # (B,C,6)
    st_batch    = st_t.unsqueeze(0).expand(B, -1)      # (B,C)
"""

from __future__ import annotations

import numpy as np
import torch

# Sensor type codes — must match BrainOmni factory/brain_constant.py
SENSOR_TYPE_EEG = 0
SENSOR_TYPE_MAG = 1
SENSOR_TYPE_GRAD = 2


def channel_pos_sensor_type(
    info,  # mne.Info
    montage_name: str = "standard_1020",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and normalise pos/sensor_type from an MNE Info object.

    Replicates BrainOmni's extract_pos_sensor_type + normalize_pos exactly.

    Args:
        info:         mne.Info with channel positions set (after set_montage).
        montage_name: Ignored (montage already applied to info). Kept for
                      documentation — caller must call raw.set_montage() first.

    Returns:
        pos:         (C, 6) float32 tensor — normalized xyz + orientation
        sensor_type: (C,)   int64  tensor — sensor type codes
    """
    pos_np, sensor_type_np = _extract_pos_sensor_type(info)
    eeg_mask = sensor_type_np == SENSOR_TYPE_EEG
    mag_mask = sensor_type_np == SENSOR_TYPE_MAG
    meg_mask = mag_mask | (sensor_type_np == SENSOR_TYPE_GRAD)
    pos_np = _normalize_pos(pos_np, eeg_mask, meg_mask)

    pos_t = torch.from_numpy(pos_np).float()            # (C, 6)
    sensor_type_t = torch.from_numpy(sensor_type_np.astype(np.int64))  # (C,)
    return pos_t, sensor_type_t


def make_batch(
    pos: torch.Tensor,
    sensor_type: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand (C,6) and (C,) tensors to batch dimension (B,C,6) and (B,C).

    The same electrode layout is shared across all items in the batch —
    appropriate when all items come from the same recording montage.

    Args:
        pos:         (C, 6) tensor from channel_pos_sensor_type()
        sensor_type: (C,)   tensor from channel_pos_sensor_type()
        batch_size:  B

    Returns:
        pos_batch:  (B, C, 6)
        st_batch:   (B, C)
    """
    pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)   # (B, C, 6)
    st_batch  = sensor_type.unsqueeze(0).expand(batch_size, -1)  # (B, C)
    return pos_batch, st_batch


# ---------------------------------------------------------------------------
# Internal helpers (exact copies of BrainOmni factory/utils.py logic)
# ---------------------------------------------------------------------------

def _extract_pos_sensor_type(info) -> tuple[np.ndarray, np.ndarray]:
    """Replicate BrainOmni factory/utils.py::extract_pos_sensor_type.

    For EEG channels: pos = [xyz, 0, 0, 0]  (no orientation)
    For MEG MAG:      pos = [xyz, loc[3:6]]
    For MEG GRAD:     pos = [xyz, loc[3:6]]  (planar: dir_idx=1 → loc[3:6])
    """
    pos = []
    sensor_type = []

    for ch in info["chs"]:
        kind = int(ch["kind"])
        coil_type = str(ch["coil_type"])

        if kind == 2:  # FIFFV_EEG_CH
            pos.append(np.hstack([ch["loc"][:3], np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_EEG)
        elif kind == 1:  # FIFFV_MEG_CH
            xyz = ch["loc"][:3]
            dir_idx = 1 if "PLANAR" in coil_type else 3
            direction = ch["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos.append(np.hstack([xyz, direction]))
            if "MAG" in coil_type:
                sensor_type.append(SENSOR_TYPE_MAG)
            else:
                sensor_type.append(SENSOR_TYPE_GRAD)
        else:
            raise ValueError(
                f"Unknown channel kind {kind} for channel '{ch['ch_name']}'. "
                "Only EEG (kind=2) and MEG (kind=1) are supported."
            )

    pos_arr = np.stack(pos).astype(np.float32)             # (C, 6)
    st_arr  = np.array(sensor_type, dtype=np.int32)        # (C,)
    return pos_arr, st_arr


def _normalize_pos(
    pos: np.ndarray,
    eeg_mask: np.ndarray,
    meg_mask: np.ndarray,
) -> np.ndarray:
    """Replicate BrainOmni factory/utils.py::normalize_pos.

    Independently centers and RMS-scales EEG and MEG xyz positions.
    Orientation columns (pos[:, 3:]) are left unchanged.
    """
    pos = pos.copy()

    if eeg_mask.any():
        mu = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= mu
        scale = np.sqrt(3.0 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        if scale > 0:
            pos[eeg_mask, :3] /= scale

    if meg_mask.any():
        mu = np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] -= mu
        scale = np.sqrt(3.0 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
        if scale > 0:
            pos[meg_mask, :3] /= scale

    return pos
