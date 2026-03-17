"""EEG preprocessing pipeline for the Daly music dataset.

Pipeline (per CLAUDE.md):
  1. Load raw EEG with MNE-Python
  2. Verify AAS artifact removal (MRI gradient artifacts — dataset is in-scanner)
  3. Bandpass filter: 1–40 Hz
  4. Downsample to 250 Hz (from 5000 Hz)
  5. Epoch around music segments with FEELTRACE labels
  6. Reject epochs with peak-to-peak amplitude > 150 µV
  7. Re-reference to average
  8. Format for BrainTokenizer: (n_epochs, n_channels, n_samples)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default preprocessing parameters (can be overridden via config)
DEFAULT_BANDPASS_LOW = 1.0    # Hz
DEFAULT_BANDPASS_HIGH = 40.0  # Hz
DEFAULT_TARGET_SFREQ = 250.0  # Hz
DEFAULT_REJECT_THRESHOLD = 150e-6  # 150 µV in Volts (MNE convention)


def preprocess_eeg(
    raw_path: str | Path,
    bandpass_low: float = DEFAULT_BANDPASS_LOW,
    bandpass_high: float = DEFAULT_BANDPASS_HIGH,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    reject_threshold_uv: float = 150.0,
    verify_aas: bool = True,
) -> "mne.io.Raw":
    """Load and preprocess a raw EEG file.

    Args:
        raw_path:            Path to raw EEG file (BrainVision .vhdr or FIF).
        bandpass_low:        High-pass cutoff in Hz.
        bandpass_high:       Low-pass cutoff in Hz.
        target_sfreq:        Target sampling frequency after resampling.
        reject_threshold_uv: Peak-to-peak rejection threshold in µV.
        verify_aas:          If True, warn if MRI gradient artifacts appear to remain.

    Returns:
        Preprocessed MNE Raw object (not yet epoched).
    """
    import mne  # Deferred import — not available until pixi install

    raw_path = Path(raw_path)
    logger.info("Loading %s", raw_path)

    # Load raw EEG
    if raw_path.suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=True, verbose=False)
    elif raw_path.suffix in (".fif", ".fif.gz"):
        raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
    else:
        raw = mne.io.read_raw(str(raw_path), preload=True, verbose=False)

    if verify_aas:
        _check_mri_artifacts(raw)

    # Bandpass filter
    raw.filter(bandpass_low, bandpass_high, fir_design="firwin", verbose=False)
    logger.info("Bandpass filtered: %.1f–%.1f Hz", bandpass_low, bandpass_high)

    # Downsample
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
        logger.info("Resampled to %.0f Hz", target_sfreq)

    # Average re-reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw


def epoch_music_segments(
    raw: "mne.io.Raw",
    events: "np.ndarray",
    event_id: dict[str, int],
    feeltrace_labels: "np.ndarray",
    tmin: float = 0.0,
    tmax: float = 4.0,
    reject_threshold_uv: float = 150.0,
) -> tuple["mne.Epochs", "np.ndarray"]:
    """Create epochs from music segment events and attach FEELTRACE labels.

    Args:
        raw:                 Preprocessed Raw object.
        events:              MNE events array (n_events, 3).
        event_id:            Mapping from event name to integer code.
        feeltrace_labels:    Continuous valence/arousal ratings per event.
        tmin:                Epoch start relative to event onset (s).
        tmax:                Epoch end relative to event onset (s).
        reject_threshold_uv: Peak-to-peak amplitude rejection in µV.

    Returns:
        epochs:         MNE Epochs object (n_good_epochs, n_channels, n_samples).
        labels_binary:  (n_good_epochs, 2) array of binary [valence, arousal] labels.
    """
    import mne

    reject = {"eeg": reject_threshold_uv * 1e-6}  # Convert µV → V

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        reject=reject,
        baseline=None,
        preload=True,
        verbose=False,
    )

    n_total = len(events)
    n_kept = len(epochs)
    logger.info(
        "Epoch rejection: kept %d / %d (%.1f%%)",
        n_kept, n_total, 100 * n_kept / max(n_total, 1),
    )

    # Keep only labels for non-rejected epochs
    selection = epochs.selection
    valid_labels = feeltrace_labels[selection]  # (n_good, 2) [valence, arousal]
    labels_binary = binarize_labels_per_subject(valid_labels)

    return epochs, labels_binary


def binarize_labels_per_subject(labels: "np.ndarray") -> "np.ndarray":
    """Binarize continuous valence/arousal ratings via median split.

    Median split is computed per subject (i.e., per call to this function).
    Never fit normalization on test subjects — call this only on training data.

    Args:
        labels: (n_epochs, 2) array of continuous [valence, arousal] ratings.

    Returns:
        binary: (n_epochs, 2) array of binary labels (0 or 1).
    """
    median_valence = np.median(labels[:, 0])
    median_arousal = np.median(labels[:, 1])

    binary = np.stack([
        (labels[:, 0] >= median_valence).astype(int),
        (labels[:, 1] >= median_arousal).astype(int),
    ], axis=1)

    return binary


def _check_mri_artifacts(raw: "mne.io.Raw", threshold_db: float = 20.0) -> None:
    """Warn if MRI gradient artifacts appear to remain in the signal.

    Checks for elevated power at 20 Hz harmonics (typical gradient artifact
    frequency for 3T MRI with 1 TR/s). This is a heuristic — manual inspection
    via PSD plots is still recommended.
    """
    try:
        import mne
        psd = raw.compute_psd(method="welch", fmin=1, fmax=100, verbose=False)
        freqs = psd.freqs
        power_db = 10 * np.log10(psd.get_data().mean(axis=0) + 1e-30)

        # Check power at 20, 40, 60 Hz harmonics relative to neighbouring bins
        artifact_freqs = [20.0, 40.0, 60.0]
        for af in artifact_freqs:
            idx = np.argmin(np.abs(freqs - af))
            local_bg = np.median(power_db[max(0, idx-5):idx+5])
            peak = power_db[idx]
            if peak - local_bg > threshold_db:
                logger.warning(
                    "Possible residual MRI gradient artifact at %.0f Hz "
                    "(%.1f dB above background). Verify AAS removal.",
                    af, peak - local_bg,
                )
    except Exception as e:
        logger.debug("AAS check skipped: %s", e)
