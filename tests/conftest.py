"""Shared pytest fixtures and configuration."""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "requires_brainomni: tests requiring real BrainOmni checkpoint"
    )
    config.addinivalue_line(
        "markers", "requires_data: tests requiring downloaded dataset"
    )


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def small_eeg_batch():
    """Minimal EEG batch for fast shape tests."""
    return torch.randn(2, 31, 500)  # (B, C, T)
