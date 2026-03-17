"""Prefix fusion: prepend z_sulcal as a prefix token to BrainOmni's token sequence.

z_sulcal (B, sulcal_dim) is projected to BrainOmni's d_model, then prepended
to the token sequence before the downstream task head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrefixFusion(nn.Module):
    """Project z_sulcal and prepend as a prefix token.

    Args:
        sulcal_dim:  Dimensionality of z_sulcal (default 256).
        d_model:     BrainOmni token dimension (must match exactly).
    """

    def __init__(self, sulcal_dim: int = 256, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(sulcal_dim, d_model)

    def forward(
        self,
        z_sulcal: torch.Tensor,
        token_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend sulcal prefix token to EEG token sequence.

        Args:
            z_sulcal:   (B, sulcal_dim)
            token_repr: (B, n_tokens, d_model)

        Returns:
            fused: (B, n_tokens + 1, d_model) — sulcal token first
        """
        prefix = self.proj(z_sulcal).unsqueeze(1)  # (B, 1, d_model)
        return torch.cat([prefix, token_repr], dim=1)
