"""Sulcal channel bias: maps z_sulcal → per-channel token embedding bias.

Original design assumed BrainOmni had a Sparse MoE router (per CLAUDE.md).
After studying source (external/BrainOmni/brainomni/model.py), BrainOmni
uses SpatialTemporalAttentionBlock — no MoE router exists.

The closest structural analog is the `neuro` per-channel positional embedding:
    x = token_embeddings + neuro_emb          (BrainOmni vanilla)
    x = token_embeddings + neuro_emb + bias   (BrainSulcal injection)

This class maps z_sulcal → channel_bias (B, 1, 1, n_dim), which is then
broadcast over all C channels and W windows. A subject with deeper Heschl's
gyrus gets a different embedding shift, biasing all channel representations.

Zero bias → numerically identical to vanilla BrainOmni (identity invariant).

Critical init: final layer weights near-zero (std=1e-3) so training starts
with near-zero bias, preserving BrainOmni's pre-trained representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MoERouterBias(nn.Module):
    """Two-layer MLP: z_sulcal → channel embedding bias.

    Output is broadcast over all channels and windows in BrainOmniWrapper.encode().

    Args:
        sulcal_dim:  Dimensionality of z_sulcal (default 256).
        n_dim:       BrainOmni token dimension (n_dim before projection).
                     Must match model.tokenizer.n_dim exactly.
        hidden_dim:  Hidden layer size (default 128).
        init_std:    Std for final layer init (default 1e-3).
    """

    def __init__(
        self,
        sulcal_dim: int = 256,
        n_dim: int | None = None,
        hidden_dim: int = 128,
        init_std: float = 1e-3,
    ):
        super().__init__()
        if n_dim is None:
            raise ValueError(
                "n_dim must match BrainOmni's tokenizer n_dim. "
                "Read BrainOmni config (external/BrainOmni/share/) to find this value."
            )
        self.n_dim = n_dim

        self.mlp = nn.Sequential(
            nn.Linear(sulcal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_dim),
        )

        # Near-zero init on final layer — critical to preserve BrainOmni at training start
        nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_sulcal: torch.Tensor) -> torch.Tensor:
        """Compute channel embedding bias from subject fingerprint.

        Args:
            z_sulcal: (B, sulcal_dim)

        Returns:
            channel_bias: (B, 1, 1, n_dim)
                Broadcast over C channels and W windows in BrainOmniWrapper.encode()
        """
        bias = self.mlp(z_sulcal)          # (B, n_dim)
        return bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, n_dim)

    def zero_(self) -> None:
        """Zero all weights and biases (for zero-bias identity test)."""
        for p in self.parameters():
            p.data.zero_()
