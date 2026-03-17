"""MoE Router Bias: maps z_sulcal to a bias over BrainOmni's expert vocabulary.

The bias is added to MoE router logits BEFORE softmax, nudging expert selection
toward anatomy-relevant experts without altering BrainOmni's weights.

Critical: final layer initialized with near-zero weights (std=1e-3) so training
starts with essentially zero router bias, preserving BrainOmni's pre-trained routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MoERouterBias(nn.Module):
    """Two-layer MLP mapping z_sulcal → per-expert router bias.

    Args:
        sulcal_dim:  Dimensionality of z_sulcal (default 256).
        n_experts:   Must match BrainOmni's MoE n_experts exactly.
                     Read from BrainOmni source before instantiating.
        hidden_dim:  Hidden layer size (default 128).
        init_std:    Std for final layer init (default 1e-3).
                     Keep small to preserve BrainOmni routing at init.
    """

    def __init__(
        self,
        sulcal_dim: int = 256,
        n_experts: int | None = None,
        hidden_dim: int = 128,
        init_std: float = 1e-3,
    ):
        super().__init__()
        if n_experts is None:
            raise ValueError(
                "n_experts must be set to match BrainOmni's MoE n_experts. "
                "Read BrainOmni source to find this value."
            )
        self.n_experts = n_experts

        self.mlp = nn.Sequential(
            nn.Linear(sulcal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

        # Near-zero init on the final layer — critical for training stability
        nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_sulcal: torch.Tensor) -> torch.Tensor:
        """Compute router bias from subject fingerprint.

        Args:
            z_sulcal: (B, sulcal_dim)

        Returns:
            bias: (B, n_experts) — added to MoE router logits before softmax
        """
        return self.mlp(z_sulcal)

    def zero_(self) -> None:
        """Zero all weights and biases (for testing zero-bias identity)."""
        for p in self.parameters():
            p.data.zero_()
