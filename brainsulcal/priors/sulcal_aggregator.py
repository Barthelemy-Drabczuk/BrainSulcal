"""Sulcal aggregator: small transformer over 56 Champollion regional embeddings.

Produces:
  z_sulcal  ∈ R^hidden_dim  — global subject fingerprint (CLS token output)
  R_sulcal  ∈ R^{56 × hidden_dim} — regional sequence (for potential cross-attention)

Architecture:
  - Input projection: embedding_dim → hidden_dim
  - Learnable [CLS] token
  - Learnable positional embeddings indexed by region ID (not sequence position)
  - TransformerEncoder with pre-norm (norm_first=True)
  - Output: CLS → z_sulcal, remaining tokens → R_sulcal
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_REGIONS = 56


class SulcalAggregator(nn.Module):
    """Transformer aggregator over 56 Champollion sulcal region embeddings.

    Args:
        input_dim:  Dimensionality of Champollion embeddings (read from wrapper).
        hidden_dim: Internal and output dimension (default 256).
        n_heads:    Number of attention heads (default 4).
        n_layers:   Number of transformer encoder layers (default 2).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project input embeddings to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings per region (index 0..N_REGIONS-1)
        # Plus one extra position for the CLS token (position index N_REGIONS)
        self.pos_emb = nn.Embedding(N_REGIONS + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            norm_first=True,  # Pre-norm for training stability
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate 56 sulcal embeddings into z_sulcal and R_sulcal.

        Args:
            embeddings: (B, 56, input_dim) — raw Champollion embeddings
            mask:       (B, 56) bool — True where embedding is valid;
                        masked positions are ignored in attention.

        Returns:
            z_sulcal:  (B, hidden_dim) — global subject fingerprint
            R_sulcal:  (B, 56, hidden_dim) — regional representations
        """
        B = embeddings.size(0)
        device = embeddings.device

        # Project to hidden_dim
        x = self.input_proj(embeddings)  # (B, 56, hidden_dim)

        # Add region positional embeddings (positions 0..55)
        region_pos = torch.arange(N_REGIONS, device=device).unsqueeze(0)  # (1, 56)
        x = x + self.pos_emb(region_pos)

        # Prepend [CLS] token with its own positional embedding (position 56)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        cls_pos = torch.full((1, 1), N_REGIONS, dtype=torch.long, device=device)
        cls = cls + self.pos_emb(cls_pos)

        x = torch.cat([cls, x], dim=1)  # (B, 57, hidden_dim)

        # Build key_padding_mask for transformer:
        # True = position should be IGNORED (opposite of our 'valid' mask convention)
        if mask is not None:
            # CLS token is always attended to (not masked)
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            padding_mask = ~torch.cat([cls_valid, mask], dim=1)  # (B, 57)
        else:
            padding_mask = None

        # Transformer encoding
        out = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, 57, hidden_dim)
        out = self.norm(out)

        z_sulcal = out[:, 0, :]      # (B, hidden_dim) — CLS output
        R_sulcal = out[:, 1:, :]     # (B, 56, hidden_dim) — regional outputs

        return z_sulcal, R_sulcal
