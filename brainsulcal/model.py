"""BrainSulcal: Champollion-conditioned BrainOmni for EEG classification.

Architecture (after studying BrainOmni source — no MoE router exists):

BrainOmni uses SpatialTemporalAttentionBlock:
  - Spatial attention over channels (B*W, C, D//2)
  - Temporal attention over windows (B*C, W, D//2)
  - encode() returns (B, C, W, lm_dim) L2-normalized

Three sulcal injection points:
  1. Channel embedding bias: z_sulcal → (B,1,1,n_dim) added to neuro embeddings
     (surgical injection at BrainOmni's internal neuro positional embeddings)
  2. Prefix virtual channel: z_sulcal projected as extra channel appended to
     window-pooled features before the classification head
  3. Downstream fusion: z_sulcal concatenated with mean-pooled z_eeg

Trainable components only:
  - SulcalAggregator
  - MoERouterBias (outputs channel embedding bias)
  - Prefix projection (z_sulcal → lm_dim virtual channel)
  - Fusion MLP + classification head

BrainOmni and Champollion are always frozen.

Input format (matching BrainOmni API):
  eeg:          (B, C, W*T)  — windowed EEG (C channels, W windows × T samples)
  pos:          (B, C, 6)    — electrode xyz + orientation (6D)
  sensor_type:  (B, C)       — sensor type integer codes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from brainsulcal.dynamics.brainomni_wrapper import BrainOmniWrapper
from brainsulcal.dynamics.moe_router_bias import MoERouterBias
from brainsulcal.priors.sulcal_aggregator import SulcalAggregator


@dataclass
class BrainSulcalConfig:
    # SulcalAggregator
    champollion_input_dim: int = 128       # Set from ChampollionWrapper.embedding_dim
    sulcal_hidden_dim: int = 256
    sulcal_n_heads: int = 4
    sulcal_n_layers: int = 2
    sulcal_dropout: float = 0.1

    # MoERouterBias (channel embedding bias)
    n_dim: int = 256                       # BrainOmni tokenizer n_dim (before projection)
    router_bias_hidden_dim: int = 128
    router_bias_init_std: float = 1e-3

    # BrainOmni
    brainomni_checkpoint: str = "OpenTSLab/BrainOmni"
    lm_dim: int = 512                      # BrainOmni backbone hidden dim (after projection)

    # Fusion
    use_router_bias: bool = True           # Channel embedding bias injection
    use_prefix_token: bool = True          # Prepend z_sulcal as virtual channel
    fusion_hidden_dim: int = 256

    # Task head
    n_classes: int = 2
    n_tasks: int = 2                       # valence + arousal

    # Ablation flags
    use_sulcal_prior: bool = True


class BrainSulcal(nn.Module):
    """Champollion-conditioned BrainOmni for EEG music classification.

    Args:
        config: BrainSulcalConfig with all hyperparameters.
    """

    def __init__(self, config: BrainSulcalConfig):
        super().__init__()
        self.config = config

        # --- Frozen backbone ---
        self.brainomni = BrainOmniWrapper(
            checkpoint=config.brainomni_checkpoint,
            freeze=True,
        )

        # --- Trainable components ---
        if config.use_sulcal_prior:
            self.sulcal_aggregator = SulcalAggregator(
                input_dim=config.champollion_input_dim,
                hidden_dim=config.sulcal_hidden_dim,
                n_heads=config.sulcal_n_heads,
                n_layers=config.sulcal_n_layers,
                dropout=config.sulcal_dropout,
            )

            if config.use_router_bias:
                self.channel_bias = MoERouterBias(
                    sulcal_dim=config.sulcal_hidden_dim,
                    n_dim=config.n_dim,
                    hidden_dim=config.router_bias_hidden_dim,
                    init_std=config.router_bias_init_std,
                )
            else:
                self.channel_bias = None

            if config.use_prefix_token:
                # Project z_sulcal as a virtual channel in BrainOmni's feature space
                self.prefix_proj = nn.Linear(config.sulcal_hidden_dim, config.lm_dim)
            else:
                self.prefix_proj = None

            fusion_input_dim = config.sulcal_hidden_dim + config.lm_dim
        else:
            self.sulcal_aggregator = None
            self.channel_bias = None
            self.prefix_proj = None
            fusion_input_dim = config.lm_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_hidden_dim),
        )

        # One classification head per task (valence, arousal)
        self.task_heads = nn.ModuleList([
            nn.Linear(config.fusion_hidden_dim, config.n_classes)
            for _ in range(config.n_tasks)
        ])

        self._verify_invariants()

    def _verify_invariants(self) -> None:
        assert not any(p.requires_grad for p in self.brainomni.parameters()), \
            "BrainOmni parameters must be frozen."

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        sulcal_embeddings: torch.Tensor,
        sulcal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            eeg:               (B, C, W*T) windowed EEG (W windows × T samples)
            pos:               (B, C, 6) electrode positions
            sensor_type:       (B, C) sensor type codes
            sulcal_embeddings: (B, 56, d_champollion) pre-computed embeddings
            sulcal_mask:       (B, 56) bool — True where valid

        Returns dict:
            logits_valence: (B, n_classes)
            logits_arousal: (B, n_classes)
            z_sulcal:       (B, sulcal_hidden_dim) or None
            z_eeg:          (B, lm_dim)
            z_fused:        (B, fusion_hidden_dim)
        """
        B = eeg.shape[0]

        # 1. Sulcal prior
        z_sulcal: torch.Tensor | None = None
        channel_bias: torch.Tensor | None = None

        if self.config.use_sulcal_prior and self.sulcal_aggregator is not None:
            z_sulcal, _ = self.sulcal_aggregator(sulcal_embeddings, sulcal_mask)
            # z_sulcal: (B, sulcal_hidden_dim)

            if self.channel_bias is not None:
                channel_bias = self.channel_bias(z_sulcal)
                # channel_bias: (B, 1, 1, n_dim) — injected into BrainOmni neuro embeddings

        # 2. BrainOmni encode (frozen) with optional channel embedding bias
        feat = self.brainomni.encode(
            eeg, pos, sensor_type, channel_bias=channel_bias
        )
        # feat: (B, C, W, lm_dim)

        # 3. Pool over windows → (B, C, lm_dim)
        feat = feat.mean(dim=2)

        # 4. Optional: prepend z_sulcal as virtual channel before mean-pooling channels
        if self.prefix_proj is not None and z_sulcal is not None:
            sulcal_channel = self.prefix_proj(z_sulcal).unsqueeze(1)  # (B, 1, lm_dim)
            feat = torch.cat([sulcal_channel, feat], dim=1)           # (B, C+1, lm_dim)

        # 5. Mean-pool over channels → z_eeg
        z_eeg = feat.mean(dim=1)  # (B, lm_dim)

        # 6. Fuse sulcal + EEG
        if z_sulcal is not None:
            z_fused_input = torch.cat([z_sulcal, z_eeg], dim=-1)  # (B, sulcal_dim+lm_dim)
        else:
            z_fused_input = z_eeg

        z_fused = self.fusion_mlp(z_fused_input)  # (B, fusion_hidden_dim)

        # 7. Task heads
        logits = [head(z_fused) for head in self.task_heads]

        # Shape assertions
        assert z_eeg.shape == (B, self.config.lm_dim)
        assert z_fused.shape == (B, self.config.fusion_hidden_dim)
        if z_sulcal is not None:
            assert z_sulcal.shape == (B, self.config.sulcal_hidden_dim)

        return {
            "logits_valence": logits[0],
            "logits_arousal": logits[1] if len(logits) > 1 else logits[0],
            "z_sulcal": z_sulcal,
            "z_eeg": z_eeg,
            "z_fused": z_fused,
        }

    def trainable_parameters(self) -> list[dict]:
        """Parameter groups with differential learning rates."""
        groups = []
        if self.sulcal_aggregator is not None:
            groups.append({
                "params": list(self.sulcal_aggregator.parameters()),
                "name": "sulcal_aggregator",
            })
        if self.channel_bias is not None:
            groups.append({
                "params": list(self.channel_bias.parameters()),
                "name": "moe_router_bias",  # keep name for config compatibility
            })
        if self.prefix_proj is not None:
            groups.append({
                "params": list(self.prefix_proj.parameters()),
                "name": "prefix_fusion",
            })
        groups.append({
            "params": (
                list(self.fusion_mlp.parameters())
                + list(self.task_heads.parameters())
            ),
            "name": "classification_head",
        })
        return groups
