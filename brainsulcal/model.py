"""BrainSulcal: top-level model combining Champollion and BrainOmni.

Three injection points for the sulcal prior:
  1. MoE router bias: z_sulcal → MoERouterBias → added to BrainOmni router logits
  2. Prefix token:    z_sulcal projected and prepended to BrainOmni token sequence
  3. Downstream head: z_sulcal concatenated with pooled BrainOmni output

Trainable components only:
  - SulcalAggregator
  - MoERouterBias
  - PrefixFusion projection
  - Fusion MLP
  - Classification head

BrainOmni and Champollion are always frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from brainsulcal.dynamics.brainomni_wrapper import BrainOmniWrapper
from brainsulcal.dynamics.moe_router_bias import MoERouterBias
from brainsulcal.fusion.prefix_fusion import PrefixFusion
from brainsulcal.priors.sulcal_aggregator import SulcalAggregator


@dataclass
class BrainSulcalConfig:
    # SulcalAggregator
    champollion_input_dim: int = 128       # Set from wrapper.embedding_dim at runtime
    sulcal_hidden_dim: int = 256
    sulcal_n_heads: int = 4
    sulcal_n_layers: int = 2
    sulcal_dropout: float = 0.1

    # MoERouterBias
    n_experts: int | None = None           # Must match BrainOmni; set after loading model
    router_bias_hidden_dim: int = 128
    router_bias_init_std: float = 1e-3

    # BrainOmni
    brainomni_checkpoint: str = "OpenTSLab/BrainOmni"
    d_model: int = 512                     # BrainOmni hidden dim; verify against source

    # Fusion
    use_router_bias: bool = True
    use_prefix_token: bool = True
    fusion_hidden_dim: int = 256

    # Task head
    n_classes: int = 2
    n_tasks: int = 2                       # valence + arousal

    # Ablation flags (set via configs/ablation.yaml)
    use_sulcal_prior: bool = True


class BrainSulcal(nn.Module):
    """Champollion-conditioned BrainOmni for EEG music classification.

    Args:
        config: BrainSulcalConfig with all hyperparameters.
    """

    def __init__(self, config: BrainSulcalConfig):
        super().__init__()
        self.config = config

        # --- Frozen components ---
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
                if config.n_experts is None:
                    raise ValueError(
                        "config.n_experts must be set to match BrainOmni's MoE n_experts. "
                        "Read BrainOmni source before instantiating BrainSulcal."
                    )
                self.moe_router_bias = MoERouterBias(
                    sulcal_dim=config.sulcal_hidden_dim,
                    n_experts=config.n_experts,
                    hidden_dim=config.router_bias_hidden_dim,
                    init_std=config.router_bias_init_std,
                )
            else:
                self.moe_router_bias = None

            if config.use_prefix_token:
                self.prefix_fusion = PrefixFusion(
                    sulcal_dim=config.sulcal_hidden_dim,
                    d_model=config.d_model,
                )
            else:
                self.prefix_fusion = None

            # Fusion: [z_sulcal ; z_eeg_pooled] → d_fused
            fusion_input_dim = config.sulcal_hidden_dim + config.d_model
        else:
            self.sulcal_aggregator = None
            self.moe_router_bias = None
            self.prefix_fusion = None
            fusion_input_dim = config.d_model

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
        """Assert frozen/trainable split is correct."""
        assert not any(p.requires_grad for p in self.brainomni.parameters()), \
            "BrainOmni parameters must be frozen."

    def forward(
        self,
        eeg: torch.Tensor,
        montage_info: dict,
        sulcal_embeddings: torch.Tensor,
        sulcal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            eeg:               (B, C, T) raw EEG signal
            montage_info:      dict with electrode positions / types
            sulcal_embeddings: (B, 56, d_champollion) pre-computed embeddings
            sulcal_mask:       (B, 56) bool — True where embedding is valid

        Returns dict with keys:
            logits_valence: (B, n_classes)
            logits_arousal: (B, n_classes)
            z_sulcal:       (B, sulcal_hidden_dim) or None
            z_eeg:          (B, d_model)
            z_fused:        (B, fusion_hidden_dim)
        """
        # 1. Sulcal prior
        z_sulcal: torch.Tensor | None = None
        router_bias: torch.Tensor | None = None

        if self.config.use_sulcal_prior and self.sulcal_aggregator is not None:
            z_sulcal, _ = self.sulcal_aggregator(sulcal_embeddings, sulcal_mask)

            if self.moe_router_bias is not None:
                router_bias = self.moe_router_bias(z_sulcal)

        # 2. BrainOmni forward (frozen) with optional router bias
        token_repr = self.brainomni(eeg, montage_info, router_bias=router_bias)
        # token_repr: (B, n_tokens, d_model)

        # 3. Optional prefix token insertion (informational — used by downstream pooling)
        if self.prefix_fusion is not None and z_sulcal is not None:
            token_repr = self.prefix_fusion(z_sulcal, token_repr)
            # token_repr: (B, n_tokens+1, d_model) — prefix is first token

        # 4. Pool token representations → z_eeg
        z_eeg = token_repr.mean(dim=1)  # (B, d_model)

        # 5. Fuse sulcal + EEG representations
        if z_sulcal is not None:
            z_fused_input = torch.cat([z_sulcal, z_eeg], dim=-1)
        else:
            z_fused_input = z_eeg

        z_fused = self.fusion_mlp(z_fused_input)  # (B, fusion_hidden_dim)

        # 6. Task-specific heads
        logits = [head(z_fused) for head in self.task_heads]

        # Validate output shapes
        B = eeg.shape[0]
        assert z_eeg.shape == (B, self.config.d_model)
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
        """Return parameter groups with differential learning rates.

        Use with configs/default.yaml lr values.
        """
        groups = []

        if self.sulcal_aggregator is not None:
            groups.append({
                "params": list(self.sulcal_aggregator.parameters()),
                "name": "sulcal_aggregator",
            })
        if self.moe_router_bias is not None:
            groups.append({
                "params": list(self.moe_router_bias.parameters()),
                "name": "moe_router_bias",
            })
        if self.prefix_fusion is not None:
            groups.append({
                "params": list(self.prefix_fusion.parameters()),
                "name": "prefix_fusion",
            })

        groups.append({
            "params": list(self.fusion_mlp.parameters()) + list(self.task_heads.parameters()),
            "name": "classification_head",
        })

        return groups
