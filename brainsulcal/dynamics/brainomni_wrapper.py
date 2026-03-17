"""Wraps BrainOmni with optional sulcal channel embedding bias injection.

After studying BrainOmni source (external/BrainOmni/brainomni/model.py):
  - Architecture: SpatialTemporalAttentionBlock (NOT MoE — no sparse router)
  - Spatial attention operates over channels (B*W, C, D//2)
  - Temporal attention operates over windows (B*C, W, D//2)
  - encode() signature: (x, pos, sensor_type) → (B, C, W, D) L2-normalized

Injection point: the `neuro` per-channel embeddings.
BrainOmni adds `self.tokenizer.encoder.neuros` (C, n_dim) to each token.
We inject `channel_bias` (B, 1, 1, n_dim) as an additional subject-specific
offset, added at the same point. This is minimal and surgical:
  x = x + neuro + channel_bias  (zero channel_bias → identical to vanilla)

Input format:
  x:           (B, C, W*T)  — EEG windows, C channels, W windows × T samples
  pos:         (B, C, 6)    — electrode positions (xyz + orientation)
  sensor_type: (B, C)       — sensor type integer codes

All BrainOmni parameters are FROZEN.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_BRAINOMNI_CHECKPOINT = "OpenTSLab/BrainOmni"


class BrainOmniWrapper(nn.Module):
    """Frozen BrainOmni with optional sulcal channel bias injection.

    Args:
        checkpoint: HuggingFace model ID or local path to BrainOmni checkpoint.
        freeze:     If True (default), freeze all BrainOmni parameters.
    """

    def __init__(
        self,
        checkpoint: str = _BRAINOMNI_CHECKPOINT,
        freeze: bool = True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = self._load_brainomni(checkpoint)

        if freeze:
            self._freeze()

    def _load_brainomni(self, checkpoint: str) -> nn.Module:
        try:
            from brainomni.model import BrainOmni  # type: ignore[import]
            import json
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(checkpoint, "model_cfg.json")
            ckpt_path = hf_hub_download(checkpoint, "BrainOmni.pt")
            with open(cfg_path) as f:
                cfg = json.load(f)
            model = BrainOmni(**cfg)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded BrainOmni from %s", checkpoint)
            return model
        except ImportError:
            logger.warning(
                "BrainOmni not installed. Run `pixi run install-brainomni`. "
                "Using a stub for shape testing only."
            )
            return _BrainOmniStub()
        except Exception as e:
            logger.warning("BrainOmni load failed (%s) — using stub.", e)
            return _BrainOmniStub()

    def _freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in self.model.parameters())
        logger.info("BrainOmni frozen (%d params).", n)

    @property
    def n_dim(self) -> int:
        """Tokenizer embedding dimension (= D in encode output)."""
        try:
            return self.model.tokenizer.n_dim
        except AttributeError:
            return 256  # BrainOmni base default

    @property
    def lm_dim(self) -> int:
        """Backbone hidden dimension (= D after projection)."""
        try:
            return self.model.lm_dim
        except AttributeError:
            return 512  # BrainOmni base default

    def encode(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        channel_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode EEG with optional sulcal channel bias injection.

        Replicates BrainOmni.encode() but injects channel_bias at the
        neuro embedding step:
            token += neuro_emb + channel_bias

        Args:
            x:            (B, C, W*T) — windowed EEG
            pos:          (B, C, 6)   — electrode positions
            sensor_type:  (B, C)      — sensor type codes
            channel_bias: (B, 1, 1, lm_dim) or None
                          Subject-specific bias over channel embeddings.
                          Zero → identical to vanilla BrainOmni.

        Returns:
            features: (B, C, W, lm_dim) L2-normalized, same as BrainOmni.encode()
        """
        model = self.model

        # Handle stub
        if isinstance(model, _BrainOmniStub):
            return model.encode(x, pos, sensor_type, channel_bias)

        # --- Replicate BrainOmni.encode() with bias injection ---
        x_tok, _ = model.tokenizer.tokenize(
            x, pos, sensor_type, model.overlap_ratio
        )
        B, C, W, D = x_tok.shape

        # Add neuro (per-channel positional) embeddings
        neuro = model.tokenizer.encoder.neuros.type_as(x_tok).detach().view(1, C, 1, -1)
        x_tok = x_tok + neuro

        # Inject subject-specific channel bias (surgical, minimal modification)
        if channel_bias is not None:
            # channel_bias: (B, 1, 1, lm_dim) → broadcasts over C and W after projection
            # Bias is injected before projection to match the neuro embedding scale
            x_tok = x_tok + channel_bias

        x_tok = model.projection(x_tok)  # (B, C, W, lm_dim)

        # All blocks except last (matching BrainOmni.encode() exactly)
        for block in model.blocks[:-1]:
            x_tok = block(x_tok)

        return F.normalize(x_tok, p=2.0, dim=-1, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        channel_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning pooled (B, C*lm_dim) representation.

        Matches BrainOmni downstream usage:
            x = encode(...)          # (B, C, W, D)
            x = x.mean(2)           # (B, C, D) — pool over windows
            x = x.view(B, -1)       # (B, C*D) — flatten channels

        Args:
            x, pos, sensor_type: same as encode()
            channel_bias: (B, 1, 1, n_dim) or None

        Returns:
            (B, C*lm_dim) flattened representation
        """
        feat = self.encode(x, pos, sensor_type, channel_bias)  # (B, C, W, D)
        feat = feat.mean(dim=2)                                  # (B, C, D)
        return feat.contiguous().view(feat.shape[0], -1)         # (B, C*D)


class _BrainOmniStub(nn.Module):
    """Minimal stub for shape testing when BrainOmni is not installed.

    Mimics BrainOmni base config:
      n_dim=256, lm_dim=512, lm_depth=12, window compression ~4×
    """

    def __init__(self, n_dim: int = 256, lm_dim: int = 512, n_windows: int = 8):
        super().__init__()
        self.n_dim = n_dim
        self.lm_dim = lm_dim
        self.n_windows = n_windows

    def encode(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        channel_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, _ = x.shape
        out = torch.zeros(B, C, self.n_windows, self.lm_dim, device=x.device)
        return out

    def forward(self, x, pos, sensor_type, channel_bias=None):
        feat = self.encode(x, pos, sensor_type, channel_bias)
        feat = feat.mean(dim=2)
        return feat.contiguous().view(feat.shape[0], -1)
