"""Wraps BrainOmni with optional MoE router bias injection.

BrainOmni has two stages:
  Stage 1 — BrainTokenizer: quantises raw EEG/MEG into discrete tokens
  Stage 2 — BrainOmni backbone: transformer with Sparse MoE + DINT attention

All BrainOmni parameters are FROZEN. The only modification is adding a router
bias to MoE router logits before softmax (surgical, minimal change).

NOTE: Before using this wrapper, study BrainOmni's source code to find:
  1. The exact class/method where MoE router logits are computed
  2. The value of n_experts
  3. The expected montage_info format

Placeholder hooks are registered via forward hooks on the router modules.
Update _find_router_modules() after studying BrainOmni internals.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_BRAINOMNI_CHECKPOINT = "OpenTSLab/BrainOmni"


class BrainOmniWrapper(nn.Module):
    """Frozen BrainOmni with optional sulcal router bias injection.

    Args:
        checkpoint:  HuggingFace model ID or local path.
        freeze:      If True (default), freeze all BrainOmni parameters.
    """

    def __init__(
        self,
        checkpoint: str = _BRAINOMNI_CHECKPOINT,
        freeze: bool = True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self._router_bias: torch.Tensor | None = None
        self._hook_handles: list[Any] = []

        self.model = self._load_brainomni(checkpoint)

        if freeze:
            self._freeze()

        # Register forward hooks for router bias injection
        self._register_router_hooks()

    def _load_brainomni(self, checkpoint: str) -> nn.Module:
        """Load BrainOmni from HuggingFace or local path.

        NOTE: Implement once BrainOmni package is available via
              `pixi run install-brainomni`.
        """
        try:
            # Import deferred — BrainOmni is an external submodule
            from brain_omni import BrainOmni  # type: ignore[import]
            model = BrainOmni.from_pretrained(checkpoint)
            logger.info("Loaded BrainOmni from %s", checkpoint)
            return model
        except ImportError:
            logger.warning(
                "BrainOmni not installed. Run `pixi run install-brainomni`. "
                "Using a stub for shape testing only."
            )
            return _BrainOmniStub()

    def _freeze(self) -> None:
        """Freeze all BrainOmni parameters."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info("BrainOmni parameters frozen (%d params).", sum(1 for _ in self.model.parameters()))

    def _find_router_modules(self) -> list[nn.Module]:
        """Return the MoE router modules inside BrainOmni.

        TODO: After studying BrainOmni source, replace this with the actual
        module path. Example:
            return [layer.moe.router for layer in self.model.backbone.layers
                    if hasattr(layer, 'moe')]
        """
        routers = []
        for name, module in self.model.named_modules():
            # Heuristic: look for modules named 'router' or containing 'moe'
            if "router" in name.lower() or type(module).__name__.lower() in ("moerouter", "sparserouter"):
                routers.append(module)
        if not routers:
            logger.warning(
                "No MoE router modules found in BrainOmni. "
                "Router bias injection will have no effect. "
                "Update _find_router_modules() after studying BrainOmni source."
            )
        return routers

    def _register_router_hooks(self) -> None:
        """Register forward hooks to inject router bias into MoE logits."""
        # Remove any previously registered hooks
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        routers = self._find_router_modules()
        for router in routers:
            handle = router.register_forward_hook(self._router_bias_hook)
            self._hook_handles.append(handle)

        logger.debug("Registered router bias hooks on %d modules.", len(routers))

    def _router_bias_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Add router bias to logits before softmax.

        This hook fires AFTER the router linear layer but BEFORE softmax.
        The output tensor is the router logits of shape (B, n_experts) or
        (B*T, n_experts) depending on BrainOmni's implementation.

        TODO: Verify output shape after studying BrainOmni source.
        """
        if self._router_bias is None:
            return output

        bias = self._router_bias
        if output.shape[-1] != bias.shape[-1]:
            raise ValueError(
                f"Router bias n_experts ({bias.shape[-1]}) does not match "
                f"BrainOmni router output ({output.shape[-1]}). "
                f"Update MoERouterBias(n_experts=...) to match BrainOmni."
            )

        # Broadcast bias over token dimension if needed
        if output.dim() == 2 and bias.dim() == 2:
            # output: (B*T, n_experts), bias: (B, n_experts)
            # Need to know B and T — stored when forward() is called
            if hasattr(self, "_batch_size") and output.shape[0] != bias.shape[0]:
                B = self._batch_size
                T = output.shape[0] // B
                bias_expanded = bias.repeat_interleave(T, dim=0)  # (B*T, n_experts)
                return output + bias_expanded

        return output + bias

    def forward(
        self,
        eeg: torch.Tensor,
        montage_info: dict,
        router_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run BrainOmni forward pass with optional router bias injection.

        Args:
            eeg:          (B, C, T) raw EEG signal
            montage_info: dict with electrode positions, types, orientations
            router_bias:  (B, n_experts) or None — sulcal router bias

        Returns:
            token_repr: (B, n_tokens, d_model) BrainOmni token representations
        """
        self._router_bias = router_bias
        self._batch_size = eeg.shape[0]

        try:
            output = self.model(eeg, montage_info=montage_info)
        finally:
            # Always clear the bias after forward to avoid stale state
            self._router_bias = None

        return output

    @property
    def d_model(self) -> int:
        """BrainOmni hidden dimension (must match prefix_proj_dim in config)."""
        try:
            return self.model.config.d_model
        except AttributeError:
            return 512  # Default from CLAUDE.md; update after studying BrainOmni source


class _BrainOmniStub(nn.Module):
    """Minimal stub for shape testing when BrainOmni is not installed."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"d_model": 512})()

    def forward(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, T = eeg.shape
        n_tokens = T // 4  # Approximate tokenization ratio
        d_model = self.config.d_model
        return torch.zeros(B, n_tokens, d_model, device=eeg.device)
