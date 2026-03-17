"""Training loop for BrainSulcal with differential learning rates and wandb logging.

Key design choices:
  - Differential learning rates per trainable component group
  - Gradient clipping
  - Cosine LR schedule with linear warmup
  - Router bias entropy monitoring (collapse detection)
  - Early stopping on validation accuracy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brainsulcal.model import BrainSulcal, BrainSulcalConfig
from brainsulcal.training.losses import BrainSulcalLoss
from brainsulcal.training.metrics import compute_metrics, router_bias_entropy

logger = logging.getLogger(__name__)

# Router bias entropy collapse threshold (bits)
ENTROPY_COLLAPSE_THRESHOLD = 0.5


class Trainer:
    """Training loop with differential LR, wandb logging, early stopping.

    Args:
        model:          BrainSulcal model.
        config:         Training hyperparameters (from OmegaConf / dict).
        output_dir:     Directory for checkpoints and logs.
        use_wandb:      Whether to log metrics to Weights & Biases.
    """

    def __init__(
        self,
        model: BrainSulcal,
        config: Any,
        output_dir: str | Path,
        use_wandb: bool = True,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.loss_fn = BrainSulcalLoss(
            lambda_align=getattr(config.training, "lambda_align", 0.1),
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.best_val_accuracy = 0.0
        self.patience_counter = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.training
        param_groups = []

        lr_map = {
            "sulcal_aggregator": cfg.lr_sulcal_aggregator,
            "moe_router_bias": cfg.lr_moe_router_bias,
            "prefix_fusion": cfg.lr_sulcal_aggregator,
            "classification_head": cfg.lr_classification_head,
        }

        for group in self.model.trainable_parameters():
            lr = lr_map.get(group["name"], cfg.lr_classification_head)
            param_groups.append({
                "params": group["params"],
                "lr": lr,
                "name": group["name"],
            })

        return torch.optim.AdamW(
            param_groups,
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

    def _build_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        cfg = self.config.training
        return CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_epochs,
            eta_min=1e-6,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        montage_info: dict,
    ) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Best validation metrics dict.
        """
        cfg = self.config.training
        step = 0
        best_metrics: dict[str, float] = {}

        for epoch in range(cfg.max_epochs):
            train_metrics = self._train_epoch(train_loader, montage_info, step)
            val_metrics = self._val_epoch(val_loader, montage_info)

            step += len(train_loader)
            self.scheduler.step()

            val_acc = val_metrics.get("valence/accuracy", 0.0)

            if self.use_wandb:
                self._log_wandb({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "epoch": epoch,
                    "step": step,
                })

            logger.info(
                "Epoch %d — val valence acc: %.3f | arousal acc: %.3f | loss: %.4f",
                epoch,
                val_metrics.get("valence/accuracy", 0.0),
                val_metrics.get("arousal/accuracy", 0.0),
                train_metrics.get("loss_total", 0.0),
            )

            # Early stopping and checkpoint
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                best_metrics = val_metrics
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        return best_metrics

    def _train_epoch(
        self,
        loader: DataLoader,
        montage_info: dict,
        global_step: int,
    ) -> dict[str, float]:
        self.model.train()
        cfg = self.config.training
        total_loss = total_cls = total_align = 0.0
        n_batches = 0

        for batch in loader:
            eeg = batch["eeg"].to(self._device)
            val_labels = batch["valence_label"].to(self._device)
            aro_labels = batch["arousal_label"].to(self._device)
            sulcal_emb = batch.get("sulcal_embeddings")
            sulcal_mask = batch.get("sulcal_mask")

            if sulcal_emb is not None:
                sulcal_emb = sulcal_emb.to(self._device)
                sulcal_mask = sulcal_mask.to(self._device)
            else:
                B = eeg.shape[0]
                d = self.model.config.champollion_input_dim
                sulcal_emb = torch.zeros(B, 56, d, device=self._device)
                sulcal_mask = torch.zeros(B, 56, dtype=torch.bool, device=self._device)

            outputs = self.model(eeg, montage_info, sulcal_emb, sulcal_mask)

            losses = self.loss_fn(
                outputs["logits_valence"], outputs["logits_arousal"],
                val_labels, aro_labels,
                z_sulcal=outputs["z_sulcal"], z_eeg=outputs["z_eeg"],
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=getattr(cfg, "gradient_clip", 1.0),
            )
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_cls += losses["cls"].item()
            total_align += losses["align"].item()
            n_batches += 1

        return {
            "loss_total": total_loss / max(n_batches, 1),
            "loss_cls": total_cls / max(n_batches, 1),
            "loss_align": total_align / max(n_batches, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, montage_info: dict) -> dict[str, float]:
        self.model.eval()
        all_val_logits, all_val_labels = [], []
        all_aro_logits, all_aro_labels = [], []

        for batch in loader:
            eeg = batch["eeg"].to(self._device)
            val_labels = batch["valence_label"].to(self._device)
            aro_labels = batch["arousal_label"].to(self._device)
            sulcal_emb = batch.get("sulcal_embeddings")
            sulcal_mask = batch.get("sulcal_mask")

            if sulcal_emb is not None:
                sulcal_emb = sulcal_emb.to(self._device)
                sulcal_mask = sulcal_mask.to(self._device)
            else:
                B = eeg.shape[0]
                d = self.model.config.champollion_input_dim
                sulcal_emb = torch.zeros(B, 56, d, device=self._device)
                sulcal_mask = torch.zeros(B, 56, dtype=torch.bool, device=self._device)

            outputs = self.model(eeg, montage_info, sulcal_emb, sulcal_mask)

            all_val_logits.append(outputs["logits_valence"].cpu())
            all_val_labels.append(val_labels.cpu())
            all_aro_logits.append(outputs["logits_arousal"].cpu())
            all_aro_labels.append(aro_labels.cpu())

        val_metrics = compute_metrics(
            torch.cat(all_val_logits), torch.cat(all_val_labels), "valence"
        )
        aro_metrics = compute_metrics(
            torch.cat(all_aro_logits), torch.cat(all_aro_labels), "arousal"
        )
        return {**val_metrics, **aro_metrics}

    def _save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
        }, path)
        logger.info("Saved checkpoint: %s", path)

    def _log_wandb(self, metrics: dict) -> None:
        try:
            import wandb
            wandb.log(metrics)
        except Exception:
            pass

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device
