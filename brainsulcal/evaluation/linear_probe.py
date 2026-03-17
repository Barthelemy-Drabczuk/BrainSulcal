"""Linear probe evaluator: frozen backbone → linear classifier.

Used to assess representation quality without fine-tuning the full model.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LinearProbeEvaluator:
    """Evaluate a frozen backbone with a linear classifier.

    Args:
        model:     BrainSulcal model (backbone will be frozen).
        device:    Torch device.
        n_classes: Number of output classes (default 2).
    """

    def __init__(self, model: nn.Module, device: torch.device, n_classes: int = 2):
        self.model = model
        self.device = device
        self.n_classes = n_classes

    def extract_features(
        self, loader: DataLoader, montage_info: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract z_fused representations from a frozen model.

        Returns:
            features: (N, d_fused)
            labels:   (N,)  — valence labels
        """
        self.model.eval()
        all_feats, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(self.device)
                val_labels = batch["valence_label"]
                sulcal_emb = batch.get("sulcal_embeddings")
                sulcal_mask = batch.get("sulcal_mask")

                if sulcal_emb is not None:
                    sulcal_emb = sulcal_emb.to(self.device)
                    sulcal_mask = sulcal_mask.to(self.device)
                else:
                    B, C, T = eeg.shape
                    d = self.model.config.champollion_input_dim
                    sulcal_emb = torch.zeros(B, 56, d, device=self.device)
                    sulcal_mask = torch.zeros(B, 56, dtype=torch.bool, device=self.device)

                outputs = self.model(eeg, montage_info, sulcal_emb, sulcal_mask)
                all_feats.append(outputs["z_fused"].cpu().numpy())
                all_labels.append(val_labels.numpy())

        return np.concatenate(all_feats), np.concatenate(all_labels)

    def evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        montage_info: dict,
    ) -> dict[str, float]:
        """Fit linear probe on train features, evaluate on test features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        X_train, y_train = self.extract_features(train_loader, montage_info)
        X_test, y_test = self.extract_features(test_loader, montage_info)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        try:
            auroc = roc_auc_score(y_test, probs[:, 1])
        except ValueError:
            auroc = float("nan")

        logger.info(
            "Linear probe — accuracy: %.3f | F1: %.3f | AUROC: %.3f",
            accuracy, f1, auroc,
        )

        return {"accuracy": accuracy, "f1_macro": f1, "auroc": auroc}
