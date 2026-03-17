"""Loss functions for BrainSulcal training.

Primary:    CrossEntropy for valence + arousal classification.
Optional:   InfoNCE alignment between z_sulcal and z_eeg_pooled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainSulcalLoss(nn.Module):
    """Combined classification + alignment loss.

    L_total = L_cls + lambda_align * L_align

    Args:
        lambda_align: Weight for InfoNCE alignment term (default 0.1).
                      Set to 0.0 to disable alignment loss.
        temperature:  InfoNCE temperature (default 0.07).
    """

    def __init__(self, lambda_align: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.lambda_align = lambda_align
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_valence: torch.Tensor,
        logits_arousal: torch.Tensor,
        valence_labels: torch.Tensor,
        arousal_labels: torch.Tensor,
        z_sulcal: torch.Tensor | None = None,
        z_eeg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            logits_valence:  (B, n_classes)
            logits_arousal:  (B, n_classes)
            valence_labels:  (B,) long
            arousal_labels:  (B,) long
            z_sulcal:        (B, d) optional — for alignment loss
            z_eeg:           (B, d) optional — for alignment loss

        Returns dict with keys: total, cls, align
        """
        l_cls = self.ce(logits_valence, valence_labels) + self.ce(logits_arousal, arousal_labels)

        l_align = torch.tensor(0.0, device=logits_valence.device)
        if (
            self.lambda_align > 0.0
            and z_sulcal is not None
            and z_eeg is not None
        ):
            l_align = infonce_loss(z_sulcal, z_eeg, temperature=self.temperature)

        l_total = l_cls + self.lambda_align * l_align

        return {"total": l_total, "cls": l_cls, "align": l_align}


def infonce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) contrastive loss.

    Treats each (z_a[i], z_b[i]) pair as a positive example and all
    other cross-batch pairs as negatives.

    Args:
        z_a: (B, d) — e.g. z_sulcal
        z_b: (B, d) — e.g. z_eeg_pooled
        temperature: softmax temperature

    Returns:
        Scalar InfoNCE loss.
    """
    B = z_a.shape[0]
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    # Similarity matrix (B, B)
    sim = torch.matmul(z_a, z_b.T) / temperature

    labels = torch.arange(B, device=z_a.device)
    loss_ab = F.cross_entropy(sim, labels)
    loss_ba = F.cross_entropy(sim.T, labels)

    return (loss_ab + loss_ba) / 2.0
