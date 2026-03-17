"""Evaluation metrics: accuracy, F1 macro, AUROC for valence/arousal."""

from __future__ import annotations

import numpy as np
import torch


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_name: str = "valence",
) -> dict[str, float]:
    """Compute classification metrics from logits and labels.

    Args:
        logits:    (N, n_classes) raw logits
        labels:    (N,) integer labels
        task_name: label prefix for the returned dict keys

    Returns dict with keys: {task_name}/accuracy, {task_name}/f1_macro, {task_name}/auroc
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)
    labels_np = labels.detach().cpu().numpy()

    accuracy = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(labels_np, probs[:, 1])
        else:
            auroc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    return {
        f"{task_name}/accuracy": float(accuracy),
        f"{task_name}/f1_macro": float(f1),
        f"{task_name}/auroc": float(auroc),
    }


def router_bias_entropy(router_logits: torch.Tensor) -> float:
    """Compute entropy of MoE routing distribution.

    Args:
        router_logits: (B, n_experts) or (B*T, n_experts)

    Returns:
        Mean entropy in bits. Values below 0.5 bits indicate expert collapse.
    """
    probs = torch.softmax(router_logits, dim=-1)
    log_probs = torch.log_softmax(router_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B,) in nats
    entropy_bits = entropy / np.log(2)           # Convert to bits
    return float(entropy_bits.mean().item())
