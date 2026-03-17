"""Leave-One-Subject-Out (LOSO) evaluation.

With N=21, trains on 20 subjects and evaluates on 1, repeats 21 times.
Reports mean ± std over subjects.

Critical: normalization statistics are ALWAYS fit on training subjects only.
Never use test subject data for normalization — this is a common leakage bug.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class LOSOEvaluator:
    """Leave-One-Subject-Out cross-subject evaluation protocol.

    Args:
        all_subject_ids:   Full list of subject IDs (21 for Daly EEG+fMRI).
        build_dataset_fn:  Callable(subject_ids, split) → Dataset.
                           split is 'train' or 'test'.
        build_model_fn:    Callable() → BrainSulcal (fresh model each fold).
        train_fn:          Callable(model, train_loader, val_loader) → metrics.
        eval_fn:           Callable(model, test_loader) → metrics.
        output_dir:        Where to save per-fold results.
    """

    def __init__(
        self,
        all_subject_ids: list[str],
        build_dataset_fn: Callable,
        build_model_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        output_dir: str | Path,
    ):
        self.all_subject_ids = all_subject_ids
        self.build_dataset_fn = build_dataset_fn
        self.build_model_fn = build_model_fn
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, float]:
        """Run all LOSO folds and aggregate results.

        Returns:
            Dict with mean ± std over subjects for each metric.
        """
        per_subject_results: dict[str, list[float]] = {}

        for test_subject in self.all_subject_ids:
            train_subjects = [s for s in self.all_subject_ids if s != test_subject]
            logger.info(
                "LOSO fold: test=%s | train=%d subjects",
                test_subject, len(train_subjects),
            )

            train_dataset = self.build_dataset_fn(train_subjects, split="train")
            test_dataset = self.build_dataset_fn([test_subject], split="test")

            model = self.build_model_fn()
            train_metrics = self.train_fn(model, train_dataset)
            test_metrics = self.eval_fn(model, test_dataset)

            for k, v in test_metrics.items():
                per_subject_results.setdefault(k, []).append(v)

            logger.info("  %s → %s", test_subject, test_metrics)

        # Aggregate: mean ± std
        summary: dict[str, float] = {}
        for metric, values in per_subject_results.items():
            arr = np.array(values)
            summary[f"{metric}_mean"] = float(arr.mean())
            summary[f"{metric}_std"] = float(arr.std())
            logger.info(
                "LOSO %s: %.3f ± %.3f",
                metric, arr.mean(), arr.std(),
            )

        self._save_results(per_subject_results, summary)
        return summary

    def _save_results(
        self,
        per_subject: dict[str, list[float]],
        summary: dict[str, float],
    ) -> None:
        import pandas as pd

        df = pd.DataFrame(per_subject, index=self.all_subject_ids)
        df.to_csv(self.output_dir / "loso_results.csv")

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "loso_summary.csv", index=False)

        logger.info("Results saved to %s", self.output_dir)
