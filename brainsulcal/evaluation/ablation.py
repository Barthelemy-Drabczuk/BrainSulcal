"""Ablation study: 4 conditions isolating each contribution.

| Condition        | sulcal_prior | router_bias | prefix_token |
|------------------|-------------|------------|-------------|
| BrainOmni only   | No           | No          | No           |
| + Prefix only    | Yes          | No          | Yes          |
| + Router only    | Yes          | Yes         | No           |
| Full BrainSulcal | Yes          | Yes         | Yes          |
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ABLATION_CONDITIONS = {
    "brainomni_only": {
        "use_sulcal_prior": False,
        "use_router_bias": False,
        "use_prefix_token": False,
    },
    "prefix_only": {
        "use_sulcal_prior": True,
        "use_router_bias": False,
        "use_prefix_token": True,
    },
    "router_only": {
        "use_sulcal_prior": True,
        "use_router_bias": True,
        "use_prefix_token": False,
    },
    "full_brainsulcal": {
        "use_sulcal_prior": True,
        "use_router_bias": True,
        "use_prefix_token": True,
    },
}


class AblationStudy:
    """Run all 4 ablation conditions using LOSO evaluation.

    Args:
        base_config:       Base BrainSulcalConfig (flags will be overridden).
        loso_runner_fn:    Callable(config) → LOSOEvaluator results dict.
        output_dir:        Where to save the ablation table.
    """

    def __init__(self, base_config, loso_runner_fn, output_dir: str | Path):
        self.base_config = base_config
        self.loso_runner_fn = loso_runner_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, dict[str, float]]:
        """Run all conditions and save ablation table.

        Returns:
            Dict mapping condition name → metrics dict.
        """
        from copy import deepcopy

        all_results: dict[str, dict[str, float]] = {}

        for condition_name, flags in ABLATION_CONDITIONS.items():
            logger.info("Running ablation condition: %s", condition_name)
            config = deepcopy(self.base_config)
            for flag, value in flags.items():
                setattr(config, flag, value)

            results = self.loso_runner_fn(config)
            all_results[condition_name] = results
            logger.info("  %s → %s", condition_name, results)

        self._save_table(all_results)
        return all_results

    def _save_table(self, all_results: dict[str, dict[str, float]]) -> None:
        import pandas as pd

        df = pd.DataFrame(all_results).T
        df.to_csv(self.output_dir / "ablation_table.csv")
        logger.info("Ablation table saved to %s", self.output_dir)
        print("\nAblation Results:")
        print(df.to_string())
