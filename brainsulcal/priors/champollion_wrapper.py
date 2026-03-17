"""Loads pre-computed Champollion V1 embeddings from disk.

Champollion outputs one CSV file per sulcal region per subject.
Region matching is done by name (via sulci_regions_champollion_V1.json),
NOT by assuming a fixed ordering of CSV files on disk.

Usage:
    wrapper = ChampollionWrapper(
        embeddings_dir="data/processed/champollion",
        region_file="external/champollion_pipeline/share/sulci_regions_champollion_V1.json",
    )
    embeddings, mask = wrapper.load_subject("sub-01")
    # embeddings: (56, embedding_dim)  mask: (56,) bool
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from brainsulcal.priors.region_index import REGION_NAMES, region_name_to_index

logger = logging.getLogger(__name__)

N_REGIONS = 56


class ChampollionWrapper:
    """Loads pre-computed Champollion sulcal embeddings for a subject.

    Embeddings must be pre-computed with scripts/01_precompute_champollion.py
    and stored as:
        {embeddings_dir}/{subject_id}.npy       shape: (56, embedding_dim)
        {embeddings_dir}/{subject_id}_mask.npy  shape: (56,) bool

    If no T1 MRI is available for a subject, returns zero embeddings with
    an all-False mask (ablation condition: no sulcal prior).
    """

    def __init__(self, embeddings_dir: str | Path, region_file: str | Path | None = None):
        self.embeddings_dir = Path(embeddings_dir)
        self.region_file = Path(region_file) if region_file else None
        self._embedding_dim: int | None = None

        # Load region name → index mapping from champollion_pipeline JSON if provided.
        # This overrides the default region_index.py ordering.
        self._champollion_region_order: list[str] | None = None
        if self.region_file and self.region_file.exists():
            with open(self.region_file) as f:
                data = json.load(f)
            # Expected JSON: {"regions": ["L.S.T.s.ter.asc.ant.", ...]}
            if "regions" in data:
                self._champollion_region_order = data["regions"]
                logger.info(
                    "Loaded %d Champollion regions from %s",
                    len(self._champollion_region_order),
                    self.region_file,
                )

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError(
                "embedding_dim unknown — call load_subject() at least once first."
            )
        return self._embedding_dim

    def load_subject(self, subject_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sulcal embeddings for one subject.

        Returns:
            embeddings: FloatTensor of shape (56, embedding_dim)
            mask:       BoolTensor of shape (56,) — True where embedding is valid
        """
        emb_path = self.embeddings_dir / f"{subject_id}.npy"
        mask_path = self.embeddings_dir / f"{subject_id}_mask.npy"

        if not emb_path.exists():
            logger.warning(
                "No Champollion embedding found for %s at %s — using zeros.",
                subject_id,
                emb_path,
            )
            return self._zero_embedding()

        embeddings_np = np.load(emb_path)   # (56, d) or (d,) for single region
        mask_np = np.load(mask_path) if mask_path.exists() else np.ones(N_REGIONS, dtype=bool)

        if embeddings_np.ndim == 1:
            # Legacy single-region format — should not happen but guard anyway
            raise ValueError(
                f"Expected 2D embedding array (56, d) for {subject_id}, "
                f"got shape {embeddings_np.shape}"
            )

        assert embeddings_np.shape[0] == N_REGIONS, (
            f"Expected {N_REGIONS} regions, got {embeddings_np.shape[0]} for {subject_id}"
        )

        self._embedding_dim = embeddings_np.shape[1]

        embeddings = torch.from_numpy(embeddings_np).float()
        mask = torch.from_numpy(mask_np.astype(bool))

        # Zero out masked positions to prevent NaN propagation
        embeddings[~mask] = 0.0

        return embeddings, mask

    def _zero_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero embeddings and all-False mask (no sulcal prior)."""
        d = self._embedding_dim if self._embedding_dim is not None else 1
        embeddings = torch.zeros(N_REGIONS, d)
        mask = torch.zeros(N_REGIONS, dtype=torch.bool)
        return embeddings, mask

    def load_batch(
        self, subject_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load embeddings for a batch of subjects.

        Returns:
            embeddings: (B, 56, embedding_dim)
            masks:      (B, 56) bool
        """
        results = [self.load_subject(sid) for sid in subject_ids]
        embeddings = torch.stack([r[0] for r in results])
        masks = torch.stack([r[1] for r in results])
        return embeddings, masks
