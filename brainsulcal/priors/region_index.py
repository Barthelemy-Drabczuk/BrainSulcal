"""Maps Champollion V1 region names to canonical indices (0-55).

Champollion V1 outputs 56 regional embeddings:
  28 sulcal regions × 2 hemispheres (left=0-27, right=28-55).
"""

# 28 sulcal region base names (Champollion V1 ordering)
_BASE_REGIONS = [
    "S.T.s.ter.asc.ant.",
    "S.T.s.ter.asc.post.",
    "F.C.M.ant.",
    "S.GSM.",
    "S.Fi.ant.",
    "S.Fi.post.",
    "OCCIPITAL",
    "S.O.1.",
    "S.O.2.",
    "S.Li.",
    "S.P.C.",
    "S.Po.",
    "S.GSM.",
    "S.C.",
    "F.I.P.",
    "S.T.pol.ant.",
    "S.T.pol.post.",
    "S.T.s.ant.",
    "S.T.s.post.",
    "S.F.inter.",
    "S.F.orb.",
    "S.F.inf.ant.",
    "S.F.inf.post.",
    "S.F.median.",
    "S.F.sup.",
    "S.Pa.inf.",
    "S.Pa.med.",
    "S.Pe.C.",
]

# Full 56 region names: left hemisphere (L.) then right hemisphere (R.)
REGION_NAMES: list[str] = (
    [f"L.{r}" for r in _BASE_REGIONS]
    + [f"R.{r}" for r in _BASE_REGIONS]
)

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(REGION_NAMES)}

# Special index for Heschl's gyrus (primary auditory cortex) — key for music processing.
# Heschl's gyrus is related to S.T.s regions; tracked explicitly for wandb logging.
HESCHL_INDICES: list[int] = [0, 1, 28, 29]  # L/R S.T.s.ter.asc.ant/post


def region_name_to_index(name: str) -> int:
    """Return the canonical index for a Champollion region name.

    Raises KeyError if the region is not found.
    """
    if name not in _NAME_TO_INDEX:
        raise KeyError(
            f"Unknown Champollion region: {name!r}. "
            f"Expected one of {REGION_NAMES[:4]}... (56 total)"
        )
    return _NAME_TO_INDEX[name]
