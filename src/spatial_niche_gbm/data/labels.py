"""Niche and MES label construction from SNUH atlas annotations.

Label rules are loaded from configs/labels.yaml so they stay auditable.
A spot gets a niche label only when its annotations match a rule; the first
matching rule in `priority` order wins. Spots matching no rule are "exclude".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NICHE_CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]
EXCLUDE_LABEL = "exclude"


def assign_niche_labels(meta: pd.DataFrame, rules: dict) -> pd.Series:
    """Assign a niche label to each spot from its annotations.

    Parameters
    ----------
    meta : per-spot metadata with columns AF, metaprogram (and others).
    rules : parsed configs/labels.yaml dict with keys niche_rules, priority.

    Returns
    -------
    pd.Series of niche labels in NICHE_CLASSES + "exclude", indexed like meta.
    """
    niche_rules = rules["niche_rules"]
    priority = rules["priority"]

    labels = pd.Series(EXCLUDE_LABEL, index=meta.index, dtype=object)
    assigned = np.zeros(len(meta), dtype=bool)

    for niche in priority:
        rule = niche_rules[niche]
        mask = np.ones(len(meta), dtype=bool)
        for col, allowed in rule.items():
            negate = False
            if col.startswith("not_"):
                negate = True
                col = col[4:]
            if col not in meta.columns:
                raise KeyError(f"label rule references missing column '{col}'")
            vals = meta[col].astype(str).values
            in_set = np.isin(vals, [str(a) for a in allowed])
            mask &= ~in_set if negate else in_set
        take = mask & ~assigned
        labels.values[take] = niche
        assigned |= mask

    return labels


def mes_target(meta: pd.DataFrame, rules: dict) -> pd.Series:
    """Binary MES indicator per spot from greenwald_metaprograms MES states.

    Returns float Series in {0.0, 1.0}; NaN where the annotation is missing.
    """
    mes_states = set(rules["mes_states"])
    col = meta["greenwald_metaprograms"].astype(str)
    out = col.isin(mes_states).astype(float)
    out[col.isin(["nan", "None", "NA"])] = np.nan
    return out


def encode_niche(labels: pd.Series) -> np.ndarray:
    """Encode niche labels as int64 class ids; exclude -> -1."""
    mapping = {c: i for i, c in enumerate(NICHE_CLASSES)}
    return labels.map(lambda x: mapping.get(x, -1)).values.astype(np.int64)
