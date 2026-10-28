"""Data regimes for the low-data and new-hospital experiments (notebook 18).

Pure pandas/numpy (torch-free) so it can be unit-tested; the training drivers live in ``new_site.py``.

Every function keeps the hospital's ORIGINAL patient-grouped split (``graph_build._patient_grouped_split``
on the full hospital frame) and only removes train / validation patients, so the TEST patients are the same
at every data level and results are comparable across levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_loader import PK
from ..graph_build import _patient_grouped_split


def _restrict(sub: pd.DataFrame, seed: int, rng: np.random.Generator, n_train, n_val):
    """Keep all test patients, ``n_train`` train patients and ``n_val`` validation patients
    (None = keep all). Returns (reduced frame, split-code dict for the kept patients)."""
    code = _patient_grouped_split(sub[PK].to_numpy(), seed)
    pats = np.array(sorted(code, key=str))
    kind = np.array([code[p] for p in pats])
    keep = [pats[kind == 2]]
    for k, n in ((0, n_train), (1, n_val)):
        pool = pats[kind == k]
        if n is None or n >= len(pool):
            keep.append(pool)
        else:
            keep.append(rng.choice(pool, size=max(int(n), 1), replace=False))
    kept = set(np.concatenate(keep).tolist())
    return sub[sub[PK].isin(kept)], {p: code[p] for p in kept}


def subsample_frame(sub: pd.DataFrame, seed: int, cid: int, frac: float):
    """Low-data regime: keep fraction ``frac`` of the hospital's train and validation patients
    (at least 2 of each), all test patients. Deterministic in (seed, cid, frac)."""
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    code = _patient_grouped_split(sub[PK].to_numpy(), seed)
    kind = np.array(list(code.values()))
    n_tr, n_va = int((kind == 0).sum()), int((kind == 1).sum())
    rng = np.random.default_rng([int(seed), int(cid), int(round(frac * 1e6))])
    return _restrict(sub, seed, rng, max(2, round(frac * n_tr)), max(2, round(frac * n_va)))


def slice_frame(sub: pd.DataFrame, seed: int, n_train: int | None):
    """New-hospital regime: the hospital has only ``n_train`` labelled training patients (None = all)
    and ``max(30, n_train // 2)`` validation patients; the whole test set is kept."""
    n_val = None if n_train is None else max(30, n_train // 2)
    rng = np.random.default_rng([int(seed), 7919, 0 if n_train is None else int(n_train) + 1])
    return _restrict(sub, seed, rng, n_train, n_val)


def holdout_hospital(assign: pd.Series, rule: str) -> int:
    """Which hospital to hold out, by a size rule fixed in advance: 'median' = the 5th largest of 10
    (the (n+1)//2-th largest in general), 'smallest' = the smallest."""
    sizes = assign.value_counts().sort_values(ascending=False, kind="stable")
    if rule == "smallest":
        return int(sizes.index[-1])
    if rule == "median":
        return int(sizes.index[(len(sizes) + 1) // 2 - 1])
    raise ValueError(f"unknown holdout rule {rule!r}")
