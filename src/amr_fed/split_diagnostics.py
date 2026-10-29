"""How different are the hospitals of a split in the thing a model has to learn? (pure pandas/numpy)

A topology- or similarity-aware method can only beat FedAvg + fine-tune if hospitals differ in
P(resistant | organism, antibiotic, ...) in a way a global model plus a local bias/threshold cannot absorb.
This module measures that BEFORE any training, from per-(organism, antibiotic, hospital) counts.

Three nested logistic models on the aggregated counts (one row per pair x hospital):

  M0  logit p = a_pair                      every hospital shares one antibiogram
  M1  logit p = a_pair + b_hospital         plus a hospital-wide base-rate shift (what a per-hospital
                                            cutoff / fine-tuned bias can fix)
  M2  saturated (pair x hospital)           every hospital has its own antibiogram

  prior_shift = (dev(M0) - dev(M1)) / n_tests      base-rate (label prior) differences
  concept     = dev(M1) / n_tests                  pair-level differences beyond a base-rate shift

Both are in nats per test (x1000 in reports). Sampling noise alone makes ``concept`` > 0 (a pair seen
40 times in a hospital will not match the pooled rate exactly), so every split is compared with a
**null**: the same hospital sizes, patients assigned at random. ``excess = observed - null mean`` is the
part that is real. Patients, not tests, are shuffled, so within-patient correlation is kept in the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import ABX, ORG, PK


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _counts(df: pd.DataFrame, assignment: pd.Series, min_pair_tests: int, strata=None) -> pd.DataFrame:
    hosp = df[PK].map(assignment)
    key = df[ORG].astype(str) + "|" + df[ABX].astype(str)
    if strata is not None:
        key = key + "|" + pd.Series(np.asarray(strata), index=df.index).astype(str)
    d = pd.DataFrame({"pair": key.to_numpy(),
                      "h": hosp.to_numpy(), "y": df["label"].to_numpy()})
    d = d[d["h"].notna()]
    tot = d.groupby("pair").size()
    d = d[d["pair"].isin(tot.index[tot >= min_pair_tests])]
    g = d.groupby(["pair", "h"])["y"].agg(["sum", "size"]).reset_index()
    return g.rename(columns={"sum": "k", "size": "n"})


def _binom_dev(k, n, p) -> float:
    """Deviance (2 x log-likelihood gap to the saturated model) of rates p for counts k of n."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    obs = k / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(obs / p), 0.0)
        t0 = np.where(n - k > 0, (n - k) * np.log((1 - obs) / (1 - p)), 0.0)
    return float(2.0 * (t1 + t0).sum())


def _fit_additive(pair_idx, h_idx, k, n, n_pair, n_h, iters: int = 200, tol: float = 1e-9):
    """logit p = a[pair] + b[h] by alternating 1-D Newton steps (counts-weighted). b[0] fixed at 0."""
    a = np.zeros(n_pair)
    b = np.zeros(n_h)
    # start a at the pooled pair logit
    kp = np.bincount(pair_idx, k, n_pair)
    np_ = np.bincount(pair_idx, n, n_pair)
    r = np.clip((kp + 0.5) / (np_ + 1.0), 1e-6, 1 - 1e-6)
    a = np.log(r / (1 - r))
    for _ in range(iters):
        old_b = b.copy()
        for _inner in range(3):
            eta = a[pair_idx] + b[h_idx]
            p = _sigmoid(eta)
            g = np.bincount(pair_idx, k - n * p, n_pair)
            hess = np.bincount(pair_idx, n * p * (1 - p), n_pair) + 1e-9
            a += g / hess
        for _inner in range(3):
            eta = a[pair_idx] + b[h_idx]
            p = _sigmoid(eta)
            g = np.bincount(h_idx, k - n * p, n_h)
            hess = np.bincount(h_idx, n * p * (1 - p), n_h) + 1e-9
            b += g / hess
            b -= b[0]
        if np.max(np.abs(b - old_b)) < tol:
            break
    return _sigmoid(a[pair_idx] + b[h_idx])


def history_strata(df: pd.DataFrame) -> np.ndarray:
    """Per-test stratum from the patient's leakage-safe prior resistance (what the GNN's history features
    carry): 'first' for a first culture, else the quartile of the prior resistance rate. Passing this as
    ``strata`` asks "do hospitals differ even among patients with the same pair AND the same history?"."""
    from .graph_build import _history_raw
    raw = _history_raw(df, global_rate=float(df["label"].mean()))
    n_prior, rate = np.expm1(raw[:, 0]), raw[:, 1]
    q = np.quantile(rate[n_prior > 0], [0.25, 0.5, 0.75])
    return np.where(n_prior > 0, np.searchsorted(q, rate).astype(str), "first")


def heterogeneity(df: pd.DataFrame, assignment: pd.Series, min_pair_tests: int = 30, strata=None) -> dict:
    """prior_shift and concept (nats per 1000 tests) for one assignment (patient -> hospital).
    ``strata`` (one value per row of ``df``) refines the pair key, e.g. ``history_strata(df)``."""
    g = _counts(df, assignment, min_pair_tests, strata)
    pair_idx = pd.factorize(g["pair"])[0]
    h_idx = pd.factorize(g["h"], sort=True)[0]
    k, n = g["k"].to_numpy(float), g["n"].to_numpy(float)
    n_pair, n_h = pair_idx.max() + 1, h_idx.max() + 1
    p0 = (np.bincount(pair_idx, k, n_pair) / np.bincount(pair_idx, n, n_pair))[pair_idx]
    p1 = _fit_additive(pair_idx, h_idx, k, n, n_pair, n_h)
    d0, d1 = _binom_dev(k, n, p0), _binom_dev(k, n, p1)
    N = n.sum()
    return {"prior_shift": 1000 * (d0 - d1) / N, "concept": 1000 * d1 / N,
            "n_tests": int(N), "n_pairs": int(n_pair), "n_hospitals": int(n_h)}


def random_like(assignment: pd.Series, seed: int) -> pd.Series:
    """Same hospital sizes, patients shuffled at random (the null)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(assignment.to_numpy()), index=assignment.index)


def heterogeneity_vs_null(df: pd.DataFrame, assignment: pd.Series, n_null: int = 5,
                          min_pair_tests: int = 30, seed: int = 0, strata=None) -> dict:
    """``heterogeneity`` plus the random-assignment null (mean, sd over ``n_null`` shuffles) and the
    excess over it. ``concept_ratio`` = observed / null concept (1 = no real pair-level difference)."""
    obs = heterogeneity(df, assignment, min_pair_tests, strata)
    nulls = [heterogeneity(df, random_like(assignment, seed + i), min_pair_tests, strata) for i in range(n_null)]
    out = dict(obs)
    for key in ("prior_shift", "concept"):
        v = np.array([x[key] for x in nulls])
        out[f"{key}_null"] = float(v.mean())
        out[f"{key}_null_sd"] = float(v.std(ddof=1)) if n_null > 1 else float("nan")
        out[f"{key}_excess"] = obs[key] - float(v.mean())
    out["concept_ratio"] = obs["concept"] / out["concept_null"] if out["concept_null"] > 0 else float("nan")
    return out
