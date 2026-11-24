"""Label-based hospital fingerprints and an offline test of whether ANY fingerprint predicts transfer.

Pure pandas/numpy (no torch). Every fingerprint is computed from a hospital's TRAIN rows only
(``topology.train_frame``), and only aggregate counts would leave a hospital — the same kind of summary
a microbiology lab already publishes as its yearly antibiogram (CLSI M39).

Why label-based: the v0 fingerprint (``topology.compute_fingerprint``) summarises graph SHAPE (degrees,
density, homophily). Whether hospital j's model works for hospital i depends on whether they map
(organism, antibiotic, patient) to resistance the same way, i.e. on P(y | x). These fingerprints
measure that directly. All of them return an (n, n) DISSIMILARITY matrix ``D`` with D[i, i] = 0, where
D[i, j] says how badly j's resistance pattern describes i's data (some are asymmetric, like the target).

The target: SimFT (notebook 19) saved ``L[i, j]`` = loss of hospital j's model on hospital i's validation
rows, every round. ``G = L - diag(L)[:, None]`` (how much worse j's model is than i's own, on i's data)
is the ground truth any similarity method is trying to predict. ``score_against`` compares a candidate
``D`` with ``G`` (pooled and per-row Spearman, plus a permutation null that relabels hospitals).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .data_loader import ABX, ORG

PRIOR_STRENGTH = 10.0   # pseudo-tests pulling a sparse pair's rate toward the hospital's own overall rate
MIN_PAIR = 10           # a pair must have this many tests in BOTH hospitals to be compared


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pair_counts(frames: Sequence[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list]:
    """(K, N) resistant counts and test counts per hospital (rows) and (organism, antibiotic) pair."""
    keys = [f[ORG].astype(str) + "|" + f[ABX].astype(str) for f in frames]
    pairs = sorted(set().union(*[set(k.unique()) for k in keys]))
    idx = {p: i for i, p in enumerate(pairs)}
    K = np.zeros((len(frames), len(pairs)))
    N = np.zeros((len(frames), len(pairs)))
    for h, (f, k) in enumerate(zip(frames, keys)):
        g = pd.DataFrame({"k": k.to_numpy(), "y": f["label"].to_numpy()}).groupby("k")["y"].agg(["sum", "size"])
        cols = [idx[p] for p in g.index]
        K[h, cols] = g["sum"].to_numpy()
        N[h, cols] = g["size"].to_numpy()
    return K, N, pairs


def _shrunk_rates(K, N):
    """Each pair's rate shrunk toward that hospital's own overall rate (empirical Bayes, like graph_build)."""
    base = K.sum(1, keepdims=True) / np.maximum(N.sum(1, keepdims=True), 1)
    return (K + PRIOR_STRENGTH * base) / (N + PRIOR_STRENGTH)


def antibiogram_distance(frames, centred: bool = False, min_pair: int = MIN_PAIR) -> np.ndarray:
    """Test-count-weighted mean |logit r_i - logit r_j| over pairs seen >= ``min_pair`` times in both.

    ``centred=True`` first removes each hospital's average logit offset (on the shared pairs), so a
    hospital that is just more resistant everywhere (a base-rate shift, which a per-hospital cutoff
    already fixes) is NOT called different; only the PATTERN across pairs counts."""
    K, N, _ = pair_counts(frames)
    L = _logit(_shrunk_rates(K, N))
    n = len(frames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m = (N[i] >= min_pair) & (N[j] >= min_pair)
            if not m.any():
                D[i, j] = np.nan
                continue
            w = np.minimum(N[i, m], N[j, m])
            diff = L[i, m] - L[j, m]
            if centred:
                diff = diff - np.average(diff, weights=w)
            D[i, j] = np.average(np.abs(diff), weights=w)
    return D


def cross_antibiogram_loss(frames) -> np.ndarray:
    """D[i, j] = mean BCE on hospital i's train tests when each test is predicted by hospital j's
    antibiogram rate for that pair (pairs j never saw fall back to j's overall rate), minus the same
    with i's own antibiogram. A non-neural stand-in for SimFT's cross-evaluation: j shares only its
    antibiogram, i scores it on its own rows and keeps the number. Asymmetric, like the target."""
    K, N, _ = pair_counts(frames)
    R = _shrunk_rates(K, N)
    n = len(frames)
    loss = np.zeros((n, n))
    for i in range(n):
        k, t = K[i], N[i]
        for j in range(n):
            p = np.clip(R[j], 1e-4, 1 - 1e-4)
            loss[i, j] = -(k * np.log(p) + (t - k) * np.log(1 - p)).sum() / t.sum()
    return loss - np.diag(loss)[:, None]


def casemix_distance(frames) -> np.ndarray:
    """Jensen-Shannon divergence between hospitals' (organism, antibiotic) TEST-FREQUENCY distributions.
    Label-free: a control for "different bugs" as opposed to "different resistance"."""
    _, N, _ = pair_counts(frames)
    P = N / N.sum(1, keepdims=True)
    n = len(frames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m = 0.5 * (P[i] + P[j])
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = lambda a: np.nansum(np.where(a > 0, a * np.log(a / m), 0.0))
            D[i, j] = 0.5 * kl(P[i]) + 0.5 * kl(P[j])
    return D


def antibiotic_rate_distance(frames, min_tests: int = 30) -> np.ndarray:
    """Mean |logit| difference of per-ANTIBIOTIC resistance rates (54-dim summary; coarser than pairs)."""
    abx = sorted(set().union(*[set(f[ABX].astype(str)) for f in frames]))
    R = np.full((len(frames), len(abx)), np.nan)
    for h, f in enumerate(frames):
        g = f.groupby(f[ABX].astype(str))["label"].agg(["sum", "size"])
        g = g[g["size"] >= min_tests]
        base = f["label"].mean()
        r = (g["sum"] + PRIOR_STRENGTH * base) / (g["size"] + PRIOR_STRENGTH)
        R[h, [abx.index(a) for a in g.index]] = _logit(r.to_numpy())
    n = len(frames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m = ~np.isnan(R[i]) & ~np.isnan(R[j])
            D[i, j] = np.abs(R[i, m] - R[j, m]).mean() if m.any() and i != j else 0.0
    return D


def topology_v3_distance(frames) -> np.ndarray:
    """The fingerprint distance topo v3 actually used (z-scored v0 features, size features dropped)."""
    from .topology import (DEFAULT_EXCLUDE, FEATURE_NAMES, SIZE_FEATURES, compute_fingerprint,
                           fingerprint_matrix, select_features, zscore)
    M = fingerprint_matrix([compute_fingerprint(f) for f in frames])
    kept, _ = select_features(M, list(FEATURE_NAMES), exclude=DEFAULT_EXCLUDE + SIZE_FEATURES)
    z = zscore(kept)
    return np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)


CANDIDATES: dict[str, Callable] = {
    "topology_v3 (current)": topology_v3_distance,
    "casemix JS (label-free)": casemix_distance,
    "antibiotic rates": antibiotic_rate_distance,
    "antibiogram": antibiogram_distance,
    "antibiogram, centred": lambda fr: antibiogram_distance(fr, centred=True),
    "cross-antibiogram loss": cross_antibiogram_loss,
}


def weights_from_dissimilarity(D, temperature: float = 1.0) -> np.ndarray:
    """Row-stochastic personalized-aggregation matrix from a dissimilarity D, with the same convention as
    ``topology.personalized_weights``: the diagonal is set to 0, D is divided by its mean off-diagonal value
    (so T means the same on every split), then ``W[i] = softmax(-D[i] / T)``. T = inf gives uniform 1/n,
    T -> 0 gives the identity (local training). Negative entries (j's pattern fits i better than i's own,
    possible after shrinkage) simply get more weight."""
    D = np.array(D, dtype=np.float64)
    n = D.shape[0]
    if n == 1:
        return np.ones((1, 1))
    if temperature is None or np.isinf(temperature):
        return np.full((n, n), 1.0 / n)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 (or inf), got {temperature}")
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D, nan=np.nanmax(D) if np.isfinite(D).any() else 0.0)
    off = D[~np.eye(n, dtype=bool)]
    scale = float(np.abs(off).mean())
    if scale <= 1e-12:
        return np.full((n, n), 1.0 / n)
    z = -D / (scale * temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---- scoring a candidate against the measured transfer gaps -----------------------------------------
def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def _stats(D, G):
    n = len(G)
    off = ~np.eye(n, dtype=bool)
    pooled = _spearman(D[off], G[off])
    rows = [_spearman(D[i][off[i]], G[i][off[i]]) for i in range(n)] if n >= 4 else []
    rows = [r for r in rows if r == r]
    return pooled, (float(np.mean(rows)) if rows else float("nan"))


def score_against(D: np.ndarray, G: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """Spearman of candidate dissimilarity D with the transfer gap G (off-diagonal, pooled), the mean
    per-hospital (row) Spearman (does D rank MY neighbours right? needs n >= 4), and a permutation p for
    the pooled value: hospitals of D are relabelled at random (rows and columns together)."""
    D, G = np.asarray(D, float), np.asarray(G, float)
    pooled, row = _stats(D, G)
    rng = np.random.default_rng(seed)
    n = len(G)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        null.append(_stats(D[np.ix_(p, p)], G)[0])
    null = np.array([x for x in null if x == x])
    p_val = float((np.sum(null >= pooled) + 1) / (len(null) + 1)) if pooled == pooled else float("nan")
    return {"pooled": pooled, "row": row, "p_perm": p_val}


def transfer_gap(L) -> np.ndarray:
    """G[i, j] = L[i, j] - L[i, i]: how much worse hospital j's model is than i's own, on i's rows."""
    L = np.asarray(L, float)
    return L - np.diag(L)[:, None]
