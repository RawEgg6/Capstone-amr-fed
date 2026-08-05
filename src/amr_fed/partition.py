"""Phase-2 partitioning: split the single institution into simulated hospitals.

Main method: a Dirichlet(alpha) ward-mixture (Hsu et al. 2019; Li et al. 2022,
NIID-Bench) — each hospital is a different blend of wards, alpha dials how
different they are (small alpha = strong non-IID, large = near-IID). Plus plain
ward and ADI-quartile baseline splits.

Every partition assigns each PATIENT to exactly one hospital (no leakage): the
returned object is a Series indexed by patient id -> hospital label. A patient's
"home ward" is their highest-acuity ward across all their cultures (priority
ICU>ER>IP>OP, else NONE) — the worst-value-in-window convention used by ICU
severity scores.

Pure pandas/numpy (no torch) — runs and is verifiable on any stack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loader import ADI, PK

WARD_COL = "ward"  # priority-collapsed per-culture ward added by data_loader


def assign_home_ward(df: pd.DataFrame) -> pd.Series:
    """Each patient's highest-priority ward across their cultures (index=patient)."""
    order = {w: i for i, w in enumerate(config.WARD_PRIORITY)}  # ICU=0 ... OP=3
    d = df[[PK, WARD_COL]].copy()
    d["rank"] = d[WARD_COL].map(lambda w: order.get(w, len(order)))  # NONE -> lowest priority
    return d.sort_values("rank").groupby(PK)[WARD_COL].first()


def _apportion(n: int, props: np.ndarray) -> np.ndarray:
    """Split n items across len(props) buckets by proportions (largest-remainder)."""
    exact = props * n
    counts = np.floor(exact).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        for c in np.argsort(exact - counts)[::-1][:rem]:
            counts[c] += 1
    return counts


def dirichlet_ward_mixture(df: pd.DataFrame, n_clients: int = 5, alpha: float = 0.5,
                           seed: int = config.SEED) -> pd.Series:
    """Assign each patient to one of n_clients hospitals via a Dirichlet(alpha)
    ward-mixture. Returns Series index=patient -> client id (0..n_clients-1)."""
    home = assign_home_ward(df)
    rng = np.random.default_rng(seed)
    assignment = pd.Series(-1, index=home.index, dtype="int64")
    for ward in sorted(home.unique()):
        pats = home.index[home == ward].to_numpy()
        rng.shuffle(pats)
        counts = _apportion(len(pats), rng.dirichlet(alpha * np.ones(n_clients)))
        start = 0
        for c, k in enumerate(counts):
            assignment.loc[pats[start:start + k]] = c
            start += k
    assert (assignment >= 0).all(), "some patient was left unassigned"
    return assignment


def ward_baseline(df: pd.DataFrame) -> pd.Series:
    """Baseline split: each home ward is its own hospital (patient -> ward)."""
    return assign_home_ward(df)


def adi_baseline(df: pd.DataFrame, n_clients: int = 4) -> pd.Series:
    """Baseline split: ADI-score quantiles (patient -> 'Q1'..'Qk' or 'Unknown')."""
    adi = pd.to_numeric(df.groupby(PK)[ADI].first(), errors="coerce")
    known = adi.dropna()
    # bin on ranks (not raw values) so heavy ADI-score ties don't collapse the quantile edges
    q = pd.Series("Unknown", index=adi.index, dtype="object")
    q.loc[known.index] = pd.qcut(known.rank(method="first"), n_clients,
                                 labels=[f"Q{i+1}" for i in range(n_clients)]).astype("object")
    return q


def partition_summary(df: pd.DataFrame, assignment: pd.Series) -> tuple[pd.DataFrame, float]:
    """Per-hospital size + resistance rate, and the non-IID score (test-weighted
    std of resistance rate across hospitals — larger = more heterogeneous)."""
    d = df[[PK, "label"]].copy()
    d["client"] = d[PK].map(assignment)
    g = d.groupby("client")["label"].agg(n_tests="size", resist_rate="mean")
    g["n_patients"] = d.groupby("client")[PK].nunique()
    g = g[g.index.astype(str) != "NONE"]  # drop the ward-less bucket from the score
    w = g["n_tests"] / g["n_tests"].sum()
    wmean = (w * g["resist_rate"]).sum()
    non_iid = float(np.sqrt((w * (g["resist_rate"] - wmean) ** 2).sum()))
    return g.round(4), non_iid


def alpha_sweep(df: pd.DataFrame, alphas=(0.1, 0.5, 1.0), n_clients: int = 5,
                seed: int = config.SEED) -> pd.DataFrame:
    """Non-IID score of the Dirichlet ward-mixture across alpha (proves the dial)."""
    rows = []
    for a in alphas:
        _, non_iid = partition_summary(df, dirichlet_ward_mixture(df, n_clients, a, seed))
        rows.append({"alpha": a, "non_iid_wstd": round(non_iid, 4)})
    return pd.DataFrame(rows)


def _self_check() -> None:
    from .data_loader import load_cohort_frame
    df = load_cohort_frame()
    n_pat = df[PK].nunique()

    print("=== home ward distribution (patients) ===")
    print(assign_home_ward(df).value_counts(dropna=False).to_dict())

    print("\n=== baseline: ward split ===")
    g, niid = partition_summary(df, ward_baseline(df))
    print(g.to_string()); print("non-IID(wstd):", niid)

    print("\n=== baseline: ADI quartile split ===")
    g, niid = partition_summary(df, adi_baseline(df))
    print(g.to_string()); print("non-IID(wstd):", niid)

    print("\n=== Dirichlet ward-mixture: alpha sweep (5 hospitals) ===")
    sweep = alpha_sweep(df)
    print(sweep.to_string(index=False))

    # the dial must work: smaller alpha -> more heterogeneity
    s = sweep.sort_values("alpha")
    assert s["non_iid_wstd"].iloc[0] > s["non_iid_wstd"].iloc[-1], \
        "non-IID should shrink as alpha grows — the Dirichlet dial isn't working"
    # no leakage: every patient assigned exactly once
    assign = dirichlet_ward_mixture(df, 5, 0.5)
    assert assign.index.is_unique and len(assign) == n_pat, "patient assignment not 1:1"
    print("\nOK: partition self-check passed (dial works, no patient leakage).")


if __name__ == "__main__":
    _self_check()
