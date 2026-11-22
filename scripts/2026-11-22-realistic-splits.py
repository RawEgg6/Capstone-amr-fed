"""How different are our simulated hospitals compared with real-world spread?

Splits and diagnostics only, no training. For every split it reports hospital sizes, the label
heterogeneity ratio (same numbers as docs/2026-10-29-label-heterogeneity-and-fingerprints.md) and the
per-hospital resistance-rate range on three headline organism-drug pairs.

Run: ARMD_DIR=<path to ARMD> python scripts/2026-11-22-realistic-splits.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed import config  # noqa: E402
from amr_fed.data_loader import PK, load_cohort_frame  # noqa: E402
from amr_fed.partition import (antibiogram_conflict, era_cluster_split, louvain_split,  # noqa: E402
                               organism_community, patient_era, specimen_baseline, ward_baseline)
from amr_fed.split_diagnostics import heterogeneity_vs_null, history_strata  # noqa: E402

ORG, ABX = config.COLUMNS["organism"], config.COLUMNS["antibiotic"]
SEED = 42
PAIRS = [("ESCHERICHIA COLI", "Ceftriaxone"), ("ESCHERICHIA COLI", "Ciprofloxacin"),
         ("KLEBSIELLA PNEUMONIAE", "Ceftriaxone")]
MIN_TESTS = 30
OUT = Path(__file__).resolve().parents[1] / "results" / "realistic_splits.csv"


def mix(assignment: pd.Series, frac: float, seed: int = SEED) -> pd.Series:
    """Move ``frac`` of patients to a random OTHER hospital (same mechanics as era_mix)."""
    if frac <= 0:
        return assignment
    rng = np.random.default_rng(seed)
    ids = np.sort(assignment.unique())
    a = assignment.to_numpy()
    move = rng.random(len(a)) < frac
    pos = np.searchsorted(ids, a)
    new_pos = (pos + rng.integers(1, len(ids), len(a))) % len(ids)
    return pd.Series(np.where(move, ids[new_pos], a), index=assignment.index, name="hospital")


def relabel(a: pd.Series) -> pd.Series:
    return pd.Series(pd.factorize(a, sort=True)[0], index=a.index, name="hospital")


def composite(df, n_eras: int, other: pd.Series, floor: float = 0.01) -> pd.Series:
    """Hospital = (era, other attribute). Cells under ``floor`` of all patients join the era's biggest cell."""
    era = patient_era(df, n_eras)
    o = other.reindex(era.index).astype(str)
    cell = era.astype(str) + "|" + o
    size = cell.value_counts()
    small = size[size < floor * len(cell)].index
    for c in small:
        e = c.split("|")[0]
        big = size[[k for k in size.index if k.startswith(e + "|") and k not in small]].idxmax()
        cell[cell == c] = big
    return relabel(cell)


def ecoli_tilt(df, n_clients: int = 5) -> pd.Series:
    """Stress test: rank patients by their own E. coli resistance rate, cut into equal blocks
    (patients with no E. coli rows are placed at random). Uses outcomes, so it is partly circular."""
    e = df[df[ORG] == "ESCHERICHIA COLI"].groupby(PK)["label"].mean()
    rng = np.random.default_rng(SEED)
    pk = pd.Index(df[PK].unique())
    score = e.reindex(pk)
    score = score.fillna(pd.Series(rng.random(len(pk)), index=pk))
    score = score + 1e-6 * rng.random(len(pk))
    return pd.Series(pd.qcut(score.rank(method="first"), n_clients, labels=False).astype(int), index=pk,
                     name="hospital")


def rate_ranges(df, a):
    h = df[PK].map(a)
    out = {}
    for org, abx in PAIRS:
        m = (df[ORG] == org) & (df[ABX] == abx)
        g = df.loc[m].groupby(h[m])["label"].agg(["mean", "size"])
        g = g[g["size"] >= MIN_TESTS]
        key = f"{org.split()[0][:4]}-{abx[:4]}"
        out[key] = f"{100*g['mean'].min():.0f}-{100*g['mean'].max():.0f}% (n={len(g)})" if len(g) else "n/a"
    return out


def main():
    df = load_cohort_frame()
    strata = history_strata(df)
    rng = np.random.default_rng(SEED)
    pk = pd.Index(df[PK].unique())
    splits = {"random10": pd.Series(rng.integers(0, 10, len(pk)), index=pk),
              "org10": organism_community(df, 10), "louvain3": louvain_split(df, 3, SEED),
              "ward": ward_baseline(df), "specimen": specimen_baseline(df)}
    for n_eras in (2, 5):
        for m in (0.0, 0.1, 0.25, 0.5):
            splits[f"era{n_eras}x{10 // n_eras}_mix{m}"] = era_cluster_split(df, 10, n_eras, m, SEED)
    for n_eras in (2, 4):
        splits[f"era{n_eras}_x_specimen"] = composite(df, n_eras, specimen_baseline(df))
        splits[f"era{n_eras}_x_ward"] = composite(df, n_eras, ward_baseline(df))
    ac = antibiogram_conflict(df, 5)
    for m in (0.0, 0.25, 0.5, 0.75):
        splits[f"stress_conflict_mix{m}"] = mix(ac, m)
    et = ecoli_tilt(df)
    for m in (0.0, 0.25, 0.5):
        splits[f"stress_ecoli_tilt_mix{m}"] = mix(et, m)

    rows = []
    for name, a in splits.items():
        a = relabel(a)
        sizes = a.value_counts()
        raw = heterogeneity_vs_null(df, a, n_null=3, seed=SEED)
        hist = heterogeneity_vs_null(df, a, n_null=3, seed=SEED, strata=strata)
        rows.append({"split": name, "hospitals": len(sizes), "smallest_pct": round(100 * sizes.min() / len(a), 1),
                     "prior": round(raw["prior_shift"], 2), "ratio_raw": round(raw["concept_ratio"], 2),
                     "ratio_hist": round(hist["concept_ratio"], 2), **rate_ranges(df, a)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
