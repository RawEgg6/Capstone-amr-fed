"""Rate-target split: hospitals that differ on key organism-drug pairs by a chosen amount, plus a like-for-like
table (max/min rate ratio and odds ratio per pair) for every split.

Reference points from published papers (read in full, see docs):
  Gaza, two hospitals (El Aila & Abu El Aish, BMC Microbiol 2025): E. coli ceftriaxone 41.2% vs 64.7%
    (ratio 1.57, odds ratio 2.6), ciprofloxacin 27.5% vs 57.0% (ratio 2.07, odds ratio 3.5).
  Washington State, 9 regions (Fenelon et al., Microorganisms 2024): ceftriaxone adjusted odds ratios vs the
    reference region 0.20 to 0.78 (spread 3.9), ciprofloxacin 0.64 to 0.81 (spread 1.3).

Run: ARMD_DIR=<path to ARMD> python scripts/2026-11-22-rate-target-splits.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs", ROOT / "scripts" / "2026-11-22-realistic-splits.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

from amr_fed import config  # noqa: E402
from amr_fed.data_loader import PK, load_cohort_frame  # noqa: E402
from amr_fed.partition import (era_cluster_split, louvain_split, organism_community,  # noqa: E402
                               specimen_baseline, ward_baseline)
from amr_fed.split_diagnostics import heterogeneity_vs_null, history_strata  # noqa: E402

ORG, ABX = config.COLUMNS["organism"], config.COLUMNS["antibiotic"]
PAIRS, MIN_TESTS, SEED = rs.PAIRS, rs.MIN_TESTS, rs.SEED
OUT = ROOT / "results" / "rate_target_splits.csv"


def rate_target_split(df, n_hospitals: int = 10, sigma: float = 0.5, seed: int = SEED) -> pd.Series:
    """Stress test: each patient goes to hospital h with probability softmax_h(sum_p beta[h,p] * r[i,p]) where
    r[i,p] = +0.5 if the patient's mean result on pair p is resistant, -0.5 if susceptible, 0 if never tested.
    beta[h,p] ~ N(0, sigma), centred over hospitals, drawn independently per pair, so hospitals differ in
    DIFFERENT directions on different pairs. Uses outcomes to assign patients (partly circular, say so)."""
    rng = np.random.default_rng(seed)
    pk = pd.Index(df[PK].unique())
    r = np.zeros((len(pk), len(PAIRS)))
    for j, (org, abx) in enumerate(PAIRS):
        m = (df[ORG] == org) & (df[ABX] == abx)
        s = df.loc[m].groupby(PK)["label"].mean().reindex(pk)
        r[:, j] = np.where(s.isna(), 0.0, np.where(s >= 0.5, 0.5, -0.5))
    beta = rng.normal(0, sigma, (n_hospitals, len(PAIRS)))
    beta -= beta.mean(axis=0, keepdims=True)
    logits = r @ beta.T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random(len(pk))[:, None]
    h = (p.cumsum(axis=1) < u).sum(axis=1).clip(0, n_hospitals - 1)
    return pd.Series(h, index=pk, name="hospital")


def spread(df, a):
    h = df[PK].map(a)
    out = {}
    for org, abx in PAIRS:
        m = (df[ORG] == org) & (df[ABX] == abx)
        g = df.loc[m].groupby(h[m])["label"].agg(["mean", "size"])
        g = g[g["size"] >= MIN_TESTS]
        lo, hi = g["mean"].min(), g["mean"].max()
        key = f"{org.split()[0][:4]}-{abx[:4]}"
        out[key + "_range"] = f"{100*lo:.0f}-{100*hi:.0f}%"
        out[key + "_ratio"] = round(hi / lo, 2)
        out[key + "_OR"] = round((hi / (1 - hi)) / (lo / (1 - lo)), 2)
    return out


def main():
    df = load_cohort_frame()
    strata = history_strata(df)
    rng = np.random.default_rng(SEED)
    pk = pd.Index(df[PK].unique())
    splits = {"random10": pd.Series(rng.integers(0, 10, len(pk)), index=pk),
              "org10": organism_community(df, 10), "louvain3": louvain_split(df, 3, SEED),
              "ward": ward_baseline(df), "specimen": specimen_baseline(df),
              "era2x5": era_cluster_split(df, 10, 2, 0.0, SEED), "era5x2": era_cluster_split(df, 10, 5, 0.0, SEED),
              "era2_x_specimen": rs.composite(df, 2, specimen_baseline(df)),
              "era4_x_ward": rs.composite(df, 4, ward_baseline(df))}
    new = {}
    for s in (0.3, 0.5, 0.8, 1.2):
        new[f"target_sigma{s}"] = rate_target_split(df, 10, s)
    splits.update(new)
    rows = []
    for name, a in splits.items():
        a = rs.relabel(a)
        row = {"split": name, "hospitals": a.nunique(), "smallest_pct": round(100 * a.value_counts().min() / len(a), 1),
               **spread(df, a)}
        if name in new or name in ("era5x2", "era4_x_ward"):
            hh = heterogeneity_vs_null(df, a, n_null=3, seed=SEED, strata=strata)
            row["ratio_hist"] = round(hh["concept_ratio"], 2)
        rows.append(row)
        print(row, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 300)
    print(res.to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
