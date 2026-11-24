"""Notebook 22: does similarity-weighted aggregation help once the hospitals REALLY differ?
(pre-registered in docs/2026-11-02-shift-test-preregistration.md)

Two ways to make ARMD's hospitals differ on purpose, each a dial:

  concept   shift<k>: patients of the LATER era (group 1 of ``era_cluster_split``) get the labels of a fraction of the
            antibiotics inverted, so the two groups of 5 hospitals follow conflicting rules (IFCA-style label swap).
  panel     panel<k>: patients are split into two random groups of 5 hospitals; each group is only tested on its own
            antibiotic panel (a shared part + an exclusive part). Labels are untouched, the graphs differ.

Arms (budget-matched: 8 rounds x 4 epochs + the same 8-epoch fine-tune for every arm):
  fedavg, v3_topo (panel only), v3_ab, v3_ab_shuffled (control, run only if v3_ab wins), oracle (true group).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import _fmt_stat, _stat
from .budget import finetune_score
from .budget_runs import ERA_PER_GROUP, PART_B_ARMS, era_group
from .final_protocol import FT_EPOCHS

SHIFT_FRACTIONS = {"shift10": 0.10, "shift25": 0.25, "shift50": 0.50, "shift100": 1.00}   # antibiotics inverted
PANEL_SHARED = {"panel50": 0.50, "panel20": 0.20}       # share of antibiotics BOTH groups are tested on
EXPERIMENTS = list(SHIFT_FRACTIONS) + list(PANEL_SHARED)
ARMS = ("fedavg", "v3_topo", "v3_ab", "v3_ab_shuffled", "oracle")
DEFAULT_ARMS = {"concept": ("fedavg", "v3_ab", "oracle"), "panel": ("fedavg", "v3_topo", "v3_ab", "oracle")}


def kind(experiment: str) -> str:
    return "concept" if experiment in SHIFT_FRACTIONS else "panel"


def shifted_df(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """Copy of ``df`` with, for LATER-era patients, the label of a fixed random ``frac`` of the antibiotics inverted.
    Deterministic and independent of the run seed (the era comes from the first-culture time)."""
    from ..data_loader import ABX, PK
    from ..partition import patient_era
    out = df.copy()
    if frac <= 0:
        return out
    abx = np.array(sorted(out[ABX].unique()))
    chosen = np.random.default_rng(seed).choice(abx, size=max(1, int(round(frac * len(abx)))), replace=False)
    late = out[PK].map(patient_era(out, 2)).to_numpy() == 1
    flip = late & out[ABX].isin(chosen).to_numpy()
    out.loc[flip, "label"] = (1 - out.loc[flip, "label"]).astype("int8")
    return out


def _patient_group(pk: pd.Series) -> np.ndarray:
    """Deterministic 0/1 patient group (hash of the patient id): the same for every run seed."""
    return (pd.util.hash_pandas_object(pk, index=False).to_numpy() % 2).astype(int)


def panel_df(df: pd.DataFrame, shared: float, seed: int = 0) -> pd.DataFrame:
    """Rows of antibiotics outside a patient group's panel are dropped. Panel = ``shared`` of the antibiotics (both
    groups) + half of the rest each (exclusive). Patients with no row left disappear."""
    from ..data_loader import ABX, PK
    abx = np.array(sorted(df[ABX].unique()))
    order = np.random.default_rng(seed).permutation(abx)
    n_shared = max(1, int(round(shared * len(abx))))
    rest = order[n_shared:]
    panels = [set(order[:n_shared]) | set(rest[: len(rest) // 2]), set(order[:n_shared]) | set(rest[len(rest) // 2:])]
    grp = _patient_group(df[PK])
    keep = np.where(grp == 0, df[ABX].isin(panels[0]).to_numpy(), df[ABX].isin(panels[1]).to_numpy())
    return df[keep].reset_index(drop=True)


def panel_partition(df: pd.DataFrame, seed: int) -> pd.Series:
    """Group = hash of the patient id; hospital = group * 5 + random 0..4 (deterministic in seed)."""
    from ..data_loader import PK
    pk = pd.Series(sorted(df[PK].unique()))
    rng = np.random.default_rng(seed)
    return pd.Series(_patient_group(pk) * ERA_PER_GROUP + rng.integers(0, ERA_PER_GROUP, len(pk)),
                     index=pk.to_numpy(), name="hospital")


def concept_partition(df: pd.DataFrame, seed: int) -> pd.Series:
    from ..partition import era_cluster_split
    return era_cluster_split(df, n_clients=2 * ERA_PER_GROUP, n_eras=2, seed=seed)


def build(experiment: str, df: pd.DataFrame):
    """(experiment df, partition_fn(df, seed))."""
    if experiment in SHIFT_FRACTIONS:
        return shifted_df(df, SHIFT_FRACTIONS[experiment]), concept_partition
    if experiment in PANEL_SHARED:
        return panel_df(df, PANEL_SHARED[experiment]), panel_partition
    raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")


def arm_specs() -> dict:
    specs = {k: PART_B_ARMS[k] for k in ARMS if k in PART_B_ARMS}
    specs["fedavg"] = dict(strategy="fedavg", finetune_epochs=FT_EPOCHS, compute_local=False)
    return specs


PRIMARY = [("oracle+ft", "fedavg+ft"), ("v3_ab+ft", "fedavg+ft"), ("v3_ab+ft", "v3_ab_shuffled+ft"),
           ("v3_topo+ft", "fedavg+ft"), ("oracle+ft", "v3_ab+ft")]


def report(res: dict) -> str:
    """Unweighted mean-over-hospitals test F1 after the same fine-tune for each arm, paired by seed."""
    v = {f"{k}+ft": np.array([finetune_score(r)["mean"] for r in res[k]["runs"]]) for k in ARMS if k in res}
    if not v:
        return "no finished arm"
    out = [f"  {k:<20}{x.mean():.4f} +/- {x.std():.4f}" for k, x in v.items()]
    out += [_fmt_stat(f"{a} - {b}", _stat(v[a] - v[b]), v[a] - v[b]) for a, b in PRIMARY if a in v and b in v]
    return "\n".join(out)


def client_groups(experiment: str, df: pd.DataFrame, seed: int) -> list:
    """Group (0/1) of each client id exactly as ``run.run_fedavg`` numbers them (factorize = first appearance)."""
    _, part = build(experiment, df)
    codes, uniques = pd.factorize(part(df, seed))
    return [era_group(u) for u in uniques]


def weights(res: dict, groups_by_seed: dict) -> str:
    """Weight each personalized arm put on itself / its own group / the other group. ``groups_by_seed[seed]`` =
    ``client_groups(...)``."""
    lines = []
    for key in ("v3_topo", "v3_ab", "v3_ab_shuffled", "oracle"):
        if key not in res:
            continue
        s, w, b = [], [], []
        for seed, r in zip(res[key]["seeds"], res[key]["runs"]):
            W = np.asarray(r["agg_weights"])
            g = np.asarray(groups_by_seed[seed])
            eye = np.eye(len(g), dtype=bool)
            s.append(W[eye].mean()); w.append(W[(g[:, None] == g[None, :]) & ~eye].sum() / len(g))
            b.append(W[g[:, None] != g[None, :]].sum() / len(g))
        lines.append(f"  {key:<16} self {np.mean(s):.3f} | own group {np.mean(w):.3f} | other group {np.mean(b):.3f}")
    return "\n".join(lines)
