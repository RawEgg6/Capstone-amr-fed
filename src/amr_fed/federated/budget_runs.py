"""Notebook 21: arm lists, run loop and reports (pre-registered in docs/2026-10-29-budget-and-era-preregistration.md).

Part A — is FedAvg + fine-tune's lead just extra training?  (Louvain 3 hospitals, organism-10)
  long   FedAvg 24 rounds x 4 local epochs, local-only 104 epochs, then fine-tune 8 epochs   (one run per seed)
  ft8    FedAvg 8 rounds + fine-tune 8 epochs (= 40 local epochs; same as notebooks 16/17, rerun on THIS machine)
  Every smaller budget is read out of ``long`` by ``budget.fed_upto`` / ``budget.local_upto``.

Part B — can similarity help when hospitals really differ?  (``partition.era_cluster_split``: 10 hospitals in
  2 groups of 5, each group from one half of the timeline; local-only 40 epochs, every arm ends with the same
  8-epoch fine-tune, so all comparisons are budget-matched at 40 local epochs)
  fedavg         FedAvg (+ local-only and FedAvg + fine-tune)
  v3_topo        v3 with the current graph fingerprint, T = 0.5, + fine-tune
  v3_ab          v3 with the cross-antibiogram fingerprint (label_fingerprints), T = 0.5, + fine-tune
  v3_ab_shuffled v3_ab with fingerprints handed to the wrong hospitals (control)
  oracle         equal weights within the TRUE era group, + fine-tune (upper bound for any clustering)
  simft          SimFT tau = 0.5 + fine-tune (also logs the cross-loss matrices for the offline fingerprint test)

Torch/flwr are imported lazily, so the report functions work on a CPU-only machine.
"""
from __future__ import annotations

import numpy as np

from .analysis import _fmt_stat, _stat
from .budget import fed_upto, finetune_score, local_upto
from .final_protocol import FT_EPOCHS, run_seedwise

T_FIXED = 0.5
LONG_ROUNDS, LOCAL_EPOCHS = 24, 4
LONG_LOCAL = LONG_ROUNDS * LOCAL_EPOCHS + FT_EPOCHS      # 104: local-only gets what FedAvg + fine-tune gets

PART_A_ARMS = {
    "long": dict(strategy="fedavg", rounds=LONG_ROUNDS, local_only_epochs=LONG_LOCAL, finetune_epochs=FT_EPOCHS,
                 compute_local=True),
    "ft8": dict(strategy="fedavg", rounds=8, finetune_epochs=FT_EPOCHS, compute_local=False),
}

ERA_PER_GROUP = 5   # era_cluster_split(n_clients=10, n_eras=2): raw label c belongs to group c // 5


def era_group(raw_label) -> int:
    return int(raw_label) // ERA_PER_GROUP


PART_B_ARMS = {
    "fedavg": dict(strategy="fedavg", finetune_epochs=FT_EPOCHS, compute_local=True),
    "v3_topo": dict(strategy="personalized", personalized_T=T_FIXED, personalized_fp="topology",
                    finetune_epochs=FT_EPOCHS, compute_local=False),
    "v3_ab": dict(strategy="personalized", personalized_T=T_FIXED, personalized_fp="antibiogram",
                  finetune_epochs=FT_EPOCHS, compute_local=False),
    "v3_ab_shuffled": dict(strategy="personalized", personalized_T=T_FIXED, personalized_fp="antibiogram",
                           personalized_shuffle=True, finetune_epochs=FT_EPOCHS, compute_local=False),
    "oracle": dict(strategy="personalized", personalized_T=T_FIXED, personalized_fp="oracle",
                   personalized_group_of=era_group, finetune_epochs=FT_EPOCHS, compute_local=False),
    "simft": dict(strategy="simft", simft_tau=T_FIXED, finetune_epochs=FT_EPOCHS, compute_local=False),
}


def run_part(res: dict, arms: dict, *, prefix: str, kw: dict, run_arms=None, checkpoint=None,
             ray_shutdown=None) -> dict:
    """Run every arm of ``arms`` not yet in ``res`` (resume-safe, checkpointed seed by seed)."""
    from .run import run_multiseed
    seeds = tuple(kw["seeds"])
    for key, extra in arms.items():
        if run_arms is not None and key not in run_arms:
            continue

        def one(seed, key=key, extra=extra):
            if ray_shutdown:
                ray_shutdown()
            args = dict(kw, seeds=(seed,), compute_pooled=False)
            args.update(extra)
            return run_multiseed(label=f"{prefix}/{key}", **args)["runs"][0]
        run_seedwise(res, key, seeds, one, checkpoint)
    return res


# ---------------------------------------------------------------- Part A report
def _vals(runs, fn, metric="f1"):
    return np.array([fn(r)[metric] for r in runs])


def part_a_values(res: dict, metric: str = "f1") -> dict:
    """Per-seed values at each budget (``metric``: 'f1' size-weighted primary, 'mean' unweighted, 'worst')."""
    long, ft8 = res["long"]["runs"], res["ft8"]["runs"]
    return {
        "local@32": _vals(long, lambda r: local_upto(r, 32), metric),
        "local@40": _vals(long, lambda r: local_upto(r, 40), metric),
        "local@64": _vals(long, lambda r: local_upto(r, 64), metric),
        "local@104": _vals(long, lambda r: local_upto(r, LONG_LOCAL), metric),
        "fedavg@8 (32)": _vals(long, lambda r: fed_upto(r, 8), metric),
        "fedavg@10 (40)": _vals(long, lambda r: fed_upto(r, 10), metric),
        "fedavg@16 (64)": _vals(long, lambda r: fed_upto(r, 16), metric),
        "fedavg@24 (96)": _vals(long, lambda r: fed_upto(r, LONG_ROUNDS), metric),
        "fedavg@8+ft (40)": _vals(ft8, finetune_score, metric),
        "fedavg@24+ft (104)": _vals(long, finetune_score, metric),
    }


PART_A_PRIMARY = [("fedavg@8+ft (40)", "local@40"), ("fedavg@8+ft (40)", "fedavg@10 (40)"),
                  ("fedavg@24+ft (104)", "local@104"), ("fedavg@24+ft (104)", "fedavg@24 (96)")]
PART_A_SECONDARY = [("fedavg@10 (40)", "local@40"), ("fedavg@24 (96)", "local@104"),
                    ("local@104", "local@32"), ("fedavg@24 (96)", "fedavg@8 (32)")]


def part_a_report(res: dict) -> str:
    out = []
    for metric, title in (("f1", "size-weighted (primary)"), ("mean", "unweighted mean over hospitals"),
                          ("worst", "worst hospital")):
        v = part_a_values(res, metric)
        out.append(f"=== {title}  (number in brackets = local epochs per hospital) ===")
        out += [f"  {k:<22}{x.mean():.4f} +/- {x.std():.4f}" for k, x in v.items()]
        for title2, pairs in (("primary", PART_A_PRIMARY), ("secondary", PART_A_SECONDARY)):
            out.append(f"  -- {title2} (paired by seed) --")
            out += [_fmt_stat(f"{a} - {b}", _stat(v[a] - v[b]), v[a] - v[b]) for a, b in pairs]
        out.append("")
    long = res["long"]["runs"]
    out.append("selected FedAvg round (of 24) per seed: " + str([fed_upto(r, LONG_ROUNDS)["round"] for r in long]))
    out.append("local-only best epoch (of 104), all hospitals x seeds: "
               + str(sorted(e for r in long for e in local_upto(r, LONG_LOCAL)["epochs"])))
    chk = [round(fed_upto(a, 8)["f1"] - b["phc_f1"], 4) for a, b in zip(long, res["ft8"]["runs"])]
    out.append(f"determinism check, FedAvg@8 from 'long' minus 'ft8' run (should be ~0): {chk}")
    return "\n".join(out)


# ---------------------------------------------------------------- Part B report
def part_b_values(res: dict, metric: str = "mean") -> dict:
    """Per-seed values. PRIMARY metric = unweighted mean over hospitals (hospitals are equal-sized here, so it is
    nearly the size-weighted one). After-fine-tune scores for every arm; FedAvg / local without fine-tune too."""
    v = {}
    fa = res["fedavg"]["runs"]
    v["local-only (40)"] = _vals(fa, lambda r: local_upto(r, 40), metric)
    v["fedavg (32)"] = _vals(fa, lambda r: fed_upto(r, len(r["records"])), metric)
    for key in PART_B_ARMS:
        if key in res:
            v[f"{key}+ft"] = _vals(res[key]["runs"], finetune_score, metric)
    return v


PART_B_PRIMARY = [("v3_ab+ft", "fedavg+ft"), ("v3_ab+ft", "v3_ab_shuffled+ft"), ("oracle+ft", "fedavg+ft")]
PART_B_SECONDARY = [("v3_topo+ft", "fedavg+ft"), ("simft+ft", "fedavg+ft"), ("v3_ab+ft", "v3_topo+ft"),
                    ("oracle+ft", "v3_ab+ft"), ("fedavg+ft", "local-only (40)"), ("fedavg (32)", "local-only (40)")]


def part_b_report(res: dict) -> str:
    out = []
    for metric, title in (("mean", "unweighted mean over hospitals (primary)"), ("f1", "size-weighted"),
                          ("worst", "worst hospital")):
        v = part_b_values(res, metric)
        out.append(f"=== {title} ===")
        out += [f"  {k:<22}{x.mean():.4f} +/- {x.std():.4f}" for k, x in v.items()]
        for title2, pairs in (("primary", PART_B_PRIMARY), ("secondary", PART_B_SECONDARY)):
            out.append(f"  -- {title2} (paired by seed) --")
            out += [_fmt_stat(f"{a} - {b}", _stat(v[a] - v[b]), v[a] - v[b]) for a, b in pairs if a in v and b in v]
        out.append("")
    return "\n".join(out)


def era_split(df, seed: int):
    """The Part B partition exactly as ``run.run_fedavg`` builds it: (assignment with client ids 0..9,
    era group of each client id)."""
    import pandas as pd
    from ..partition import era_cluster_split
    raw = era_cluster_split(df, n_clients=2 * ERA_PER_GROUP, n_eras=2, seed=seed)
    codes, uniques = pd.factorize(raw)
    return pd.Series(codes, index=raw.index), [era_group(u) for u in uniques]


def era_train_frames(df, seed: int) -> list:
    """Every hospital's TRAIN rows (client-id order) for the offline fingerprint test."""
    from ..data_loader import PK
    from ..topology import train_frame
    assign, _ = era_split(df, seed)
    return [train_frame(df[df[PK].map(assign) == c], seed) for c in range(int(assign.max()) + 1)]


def part_b_weights(res: dict, groups_by_seed: dict) -> str:
    """How much weight each personalized arm put on same-era neighbours, other-era hospitals and itself."""
    lines = []
    for key in ("v3_topo", "v3_ab", "v3_ab_shuffled", "oracle"):
        if key not in res:
            continue
        within, between, self_w = [], [], []
        for seed, r in zip(res[key]["seeds"], res[key]["runs"]):
            W = np.asarray(r["agg_weights"])
            g = np.asarray(groups_by_seed[seed])
            eye = np.eye(len(g), dtype=bool)
            same = (g[:, None] == g[None, :]) & ~eye
            within.append(W[same].sum() / len(g))
            between.append(W[g[:, None] != g[None, :]].sum() / len(g))
            self_w.append(W[eye].mean())
        lines.append(f"  {key:<16} self {np.mean(self_w):.3f} | same-era neighbours {np.mean(within):.3f} | "
                     f"other era {np.mean(between):.3f}")
    return "\n".join(lines)


def offline_fingerprint_test(res: dict, frames_by_seed: dict) -> str:
    """Score every label fingerprint against the SimFT cross-loss matrices of this split (round 1 and all rounds).
    ``frames_by_seed[seed]`` = list of hospital TRAIN frames in client-id order."""
    from ..label_fingerprints import CANDIDATES, score_against, transfer_gap
    rows = []
    for seed, run in zip(res["simft"]["seeds"], res["simft"]["runs"]):
        frames = frames_by_seed[seed]
        G1 = transfer_gap(run["simft_rounds"][0]["L"])
        Gm = np.mean([transfer_gap(x["L"]) for x in run["simft_rounds"]], axis=0)
        for name, fn in CANDIDATES.items():
            D = fn(frames)
            rows.append((name, score_against(D, G1, n_perm=500)["pooled"], score_against(D, Gm, n_perm=500)["pooled"]))
    out = [f"{'fingerprint':<28}{'Spearman vs round-1 gap':<26}all-rounds gap"]
    for name in CANDIDATES:
        a = np.array([r[1] for r in rows if r[0] == name])
        b = np.array([r[2] for r in rows if r[0] == name])
        out.append(f"{name:<28}{a.mean():+.3f} +/- {a.std():.3f}{'':<10}{b.mean():+.3f} +/- {b.std():.3f}")
    return "\n".join(out)
