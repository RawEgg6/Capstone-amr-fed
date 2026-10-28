"""Torch-free analysis of the final-protocol results (notebooks 16 / 17).

``res`` is the JSON written by ``final_protocol.run_all``: arm key -> ``run_multiseed`` result.
Pseudo-arms: ``local`` (local-only, trained once per seed inside the FedAvg arm) and ``fedavg_ft``
(FedAvg followed by per-hospital fine-tuning).

PRIMARY metric = test macro-F1 with each hospital's own cutoff chosen on its own validation rows and the
round chosen on size-weighted validation (``phc_f1``). Secondary = one shared cutoff (``val_fed``).
"""
from __future__ import annotations

import numpy as np

NAMES = {
    "local": "Local-only", "fedavg": "FedAvg", "fedavg_ft": "FedAvg+fine-tune", "fedprox": "FedProx",
    "fedmedian": "FedMedian", "fedtrimmedavg": "FedTrimmedAvg", "uniform": "Uniform",
    "topo": "Topo v1", "topo_shuffled": "Topo v1-shuffled", "invsize": "Inv-size",
    "v3": "v3 (chosen T)", "v3_shuffled": "v3-shuffled",
}

# metric name -> (key in a federated arm's run, key for the 'local' pseudo-arm, key for 'fedavg_ft')
METRICS = {
    "phc_f1": ("phc_f1", "val_local", "ft_f1"),
    "phc_worst": ("phc_worst", "val_worst_local", "ft_worst"),
    "val_fed": ("val_fed", "val_local", None),          # shared cutoff (local-only is per-hospital anyway)
    "val_worst": ("val_worst_fed", "val_worst_local", None),
    "auroc": ("phc_auroc", "val_local_auroc", None),
}

# Pre-registered comparisons (docs/2026-10-28-final-protocol-preregistration.md)
PRIMARY = [("v3", "local"), ("v3", "fedavg_ft"), ("v3", "v3_shuffled")]
SECONDARY = [("v3", "fedavg"), ("fedavg", "local"), ("fedavg_ft", "local"), ("topo", "uniform"),
             ("uniform", "fedavg"), ("fedavg", "fedprox"), ("fedavg", "fedmedian")]


def arms(res: dict) -> list[str]:
    return [k for k in res if not k.startswith("_")]


def resolve(res: dict, arm: str) -> str:
    """'v3' / 'v3_shuffled' mean the chosen-temperature arm; everything else is a real key."""
    T = res.get("_meta", {}).get("chosen_T")
    if arm == "v3":
        return f"v3_T{T:g}" if T is not None else "v3"
    return arm


def vec(res: dict, arm: str, metric: str = "phc_f1") -> np.ndarray:
    """Per-seed values of ``metric`` for an arm (NaN where unavailable)."""
    k_fed, k_local, k_ft = METRICS[metric]
    if arm == "local":
        runs, key = res["fedavg"]["runs"], k_local
    elif arm == "fedavg_ft":
        runs, key = res["fedavg"]["runs"], k_ft
    else:
        runs, key = res[resolve(res, arm)]["runs"], k_fed
    if key is None:
        return np.full(len(runs), np.nan)
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in runs])


def have(res: dict, arm: str) -> bool:
    try:
        v = vec(res, arm)
    except (KeyError, TypeError):
        return False
    return bool(len(v)) and not np.isnan(v).all()


def paired(res: dict, a: str, b: str, metric: str = "phc_f1") -> dict:
    """Paired-by-seed difference a - b: mean, 95% t-interval, two-sided p, per-seed values."""
    from scipy import stats
    d = vec(res, a, metric) - vec(res, b, metric)
    n = len(d)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    p = float(stats.ttest_1samp(d, 0).pvalue) if n > 1 and sd > 0 else float("nan")
    return {"a": a, "b": b, "metric": metric, "mean": m, "lo": m - tcrit * se, "hi": m + tcrit * se,
            "p": p, "per_seed": d, "n_pos": int((d > 0).sum()), "n": n}


def fmt_paired(r: dict) -> str:
    return (f"{NAMES.get(r['a'], r['a']):>18} - {NAMES.get(r['b'], r['b']):<18} {r['mean']:+.4f}  "
            f"95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  p={r['p']:.2f}  {r['n_pos']}/{r['n']} seeds > 0  "
            f"{np.round(r['per_seed'], 4)}")


def ms(v: np.ndarray) -> str:
    v = v[~np.isnan(v)]
    return "-" if not len(v) else f"{v.mean():.4f} +/- {v.std():.4f}"


def summary_table(res: dict) -> str:
    """One line per arm: primary metric, shared-cutoff F1, worst hospital, AUROC, chosen round per seed."""
    order = ["local", "fedavg", "fedavg_ft", "v3", "v3_shuffled", "uniform", "topo", "topo_shuffled",
             "invsize", "fedprox", "fedmedian", "fedtrimmedavg"]
    lines = [f"{'method':<19}{'per-hosp cutoffs':<19}{'shared cutoff':<19}{'worst (per-hosp)':<19}{'AUROC':<19}rounds"]
    for a in order:
        if not have(res, a):
            continue
        rounds = ""
        if a not in ("local", "fedavg_ft"):
            rounds = [r["phc_round"] for r in res[resolve(res, a)]["runs"]]
        lines.append(f"{NAMES[a]:<19}{ms(vec(res, a, 'phc_f1')):<19}{ms(vec(res, a, 'val_fed')):<19}"
                     f"{ms(vec(res, a, 'phc_worst')):<19}{ms(vec(res, a, 'auroc')):<19}{rounds}")
    return "\n".join(lines)


def comparison_report(res: dict, metrics=("phc_f1", "phc_worst")) -> str:
    out = []
    for metric in metrics:
        out.append(f"--- {metric}: PRIMARY (pre-registered) ---")
        out += [fmt_paired(paired(res, a, b, metric)) for a, b in PRIMARY if have(res, a) and have(res, b)]
        out.append(f"--- {metric}: secondary ---")
        out += [fmt_paired(paired(res, a, b, metric)) for a, b in SECONDARY if have(res, a) and have(res, b)]
        out.append("")
    return "\n".join(out)


def verdict(res: dict) -> str:
    """The pre-registered win rule: v3 beats local-only, FedAvg+fine-tune AND v3-shuffled on the primary
    metric, each with a paired 95% interval that excludes zero."""
    rows = []
    for a, b in PRIMARY:
        if not (have(res, a) and have(res, b)):
            return "INCOMPLETE: need v3, local, fedavg_ft and v3_shuffled"
        r = paired(res, a, b, "phc_f1")
        rows.append((r, r["lo"] > 0))
    ok = all(w for _, w in rows)
    lines = [f"{'WIN' if w else 'no win'}: {NAMES[r['a']]} - {NAMES[r['b']]} = {r['mean']:+.4f} "
             f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" for r, w in rows]
    return ("PRE-REGISTERED VERDICT: " + ("v3 WINS on all three primary comparisons" if ok else
            "v3 does NOT win (at least one primary comparison's interval includes 0 or is negative)")
            + "\n  " + "\n  ".join(lines))


def per_hospital(res: dict, arm: str, rank_by_size: bool) -> tuple[np.ndarray, np.ndarray]:
    """(mean per-hospital test F1 over seeds, mean patients). With ``rank_by_size`` hospitals are
    sorted largest-first within each seed (Louvain: hospital ids are not stable across seeds)."""
    if arm == "local":
        runs, key = res["fedavg"]["runs"], "val_local_per_client"
    elif arm == "fedavg_ft":
        runs, key = res["fedavg"]["runs"], None
    else:
        runs, key = res[resolve(res, arm)]["runs"], "phc_per_client"
    rows, sizes = [], []
    for r in runs:
        v = r["finetune"]["per_client"] if key is None else r[key]
        s = np.asarray(r["sizes"], dtype=float)
        v = np.array([np.nan if x is None else x for x in v], dtype=float)
        if rank_by_size:
            o = np.argsort(-s, kind="stable")
            v, s = v[o], s[o]
        rows.append(v)
        sizes.append(s)
    return np.nanmean(rows, axis=0), np.mean(sizes, axis=0)


def per_hospital_table(res: dict, rank_by_size: bool) -> str:
    cols = [a for a in ("local", "fedavg", "fedavg_ft", "v3", "v3_shuffled", "uniform", "topo", "invsize",
                        "fedmedian", "fedprox") if have(res, a)]
    data = {a: per_hospital(res, a, rank_by_size) for a in cols}
    n = len(next(iter(data.values()))[0])
    head = f"{'hospital':<9}{'patients':>9}" + "".join(f"{NAMES[a][:12]:>14}" for a in cols)
    lines = [head]
    for h in range(n):
        size = next(iter(data.values()))[1][h]
        lines.append(f"{('#%d' % (h + 1)) if rank_by_size else ('H%d' % h):<9}{size:>9.0f}"
                     + "".join(f"{data[a][0][h]:>14.4f}" for a in cols))
    loc = data["local"][0]
    wins = {NAMES[a]: int((data[a][0] > loc).sum()) for a in cols if a != "local"}
    lines.append(f"hospitals beating local-only (of {n}): {wins}")
    return "\n".join(lines)


def weights_report(res: dict) -> str:
    """The aggregation weights each topology / v3 arm used (static per seed)."""
    out = []
    for a in ("topo", "v3", "v3_shuffled"):
        if a not in res and resolve(res, a) not in res:
            continue
        key = resolve(res, a)
        for r, s in zip(res[key]["runs"], res[key]["seeds"]):
            w = r.get("agg_weights")
            if w is None:
                continue
            w = np.array(w)
            if w.ndim == 2:
                out.append(f"{NAMES[a]} seed {s}: self-weights {np.round(np.diag(w), 3).tolist()} | "
                           f"mean off-diagonal {w[~np.eye(len(w), dtype=bool)].mean():.3f}")
            else:
                out.append(f"{NAMES[a]} seed {s}: weights {np.round(w, 3).tolist()}")
    return "\n".join(out)
