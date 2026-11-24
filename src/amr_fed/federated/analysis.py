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


# ---------------------------------------------------------------- notebook 18: low-data curve + new hospital
def _stat(d: np.ndarray) -> dict:
    """Paired summary of per-seed differences ``d``: mean, 95% t-interval, two-sided p, #seeds > 0."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    p = float(stats.ttest_1samp(d, 0).pvalue) if n > 1 and sd > 0 else float("nan")
    return {"mean": m, "lo": m - t * se, "hi": m + t * se, "p": p, "n_pos": int((d > 0).sum()), "n": n}


def _fmt_stat(label: str, st: dict, per_seed=None) -> str:
    tail = f"  {np.round(per_seed, 4)}" if per_seed is not None else ""
    return (f"  {label:<28}{st['mean']:+.4f}  95% CI [{st['lo']:+.4f}, {st['hi']:+.4f}]  p={st['p']:.2f}  "
            f"{st['n_pos']}/{st['n']} seeds > 0{tail}")


def scarcity_values(res: dict, key: str) -> dict:
    """Per-seed values at one data level. PRIMARY = UNWEIGHTED mean over hospitals of each hospital's test
    F1 (every hospital at its own validation-chosen cutoff); also the size-weighted version."""
    runs = res[key]["runs"]
    pc = lambda arr: np.array([np.nan if v is None else v for v in arr], dtype=float)
    local = [pc(r["val_local_per_client"]) for r in runs]
    fed = [pc(r["phc_per_client"]) for r in runs]
    ft = [pc(r["finetune"]["per_client"]) for r in runs]
    w = [np.array([t["test"] for t in r["n_triples"]], dtype=float) for r in runs]
    mean = lambda L: np.array([np.nanmean(x) for x in L])
    wmean = lambda L: np.array([np.nansum(x * ww) / ww[~np.isnan(x)].sum() for x, ww in zip(L, w)])
    return {"local": mean(local), "fedavg": mean(fed), "fedavg_ft": mean(ft),
            "local_w": wmean(local), "fedavg_w": wmean(fed), "fedavg_ft_w": wmean(ft),
            "n_hospitals": len(local[0]),
            "mean_train_triples": float(np.mean([np.mean([t["train"] for t in r["n_triples"]]) for r in runs]))}


def scarcity_report(res: dict) -> str:
    keys = sorted((k for k in res if k.startswith("scarcity_f")), key=lambda k: -float(k[len("scarcity_f"):]))
    if not keys:
        return "no scarcity results yet"
    lines = [f"{'kept':<7}{'train rows/hosp':<17}{'local-only':<20}{'FedAvg':<20}{'FedAvg+fine-tune':<20}"
             f"(primary: unweighted mean over hospitals, test F1 at own cutoff)"]
    for k in keys:
        v = scarcity_values(res, k)
        lines.append(f"{float(k[len('scarcity_f'):]):<7g}{v['mean_train_triples']:<17.0f}{ms(v['local']):<20}"
                     f"{ms(v['fedavg']):<20}{ms(v['fedavg_ft']):<20}")
    lines.append("\npaired by seed (primary metric):")
    for k in keys:
        v = scarcity_values(res, k)
        f = float(k[len("scarcity_f"):])
        lines.append(f" data kept = {f:g}")
        for a, b, name in (("fedavg", "local", "FedAvg - local-only"), ("fedavg_ft", "local", "FedAvg+fine-tune - local-only"),
                           ("fedavg_ft", "fedavg", "FedAvg+fine-tune - FedAvg")):
            d = v[a] - v[b]
            lines.append(_fmt_stat(name, _stat(d), d))
    lines.append("\nsame comparisons, size-weighted over hospitals:")
    for k in keys:
        v = scarcity_values(res, k)
        f = float(k[len("scarcity_f"):])
        lines.append(f" data kept = {f:g}: " + " | ".join(
            f"{name} {np.mean(v[a + '_w'] - v[b + '_w']):+.4f}"
            for a, b, name in (("fedavg", "local", "FedAvg-local"), ("fedavg_ft", "local", "FT-local"))))
    return "\n".join(lines)


def newsite_report(res: dict) -> str:
    out = []
    for rule in ("median", "smallest"):
        key = f"newsite_{rule}"
        if key not in res:
            continue
        runs = res[key]["runs"]
        ks = list(runs[0]["ks"])
        out.append(f"--- held-out hospital = {rule} (sizes by seed: {[r['holdout_patients'] for r in runs]} patients) ---")
        out.append(f"{'labelled patients':<19}{'zero-shot':<20}{'fed + fine-tune':<20}{'scratch (local)':<20}")
        for k in ks:
            col = lambda m: np.array([r["ks"][k][m]["f1"] for r in runs])
            n = np.mean([r["ks"][k]["n_train_patients"] for r in runs])
            out.append(f"{k + ' (~' + format(n, '.0f') + ')':<19}{ms(col('zero_shot')):<20}{ms(col('finetune')):<20}{ms(col('scratch')):<20}")
        out.append("paired by seed:")
        for k in ks:
            col = lambda m: np.array([r["ks"][k][m]["f1"] for r in runs])
            for a, b, name in (("finetune", "scratch", "fine-tune - scratch"), ("zero_shot", "scratch", "zero-shot - scratch")):
                d = col(a) - col(b)
                out.append(_fmt_stat(f"k={k}: {name}", _stat(d), d))
        out.append("")
    return "\n".join(out) if out else "no new-hospital results yet"


# ---------------------------------------------------------------- notebook 19: SimFT
def _pc(arr) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in arr], dtype=float)


def simft_arm_runs(res: dict, arm: str):
    """(runs, how) for a pseudo-arm. SimFT arms are scored after the final fine-tune, ``*_noft`` before it."""
    tau = res.get("_meta", {}).get("chosen_tau")
    if arm in ("local", "fedavg", "fedavg_ft"):
        return res["fedavg"]["runs"]
    if arm in ("simft", "simft_noft"):
        return res[f"simft_t{tau:g}"]["runs"]
    if arm in ("simft_flat", "simft_flat_noft"):
        return res["simft_flat"]["runs"]
    raise KeyError(arm)


def simft_values(res: dict, arm: str) -> dict:
    """Per-seed values for an arm: PRIMARY ``mean`` = UNWEIGHTED mean over hospitals of test macro-F1 (every
    hospital at its own validation-chosen cutoff); also size-weighted ``wmean`` and ``worst`` hospital."""
    runs = simft_arm_runs(res, arm)
    if arm == "local":
        per = [_pc(r["val_local_per_client"]) for r in runs]
    elif arm in ("fedavg", "simft_noft", "simft_flat_noft"):
        per = [_pc(r["phc_per_client"]) for r in runs]
    else:                                              # fedavg_ft, simft, simft_flat: after the final fine-tune
        per = [_pc(r["finetune"]["per_client"]) for r in runs]
    w = [np.array([t["test"] for t in r["n_triples"]], dtype=float) for r in runs]
    return {"mean": np.array([np.nanmean(x) for x in per]),
            "wmean": np.array([np.nansum(x * ww) / ww[~np.isnan(x)].sum() for x, ww in zip(per, w)]),
            "worst": np.array([np.nanmin(x) for x in per])}


SIMFT_PRIMARY = [("simft", "fedavg_ft"), ("simft", "local"), ("simft", "simft_flat")]
SIMFT_SECONDARY = [("simft", "fedavg"), ("simft_noft", "fedavg"), ("simft_flat", "fedavg_ft"),
                   ("fedavg_ft", "local"), ("fedavg", "local")]
SIMFT_NAMES = {"local": "Local-only", "fedavg": "FedAvg", "fedavg_ft": "FedAvg+fine-tune", "simft": "SimFT+fine-tune",
               "simft_flat": "SimFT-flat+fine-tune", "simft_noft": "SimFT (no fine-tune)",
               "simft_flat_noft": "SimFT-flat (no fine-tune)"}


def simft_report(res: dict) -> str:
    arms = ["local", "fedavg", "fedavg_ft", "simft_noft", "simft", "simft_flat_noft", "simft_flat"]
    avail = []
    for a in arms:
        try:
            simft_values(res, a)
            avail.append(a)
        except (KeyError, TypeError):
            pass
    tau = res.get("_meta", {}).get("chosen_tau")
    lines = [f"SimFT temperature chosen on validation half B: {tau} | scores {res.get('_meta', {}).get('tau_val_scores')}",
             f"{'method':<27}{'mean over hospitals':<22}{'size-weighted':<22}{'worst hospital':<22}"]
    for a in avail:
        v = simft_values(res, a)
        lines.append(f"{SIMFT_NAMES[a]:<27}{ms(v['mean']):<22}{ms(v['wmean']):<22}{ms(v['worst']):<22}")

    def block(title, pairs):
        out = [f"--- {title} ---"]
        for a, b in pairs:
            if a in avail and b in avail:
                for metric, mname in (("mean", "mean over hospitals"), ("wmean", "size-weighted"), ("worst", "worst hospital")):
                    d = simft_values(res, a)[metric] - simft_values(res, b)[metric]
                    out.append(_fmt_stat(f"{SIMFT_NAMES[a][:17]} - {SIMFT_NAMES[b][:17]} [{mname[:8]}]", _stat(d), d))
        return out

    lines += [""] + block("PRIMARY (pre-registered; first metric of each triple is the primary one)", SIMFT_PRIMARY)
    lines += [""] + block("secondary", SIMFT_SECONDARY)
    if all(a in avail for a in ("simft", "fedavg_ft", "local", "simft_flat")):
        rows = [(a, b, _stat(simft_values(res, a)["mean"] - simft_values(res, b)["mean"])) for a, b in SIMFT_PRIMARY]
        ok = all(st["lo"] > 0 for _, _, st in rows)
        lines += ["", "PRE-REGISTERED VERDICT: " + ("SimFT WINS (all three primary intervals above zero)" if ok else
                                                  "SimFT does NOT win")]
        lines += [f"  {'WIN' if st['lo'] > 0 else 'no win'}: {SIMFT_NAMES[a]} - {SIMFT_NAMES[b]} = {st['mean']:+.4f} "
                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}]" for a, b, st in rows]
    return "\n".join(lines)


def simft_weights_report(res: dict) -> str:
    """Mean mixing matrix at the last round and how much each hospital keeps of itself, per seed."""
    out = []
    for key in [k for k in res if k.startswith("simft_t") or k == "simft_flat"]:
        for r, s in zip(res[key]["runs"], res[key]["seeds"]):
            rounds = r.get("simft_rounds") or []
            if not rounds:
                continue
            W = np.array(rounds[-1]["W"])
            out.append(f"{key} seed {s}: self-weight {np.round(np.diag(W), 2).tolist()} | most-used neighbour per hospital "
                       f"{[int(np.argmax(np.where(np.eye(len(W), dtype=bool)[i], -1, W[i]))) for i in range(len(W))]}")
    return "\n".join(out)
