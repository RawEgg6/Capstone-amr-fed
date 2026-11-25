"""Low-data runs on the pair-rate split (split D, scripts/2026-11-24-pair-rate-split.py).

Pre-registration: docs/2026-11-25-pair-rate-lowdata-preregistration.md (written before any training run).
10 hospitals in 3 regions (hospital c is in region c % 3). Each hospital keeps a fraction of its train/validation patients
(same test patients). Levels 10%, 3%, 1%, 0.3%. Seeds 42-46 = training randomness only (the split is fixed).
Arms, all + the same 8-epoch fine-tune (8 rounds x 4 local epochs, hidden 128; local-only gets 40 epochs):
  fedavg                 (also trains local-only)
  v3_topo / v3_topo_shuffled   similarity weights from the graph-shape fingerprint, real and handed to the wrong hospitals
  v3_ab  / v3_ab_shuffled      same with the label (cross-antibiogram) fingerprint
  oracle                 equal weights inside the true region: the upper bound for any similarity method
Metric: unweighted mean over hospitals of test macro-F1 after fine-tuning (hospitals are equal-sized).
Results: results/amr_pairrate_f<per-mille>.json (resume-safe). Runs on the Mac CPU.
Use: headroom() first, then run(), then report().
"""
import functools
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from amr_fed.federated import budget_runs as BR  # noqa: E402
from amr_fed.federated import final_protocol as P  # noqa: E402
from amr_fed.federated.analysis import _fmt_stat, _stat  # noqa: E402
from amr_fed.federated.budget import fed_upto, finetune_score, local_upto  # noqa: E402

_spec = importlib.util.spec_from_file_location("pr", ROOT / "scripts" / "2026-11-24-pair-rate-split.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

FRACS, SEEDS, GATE_FRAC = (0.1, 0.03, 0.01, 0.003), (42, 43, 44, 45, 46), 0.03
RES = ROOT / "results"
FT = BR.FT_EPOCHS


def region_group(raw_label) -> int:
    return int(pr.REGION[int(raw_label)])


ARMS = {
    "fedavg": dict(BR.PART_B_ARMS["fedavg"]),
    "v3_topo": dict(BR.PART_B_ARMS["v3_topo"]),
    "v3_topo_shuffled": dict(BR.PART_B_ARMS["v3_topo"], personalized_shuffle=True),
    "v3_ab": dict(BR.PART_B_ARMS["v3_ab"]),
    "v3_ab_shuffled": dict(BR.PART_B_ARMS["v3_ab_shuffled"]),
    "oracle": dict(strategy="personalized", personalized_T=BR.T_FIXED, personalized_fp="oracle",
                   personalized_group_of=region_group, finetune_epochs=FT, compute_local=False),
}


def _random_group(perm, raw_label) -> int:
    return int(pr.REGION[perm[int(raw_label)]])


def _random_perms():
    """Three random re-shufflings of the hospitals into groups of the same sizes as the true regions, each sharing at most
    1/3 of its same-group pairs with the true regions (so they are not the regions in disguise)."""
    true = pr.REGION[:, None] == pr.REGION[None, :]
    out, seed = [], 0
    while len(out) < 3:
        seed += 1
        perm = np.random.default_rng(seed).permutation(pr.N_HOSP)
        g = pr.REGION[perm]
        same = (g[:, None] == g[None, :]) & ~np.eye(pr.N_HOSP, dtype=bool)
        if (same & true).sum() / same.sum() <= 1 / 3:
            out.append(perm.tolist())
    return out


# Iteration 2026-11-25 (after the gate failed): oracle with RANDOM groups of the same sizes. If it scores like the true-region oracle,
# the oracle's gap to FedAvg is just "fewer hospitals in the average", not the regions carrying (or lacking) information.
RANDOM_ARMS = {f"random_{n}": dict(strategy="personalized", personalized_T=BR.T_FIXED, personalized_fp="oracle",
                                    personalized_group_of=functools.partial(_random_group, perm),
                                    finetune_epochs=FT, compute_local=False)
               for n, perm in zip("abc", _random_perms())}
ALL_ARMS = {**ARMS, **RANDOM_ARMS}
COMPARE = [("fedavg+ft", "local-only (40)"), ("oracle+ft", "fedavg+ft"),
           ("v3_topo+ft", "fedavg+ft"), ("v3_topo+ft", "v3_topo_shuffled+ft"),
           ("v3_ab+ft", "fedavg+ft"), ("v3_ab+ft", "v3_ab_shuffled+ft")]


def path(frac):
    return RES / f"amr_pairrate_f{int(round(frac * 1000)):03d}.json"


_DF = {}


def data():
    """The cohort with the pair-rate shift applied (built once per session)."""
    if "df" not in _DF:
        from amr_fed.data_loader import load_cohort_frame
        _DF["df"] = pr.make_df(load_cohort_frame())[0]
    return _DF["df"]


def run(fracs=None, arms=None):
    """Run every (level, arm) not yet saved. Safe to stop and re-run."""
    import ray
    os.environ.pop("AMR_RNG_PROBE", None)   # SIGFPE guard, see AGENTS.md
    df = data()
    RES.mkdir(exist_ok=True)
    chosen = {k: ALL_ARMS[k] for k in (arms or ARMS)}
    for frac in (fracs or FRACS):
        p = path(frac)
        res = json.load(open(p)) if p.exists() else {}
        kw = dict(seeds=SEEDS, df=df, rounds=8, local_epochs=BR.LOCAL_EPOCHS, local_only_epochs=40, hidden=128,
                  n_clients=pr.N_HOSP, train_frac=frac, partition_fn=pr.partition_fn)
        for key in chosen:
            try:
                res = BR.run_part(res, chosen, prefix=f"pairrate/{frac}", kw=kw, run_arms=[key],
                                  checkpoint=lambda r, p=p: P.save_json(r, p), ray_shutdown=ray.shutdown)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {frac} {key}: {type(e).__name__}: {e}", flush=True)
        P.save_json(res, p)
        print(f"DONE {frac}", flush=True)


def values(res, metric="mean"):
    """Per-seed score of every arm that has been run, after fine-tuning (local-only and plain FedAvg without)."""
    v = {}
    if "fedavg" in res:
        fa = res["fedavg"]["runs"]
        v["local-only (40)"] = np.array([local_upto(r, 40)[metric] for r in fa])
        v["fedavg (32)"] = np.array([fed_upto(r, len(r["records"]))[metric] for r in fa])
    for key in ALL_ARMS:
        if key in res:
            v[f"{key}+ft"] = np.array([finetune_score(r)[metric] for r in res[key]["runs"]])
    return v


def headroom():
    """Pre-registered gate at 3%: does knowing the true regions help at all? Runs fedavg and oracle only."""
    run([GATE_FRAC], arms=["fedavg", "oracle"])
    v = values(json.load(open(path(GATE_FRAC))))
    d = v["oracle+ft"] - v["fedavg+ft"]
    st = _stat(d)
    print(_fmt_stat("oracle+ft - fedavg+ft", st, d))
    ok = st["lo"] > 0
    print("GATE PASSED: the oracle beats FedAvg + fine-tune, similarity weighting has room to help." if ok else
          "GATE FAILED: even the true regions do not beat FedAvg + fine-tune. Stop; report a null result.")
    return ok


def random_groups_test():
    """At the gate level: oracle (true regions) vs oracle with random groups of the same sizes vs FedAvg + fine-tune."""
    run([GATE_FRAC], arms=list(RANDOM_ARMS))
    v = values(json.load(open(path(GATE_FRAC))))
    rnd = np.mean([v[f"{k}+ft"] for k in RANDOM_ARMS], axis=0)
    print(f"perms used: {_random_perms()}")
    for k in ("local-only (40)", "fedavg+ft", "oracle+ft", *[f"{k}+ft" for k in RANDOM_ARMS]):
        print(f"  {k:<20}{v[k].mean():.4f} +/- {v[k].std():.4f}  {np.round(v[k], 4)}")
    print(f"  random groups, mean of 3  {rnd.mean():.4f}")
    for name, d in (("oracle(true) - random groups", v["oracle+ft"] - rnd), ("random groups - fedavg+ft", rnd - v["fedavg+ft"]),
                    ("oracle(true) - fedavg+ft", v["oracle+ft"] - v["fedavg+ft"])):
        print(_fmt_stat(name, _stat(d), d))


def report():
    for frac in FRACS:
        p = path(frac)
        if not p.exists():
            continue
        v = values(json.load(open(p)))
        if not v:
            continue
        print(f"######## pair-rate split, {frac * 100:g}% of each hospital's data (unweighted mean F1 over hospitals) ########")
        for k, x in v.items():
            print(f"  {k:<24}{x.mean():.4f} +/- {x.std():.4f}   per seed {np.round(x, 4)}")
        for a, b in COMPARE:
            if a in v and b in v and len(v[a]) == len(v[b]):
                d = v[a] - v[b]
                print(_fmt_stat(f"{a} - {b}", _stat(d), d))
        print()


def gaps():
    """{arm: {fraction: per-seed gain over local-only}} for the figure."""
    out = {}
    for frac in FRACS:
        p = path(frac)
        if not p.exists():
            continue
        v = values(json.load(open(p)))
        if "local-only (40)" not in v:
            continue
        for k, x in v.items():
            if k != "local-only (40)":
                out.setdefault(k, {})[frac] = x - v["local-only (40)"]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "report":
        report()
    elif a and a[0] == "random":
        random_groups_test()
    elif a and a[0] == "headroom":
        headroom()
    else:
        run([float(x) for x in a[1].split(",")] if len(a) > 1 else None)
