"""The final-protocol arm list and run loop shared by notebooks 16 (Louvain) and 17 (organism, 10).

Imported lazily (``run`` needs torch + flwr) so the analysis half can be used without them.

Arms (9 federated runs per seed): FedAvg (+ local-only, pooled and a fine-tune baseline, computed once
per seed), v3 at three temperatures, v3-shuffled at the temperature chosen on VALIDATION, Uniform,
Topo v1, FedProx, FedMedian. Only the FedAvg arm trains local-only: every other arm is compared with it.

Deliberately NOT run (answered already, would not change a conclusion): Topo-shuffled and Inverse-size
(notebook 15: topo ties both; v3-shuffled now carries the "does the fingerprint matter" test) and
FedTrimmedAvg (equals Uniform on 3 hospitals, within 0.0005 of it on 10). ``include_trimmed=True`` re-adds it.
"""
from __future__ import annotations

import json

import numpy as np

T_GRID = (0.25, 0.5, 1.0)        # v3 similarity temperatures; the best on validation is used for the control
FT_EPOCHS = 8                    # fine-tune baseline: local epochs at each hospital after FedAvg
INF = float("inf")


def v3_key(T: float) -> str:
    return f"v3_T{T:g}"


def stage1_specs(prefix: str, include_trimmed: bool) -> list[tuple[str, str, dict]]:
    """(key, label, run_multiseed kwargs) in run order, everything except v3_shuffled."""
    topo = dict(strategy="topology", topo_mode="distinctiveness")
    specs = [
        ("fedavg", f"{prefix}/FedAvg", dict(strategy="fedavg", compute_local=True, compute_pooled=True,
                                            finetune_epochs=FT_EPOCHS, save_assignment=True)),
    ]
    specs += [(v3_key(T), f"{prefix}/v3 (T={T:g})", dict(strategy="personalized", personalized_T=T))
              for T in T_GRID]
    specs += [
        ("uniform", f"{prefix}/UNIFORM (T=inf control)", dict(topo, topo_temperature=INF)),
        ("topo", f"{prefix}/TOPO v1 (T=1)", dict(topo, topo_temperature=1.0)),
        ("fedprox", f"{prefix}/FedProx", dict(strategy="fedprox")),
        ("fedmedian", f"{prefix}/FedMedian", dict(strategy="fedmedian")),
    ]
    if include_trimmed:
        specs.append(("fedtrimmedavg", f"{prefix}/FedTrimmedAvg", dict(strategy="fedtrimmedavg")))
    return specs


def choose_temperature(res: dict) -> tuple[float, dict]:
    """T with the best mean VALIDATION score (per-hospital cutoffs) across seeds. Test is not used.
    Returns (T, {T: mean val score}); ties go to the smaller T."""
    scores = {}
    for T in T_GRID:
        arm = res.get(v3_key(T))
        if arm is None:
            raise KeyError(f"{v3_key(T)} not finished yet")
        scores[T] = float(np.mean([r["phc_val_score"] for r in arm["runs"]]))
    best = max(sorted(scores), key=lambda t: scores[t])
    return best, scores


def run_seedwise(res: dict, key: str, seeds, one_seed, checkpoint=None) -> None:
    """Run ``one_seed(seed) -> dict`` for every seed of arm ``key`` (resume-safe). Each finished seed is
    checkpointed under ``res['_partial']``, so a Colab disconnect costs at most the seed that was running;
    when all seeds are done the arm is stored as ``{"seeds": [...], "runs": [...]}``."""
    if key in res:
        print(f"skip {key}: already saved")
        return
    part = res.setdefault("_partial", {}).setdefault(key, {})
    for s in seeds:
        if str(s) in part:
            print(f"skip {key} seed {s}: already saved")
            continue
        part[str(s)] = one_seed(s)
        if checkpoint:
            checkpoint(res)
    res[key] = {"seeds": list(seeds), "runs": [part[str(s)] for s in seeds]}
    del res["_partial"][key]
    if checkpoint:
        checkpoint(res)
    print(f"checkpoint saved after {key}")


def run_all(res: dict, *, prefix: str, include_trimmed: bool, kw: dict, run_arms=None, checkpoint=None,
            ray_shutdown=None) -> dict:
    """Run every arm that is not already in ``res`` (resume-safe). ``checkpoint(res)`` is called
    after each arm; ``run_arms`` restricts to a subset of keys (to split work across accounts)."""
    from .run import run_multiseed
    specs = stage1_specs(prefix, include_trimmed)
    want = lambda k: run_arms is None or k in run_arms
    res.setdefault("_meta", {})

    seeds = tuple(kw["seeds"])

    def run_one(key, label, extra):
        def one(seed):
            if ray_shutdown:
                ray_shutdown()
            return run_multiseed(label=label, **dict(kw, seeds=(seed,)), **extra)["runs"][0]
        run_seedwise(res, key, seeds, one, checkpoint)

    for key, label, extra in specs:
        if want(key):
            run_one(key, label, dict({"compute_local": False, "compute_pooled": False}, **extra))

    if want("v3_shuffled"):
        if all(v3_key(T) in res for T in T_GRID):
            T, scores = choose_temperature(res)
            res["_meta"].update(chosen_T=T, T_val_scores=scores)
            print(f"v3 temperature chosen on validation: T={T:g}  (mean val F1 by T: "
                  f"{ {t: round(v, 4) for t, v in scores.items()} })")
            run_one("v3_shuffled", f"{prefix}/v3-SHUFFLED (T={T:g})",
                    dict(strategy="personalized", personalized_T=T, personalized_shuffle=True,
                         compute_local=False, compute_pooled=False))
        else:
            print("v3_shuffled waits for all three v3 temperatures (run them, or merge the files first)")
    return res


def save_json(res: dict, *paths) -> None:
    for p in paths:
        if p is None:
            continue
        with open(p, "w") as f:
            json.dump(res, f, default=str)


def merge_results(paths) -> dict:
    """Merge result files written by several sessions: each arm is taken from the first file that has
    it, and ``_meta`` entries are combined (so ``chosen_T`` survives). Arms must not overlap
    silently, so a duplicated arm is reported."""
    merged: dict = {"_meta": {}}
    for p in paths:
        with open(p) as f:
            part = json.load(f)
        for k, v in part.items():
            if k == "_meta":
                for mk, mv in v.items():
                    merged["_meta"].setdefault(mk, mv)
            elif k == "_partial":
                if any(v.values()):
                    print(f"WARNING: {p} has unfinished arms {[a for a, x in v.items() if x]} - finish them first")
            elif k in merged:
                print(f"note: arm {k!r} already present, keeping the first copy (also in {p})")
            else:
                merged[k] = v
    return merged
