"""Run loops for notebook 18: the low-data curve (A) and the new-hospital test (B). Resume-safe, seed by seed."""
from __future__ import annotations

from .final_protocol import FT_EPOCHS, run_seedwise

FRACS = (1.0, 0.25, 0.10, 0.03)     # fraction of every hospital's train/validation patients kept
RULES = ("median", "smallest")      # which hospital is held out (regimes.holdout_hospital)


def frac_key(f: float) -> str:
    return f"scarcity_f{f:g}"


def run_scarcity(res: dict, *, kw: dict, fracs=FRACS, checkpoint=None, ray_shutdown=None) -> dict:
    """Part A. FedAvg (+ local-only and the fine-tune baseline, same seed) at each data level. ``kw`` holds
    seeds, df, partition_fn, n_clients, rounds, local_epochs, local_only_epochs, hidden."""
    from .run import run_multiseed
    res.setdefault("_meta", {})
    seeds = tuple(kw["seeds"])
    for f in fracs:
        def one(seed, f=f):
            if ray_shutdown:
                ray_shutdown()
            return run_multiseed(label=f"scarcity f={f:g}", **dict(kw, seeds=(seed,)), strategy="fedavg",
                                 compute_local=True, compute_pooled=False, finetune_epochs=FT_EPOCHS,
                                 train_frac=f)["runs"][0]
        run_seedwise(res, frac_key(f), seeds, one, checkpoint)
    return res


def run_newsite(res: dict, *, kw: dict, rules=RULES, checkpoint=None, ray_shutdown=None) -> dict:
    """Part B. For each seed and holdout rule: federated training without the held-out hospital, then
    zero-shot / fine-tuned / from-scratch at several amounts of local data."""
    from .new_site import run_new_site
    res.setdefault("_meta", {})
    seeds = tuple(kw["seeds"])
    for rule in rules:
        def one(seed, rule=rule):
            if ray_shutdown:
                ray_shutdown()
            return run_new_site(df=kw["df"], partition_fn=kw["partition_fn"], seed=seed, rule=rule,
                                rounds=kw["rounds"], local_epochs=kw["local_epochs"],
                                local_only_epochs=kw["local_only_epochs"], hidden=kw["hidden"],
                                label="newsite")
        run_seedwise(res, f"newsite_{rule}", seeds, one, checkpoint)
    return res
