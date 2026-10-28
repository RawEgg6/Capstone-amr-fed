"""Arm list and run loop for notebook 19 (SimFT). Resume-safe, seed by seed. See docs/2026-10-28-simft-preregistration.md.

Two settings (one partition per seed in both):
  conflict  antibiogram_conflict(n_clients=5): FedAvg (+ local-only + fine-tune) AND the SimFT arms are run here
  ward10    dirichlet_ward_mixture(10, alpha=0.5) keeping 10% of every hospital's train/validation patients: ONLY the
            SimFT arms are run; local-only / FedAvg / FedAvg + fine-tune come from notebook 18 (``scarcity_f0.1``)
Arms: SimFT at two temperatures (best chosen on validation half B) and the equal-neighbour ablation (``flat``).
"""
from __future__ import annotations

import numpy as np

from .final_protocol import FT_EPOCHS, run_seedwise

TAUS = (0.5, 1.0)
SELF_FLOOR = 0.25
FRAC_WARD = 0.10


def tau_key(t: float) -> str:
    return f"simft_t{t:g}"


def arm_specs(setting: str) -> list[tuple[str, str, dict]]:
    """(key, label, run_multiseed kwargs) in run order."""
    common = dict(strategy="simft", simft_self_floor=SELF_FLOOR, finetune_epochs=FT_EPOCHS,
                  compute_local=False, compute_pooled=False)
    if setting == "ward10":
        common["train_frac"] = FRAC_WARD
    specs = []
    if setting == "conflict":
        specs.append(("fedavg", "conflict/FedAvg", dict(strategy="fedavg", compute_local=True, compute_pooled=False,
                                                      finetune_epochs=FT_EPOCHS)))
    specs += [(tau_key(t), f"{setting}/SimFT (tau={t:g})", dict(common, simft_tau=t, simft_flat=False)) for t in TAUS]
    specs.append(("simft_flat", f"{setting}/SimFT-flat (ablation)", dict(common, simft_tau=1.0, simft_flat=True)))
    return specs


def choose_tau(res: dict) -> tuple[float, dict]:
    """tau with the best mean VALIDATION (half B) score across seeds; ties go to the smaller tau. Test is not used."""
    scores = {}
    for t in TAUS:
        arm = res.get(tau_key(t))
        if arm is None:
            raise KeyError(f"{tau_key(t)} not finished yet")
        scores[t] = float(np.mean([r["phc_val_score"] for r in arm["runs"]]))
    return max(sorted(scores), key=lambda t: scores[t]), scores


def run_setting(res: dict, setting: str, *, kw: dict, checkpoint=None, ray_shutdown=None) -> dict:
    """Run every arm of ``setting`` that is not already in ``res``."""
    from .run import run_multiseed
    res.setdefault("_meta", {})
    seeds = tuple(kw["seeds"])
    for key, label, extra in arm_specs(setting):
        def one(seed, label=label, extra=extra):
            if ray_shutdown:
                ray_shutdown()
            return run_multiseed(label=label, **dict(kw, seeds=(seed,)), **extra)["runs"][0]
        run_seedwise(res, key, seeds, one, checkpoint)
    if all(tau_key(t) in res for t in TAUS):
        tau, scores = choose_tau(res)
        res["_meta"]["chosen_tau"] = tau
        res["_meta"]["tau_val_scores"] = scores
        print(f"SimFT temperature chosen on validation half B: tau={tau:g} (mean val F1 by tau: "
              f"{ {t: round(v, 4) for t, v in scores.items()} })")
    return res
