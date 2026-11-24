"""Arm list and run loop for notebook 19 (SimFT). Resume-safe, seed by seed. See docs/2026-10-28-simft-preregistration.md.

Settings (one partition per seed). In every setting ONLY the SimFT arms are trained; local-only, FedAvg and
FedAvg + fine-tune are read from the notebook that already ran them on the same split and seeds:
  louvain  louvain_split(n_clients=3)                     baselines: notebook 16 (results/amr_louvain_final_all.json)
  org10    organism_community(n_clients=10)               baselines: notebook 17 (results/amr_org10_final_all.json)
  ward10   dirichlet_ward_mixture(10, alpha=0.5), 10% of train/validation patients kept
                                                           baselines: notebook 18 (scarcity_f0.1 in results/amr_regimes_all.json)
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
    """(key, label, run_multiseed kwargs) in run order: SimFT at each tau, then the flat ablation."""
    if setting not in ("louvain", "org10", "ward10"):
        raise ValueError(f"unknown setting {setting!r}")
    common = dict(strategy="simft", simft_self_floor=SELF_FLOOR, finetune_epochs=FT_EPOCHS,
                  compute_local=False, compute_pooled=False)
    if setting == "ward10":
        common["train_frac"] = FRAC_WARD
    specs = [(tau_key(t), f"{setting}/SimFT (tau={t:g})", dict(common, simft_tau=t, simft_flat=False)) for t in TAUS]
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


# where each setting's baselines (local-only, FedAvg, FedAvg + fine-tune) were produced
BASELINES = {
    "louvain": ("amr_louvain_final_all.json", "fedavg"),
    "org10": ("amr_org10_final_all.json", "fedavg"),
    "ward10": ("amr_regimes_all.json", "scarcity_f0.1"),
}


def baseline_from_file(setting: str, search_dirs, n_seeds: int = 5):
    """Load the finished baseline arm for ``setting`` from the first directory that has its results file.
    Raises if the arm is missing or unfinished (e.g. notebook 17 still running)."""
    import json
    from pathlib import Path
    fname, key = BASELINES[setting]
    path = next((Path(d) / fname for d in search_dirs if (Path(d) / fname).exists()), None)
    if path is None:
        raise FileNotFoundError(f"{fname} not found in {[str(d) for d in search_dirs]} - put the notebook's results file there")
    res = json.load(open(path))
    arm = res.get(key)
    if arm is None or len(arm.get("runs", [])) < n_seeds or "finetune" not in arm["runs"][0]:
        raise RuntimeError(f"{path}: baseline arm {key!r} is missing or unfinished (needs {n_seeds} seeds with the fine-tune baseline)")
    return arm, path
