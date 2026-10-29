"""Budget-matched analysis for notebook 21 (torch-free).

Why: every earlier fine-tune comparison gave FedAvg + fine-tune 8 more local epochs than local-only and FedAvg
(40 vs 32), and every arm was still improving at the budget (round 8 / epoch 32 selected almost always). This module
reads ANY smaller budget out of one long run, because training is identical up to round R / epoch E however long the
run continues (constant learning rate, no schedule, deterministic seeds):

  fed_upto(run, R)    FedAvg restricted to rounds 1..R: round and per-hospital cutoffs chosen on validation among
                      those rounds only (``selection.select_federated_phc`` on the first R records)
  local_upto(run, E)  local-only restricted to checkpoints at epochs <= E, (checkpoint, cutoff) chosen on validation
                      per hospital (``selection.select_local``)

Both are rebuilt from the stored confusion counts, so they reproduce the run's own ``phc_f1`` / ``val_local`` at the
full budget exactly (tests/test_budget.py). One local epoch = one full-batch step; FedAvg round R costs
R x local_epochs local epochs per hospital; fine-tuning after round R adds ``ft_epochs``.
"""
from __future__ import annotations

import numpy as np

from .selection import macro_f1_from_counts, select_federated_phc, select_local


def _grid(cm):
    return macro_f1_from_counts(np.asarray(cm)).tolist()


def _with_grids(rec: dict) -> dict:
    """A slim history record (confusion counts only) with the per-hospital F1 grids rebuilt."""
    out = dict(rec)
    out["per_client_val_grid"] = {c: _grid(cm) for c, cm in rec["per_client_val_cm"].items()}
    out["per_client_test_grid"] = {c: _grid(cm) for c, cm in rec["per_client_test_cm"].items()}
    return out


def _summ(per_client, n_test) -> dict:
    pc = np.asarray(per_client, dtype=float)
    w = np.asarray(n_test, dtype=float)
    ok = ~np.isnan(pc) & (w > 0)
    return {"f1": float(np.average(pc[ok], weights=w[ok])), "mean": float(pc[ok].mean()),
            "worst": float(pc[ok].min()), "per_client": pc.round(4).tolist()}


def n_test(run: dict) -> list[int]:
    return [t["test"] for t in run["n_triples"]]


def fed_upto(run: dict, max_round: int) -> dict:
    """Primary-protocol FedAvg score when only rounds 1..max_round exist."""
    recs = run["records"][:max_round]
    if len(recs) < max_round:
        raise ValueError(f"run has {len(run['records'])} rounds, asked for {max_round}")
    n = len(run["n_triples"])
    sel = select_federated_phc([_with_grids(r) for r in recs], n)
    out = _summ([np.nan if v is None else v for v in sel["per_client"]], n_test(run))
    out["round"] = sel["round"]
    return out


def local_upto(run: dict, max_epoch: int) -> dict:
    """Local-only score when each hospital may only use checkpoints at epochs <= max_epoch."""
    pcs, eps = [], []
    for hist in run["local_detail"]:
        h = [x for x in hist if x["epoch"] <= max_epoch]
        if not h:
            raise ValueError(f"no local checkpoint at or before epoch {max_epoch}")
        sel = select_local([_grid(x["val_cm"]) for x in h], [_grid(x["test_cm"]) for x in h])
        pcs.append(sel["test_f1"])
        eps.append(h[sel["ckpt"]]["epoch"])
    out = _summ(pcs, n_test(run))
    out["epochs"] = eps
    return out


def finetune_score(run: dict) -> dict:
    ft = run["finetune"]
    out = _summ(ft["per_client"], n_test(run))
    out["epochs"] = ft["epoch"]
    return out


def last_choice_share(run: dict, max_round: int | None = None) -> float:
    """Share of hospitals whose local-only best checkpoint is the LAST one allowed (budget-limited sign)."""
    last = max(x["epoch"] for x in run["local_detail"][0]) if max_round is None else max_round
    sel = local_upto(run, last)["epochs"]
    return float(np.mean([e == last for e in sel]))
