"""Validation-based round / threshold selection for the federated runs (torch-free).

Why this exists: the original protocol reported the *best round by test macro-F1* at a fixed
0.5 cutoff, so every federated arm got to pick its peak on the data it was scored on, while
local-only was scored once. Here each hospital scores the global model on BOTH its validation
and its test rows every round, at a grid of cutoffs. The server picks the (round, cutoff) with
the best size-weighted *validation* macro-F1, and the test score is read only at that choice.

Only per-hospital F1 values at 29 cutoffs travel to the server (no patient data, no logits).
The F1 definition is unchanged: per-hospital macro-F1, size-weighted over hospitals.

Pure numpy so it can be unit-tested without torch / flwr (tests/test_val_selection.py).
"""
from __future__ import annotations

import json

import numpy as np

# 0.150, 0.175, ..., 0.850 — the same range as train_local._best_threshold, built from
# integers so 0.5 is represented exactly (index IDX_05).
THRESHOLDS = np.array([(150 + 25 * i) / 1000 for i in range(29)], dtype=np.float64)
IDX_05 = 14
assert THRESHOLDS[IDX_05] == 0.5


def macro_f1_grid(y, p, thresholds=THRESHOLDS) -> np.ndarray:
    """Macro-F1 of ``p >= t`` against binary ``y`` for every t, vectorised.

    Matches ``sklearn.metrics.f1_score(average="macro")`` including its label handling: a
    class that is absent from both y and the predictions is skipped (not counted as 0), so a
    single-class hospital behaves exactly as the legacy ``_macro_f1``. Returns NaN for every
    threshold when y is empty.
    """
    y = np.asarray(y).astype(np.int64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.asarray(thresholds, dtype=np.float64)
    if y.size == 0:
        return np.full(t.shape, np.nan)
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    n_pos, n_neg = pos.size, neg.size
    tp = n_pos - np.searchsorted(pos, t, side="left")   # positives predicted positive
    fp = n_neg - np.searchsorted(neg, t, side="left")   # negatives predicted positive
    fn = n_pos - tp
    tn = n_neg - fp
    # class 1: (tp, fp, fn); class 0 swaps roles: (tn, fn, fp)
    d1, d0 = 2 * tp + fp + fn, 2 * tn + fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(d1 > 0, 2.0 * tp / d1, 0.0)
        f0 = np.where(d0 > 0, 2.0 * tn / d0, 0.0)
    present = (d1 > 0).astype(float) + (d0 > 0).astype(float)   # sklearn: labels in y or pred
    return (f1 + f0) / present


def weighted_grid(grids, weights) -> np.ndarray:
    """Weighted mean of per-hospital grids; hospitals with weight 0 or a NaN grid are skipped.
    Returns all-NaN when nothing is usable."""
    g = np.asarray(grids, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = (w > 0) & ~np.isnan(g).any(axis=1)
    if not ok.any():
        return np.full(g.shape[1], np.nan)
    return (g[ok] * w[ok, None]).sum(axis=0) / w[ok].sum()


def _argmax2d(m: np.ndarray) -> tuple[int, int]:
    m = np.asarray(m, dtype=np.float64)
    if np.isnan(m).all():
        raise ValueError("no usable validation scores to select on")
    return tuple(int(i) for i in np.unravel_index(np.nanargmax(m), m.shape))


# ---- server side: turn the clients' evaluate() metrics into one history record ----------
def grid_record(metrics) -> dict:
    """metrics: list of (n_test, client_metrics) as handed to the server's
    ``evaluate_metrics_aggregation_fn``. Each client_metrics carries ``cid``, ``n_val`` and the
    JSON-encoded ``val_grid`` / ``test_grid``. Returns the extra fields appended to the round's
    history record: size-weighted val/test grids plus the per-hospital grids."""
    cids, n_test, n_val, vg, tg = [], [], [], [], []
    for i, (n, m) in enumerate(metrics):
        cids.append(str(m.get("cid", i)))
        n_test.append(int(n))
        n_val.append(int(m.get("n_val", 0)))
        vg.append(json.loads(m["val_grid"]))
        tg.append(json.loads(m["test_grid"]))
    return {
        "val_grid": weighted_grid(vg, n_val).tolist(),
        "test_grid": weighted_grid(tg, n_test).tolist(),
        "per_client_val_grid": dict(zip(cids, vg)),
        "per_client_test_grid": dict(zip(cids, tg)),
        "n_val": dict(zip(cids, n_val)),
        "n_test": dict(zip(cids, n_test)),
    }


# ---- driver side: pick the round/cutoff on validation, read test there ------------------
def select_federated(records, n_clients: int) -> dict | None:
    """records: per-round history (task.read_fed_records()). Returns None when the records
    carry no validation grids (old-format history). Otherwise:

      round / thr       1-based round and cutoff chosen by best weighted VAL macro-F1
      val_score         that validation score
      test_f1           weighted TEST macro-F1 at (round, thr)  <- the reported number
      test_f1_at05      test macro-F1 at the chosen round but the default 0.5 cutoff
      final_f1          test macro-F1 after the last round, cutoff tuned on val for that round
      per_client        per-hospital test F1 at (round, thr); worst = their min
      auroc / per_client_auroc   test AUROC at the chosen round (threshold-free)
    """
    if not records or "val_grid" not in records[0]:
        return None
    val = np.array([r["val_grid"] for r in records], dtype=np.float64)     # rounds x thresholds
    test = np.array([r["test_grid"] for r in records], dtype=np.float64)
    r, j = _argmax2d(val)
    j_last = int(np.nanargmax(val[-1]))
    rec = records[r]
    per_client = [rec["per_client_test_grid"].get(str(c)) for c in range(n_clients)]
    per_client = [None if g is None else round(float(g[j]), 4) for g in per_client]
    ok = [v for v in per_client if v is not None and v == v]
    auc = rec.get("auroc")
    return {
        "round": r + 1, "thr": float(THRESHOLDS[j]), "val_score": float(val[r, j]),
        "test_f1": float(test[r, j]), "test_f1_at05": float(test[r, IDX_05]),
        "final_f1": float(test[-1, j_last]), "final_thr": float(THRESHOLDS[j_last]),
        "per_client": per_client, "worst": round(min(ok), 4) if ok else None,
        "auroc": auc if auc is not None and auc == auc else None,
        "per_client_auroc": [rec.get("per_client_auroc", {}).get(str(c)) for c in range(n_clients)],
    }


def select_local(val_grids, test_grids) -> dict:
    """One hospital's local-only run: grids are checkpoints x thresholds. Picks the
    (checkpoint, cutoff) with the best VAL macro-F1 and reads TEST there."""
    val = np.asarray(val_grids, dtype=np.float64)
    test = np.asarray(test_grids, dtype=np.float64)
    k, j = _argmax2d(val)
    return {"ckpt": k, "thr": float(THRESHOLDS[j]), "val_score": float(val[k, j]),
            "test_f1": float(test[k, j])}
