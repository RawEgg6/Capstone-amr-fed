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


N_HIST_BINS = 20   # predicted-probability histogram resolution (0.05 per bin)


def confusion_counts(y, p, thresholds=THRESHOLDS) -> np.ndarray:
    """(len(thresholds), 4) integer array of [tp, fp, fn, tn] for ``p >= t`` against binary ``y``.

    Every threshold metric (macro-F1, per-class F1, balanced accuracy, ...) can be recomputed from
    these counts later, which is why the federated runs ship them instead of one F1 number."""
    y = np.asarray(y).astype(np.int64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.asarray(thresholds, dtype=np.float64)
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    tp = pos.size - np.searchsorted(pos, t, side="left")
    fp = neg.size - np.searchsorted(neg, t, side="left")
    return np.stack([tp, fp, pos.size - tp, neg.size - fp], axis=1).astype(np.int64)


def macro_f1_from_counts(cm) -> np.ndarray:
    """Macro-F1 per row of a ``confusion_counts`` array (same label handling as macro_f1_grid)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp, fp, fn, tn = cm[:, 0], cm[:, 1], cm[:, 2], cm[:, 3]
    d1, d0 = 2 * tp + fp + fn, 2 * tn + fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(d1 > 0, 2.0 * tp / d1, 0.0)
        f0 = np.where(d0 > 0, 2.0 * tn / d0, 0.0)
    present = (d1 > 0).astype(float) + (d0 > 0).astype(float)
    return (f1 + f0) / present


def prob_hist(y, p, bins: int = N_HIST_BINS) -> list[list[int]]:
    """[positives per probability bin, negatives per probability bin] over [0, 1]."""
    y = np.asarray(y).astype(np.int64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    pos = np.histogram(p[y == 1], bins=bins, range=(0.0, 1.0))[0]
    neg = np.histogram(p[y == 0], bins=bins, range=(0.0, 1.0))[0]
    return [pos.astype(int).tolist(), neg.astype(int).tolist()]


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
    rec = {
        "val_grid": weighted_grid(vg, n_val).tolist(),
        "test_grid": weighted_grid(tg, n_test).tolist(),
        "per_client_val_grid": dict(zip(cids, vg)),
        "per_client_test_grid": dict(zip(cids, tg)),
        "n_val": dict(zip(cids, n_val)),
        "n_test": dict(zip(cids, n_test)),
    }
    # optional raw detail (confusion counts, probability histograms, validation AUROC): stored so
    # any threshold metric / calibration plot can be recomputed without retraining
    for key in ("val_cm", "test_cm", "val_hist", "test_hist"):
        if all(key in m for _, m in metrics):
            rec[f"per_client_{key}"] = {c: json.loads(m[key]) for c, (_, m) in zip(cids, metrics)}
    if all("val_auroc" in m for _, m in metrics):
        rec["per_client_val_auroc"] = {c: m["val_auroc"] for c, (_, m) in zip(cids, metrics)}
    return rec


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


def select_federated_phc(records, n_clients: int) -> dict | None:
    """Per-hospital-cutoff selection (the primary protocol of the final runs).

    Same as ``select_federated`` except that every hospital picks its OWN cutoff on its OWN
    validation rows, which is what local-only has always done. The round is the one with the best
    size-weighted (by n_val) validation macro-F1 when each hospital uses its best cutoff; test is
    read once, at that round and those cutoffs, weighted by n_test.

      round / val_score       chosen round (1-based) and its validation score
      thr                     per-hospital cutoffs (None for a hospital with no validation rows)
      test_f1 / per_client    weighted test macro-F1 and each hospital's, at (round, own cutoff)
      worst                   lowest per-hospital test F1
      final_f1 / final_thr    same rule applied to the last round only
      auroc / per_client_auroc  test AUROC at the chosen round (cutoff-free)
    Returns None for history without per-hospital validation grids."""
    if not records or "per_client_val_grid" not in records[0]:
        return None
    cids = [str(c) for c in range(n_clients)]

    def at_round(rec):
        thr_idx, val_best, test_at, n_v, n_t = [], [], [], [], []
        for c in cids:
            vg = rec["per_client_val_grid"].get(c)
            tg = rec["per_client_test_grid"].get(c)
            nv, nt = rec["n_val"].get(c, 0), rec["n_test"].get(c, 0)
            if vg is None or tg is None or np.isnan(np.asarray(vg, dtype=float)).all():
                thr_idx.append(None); val_best.append(np.nan); test_at.append(np.nan)
            else:
                j = int(np.nanargmax(np.asarray(vg, dtype=float)))
                thr_idx.append(j); val_best.append(float(vg[j])); test_at.append(float(tg[j]))
            n_v.append(nv); n_t.append(nt)
        ok_v = [(v, w) for v, w in zip(val_best, n_v) if v == v and w > 0]
        if not ok_v:
            return None
        val_score = sum(v * w for v, w in ok_v) / sum(w for _, w in ok_v)
        ok_t = [(v, w) for v, w in zip(test_at, n_t) if v == v and w > 0]
        test = sum(v * w for v, w in ok_t) / sum(w for _, w in ok_t) if ok_t else float("nan")
        return {"val": val_score, "test": test, "thr_idx": thr_idx, "per_client": test_at}

    per_round = [at_round(r) for r in records]
    if all(x is None for x in per_round):
        raise ValueError("no usable validation scores to select on")
    r = max((i for i, x in enumerate(per_round) if x is not None), key=lambda i: per_round[i]["val"])
    sel, last = per_round[r], per_round[-1]
    thr = [None if j is None else float(THRESHOLDS[j]) for j in sel["thr_idx"]]
    pc = [None if v != v else round(float(v), 4) for v in sel["per_client"]]
    ok = [v for v in pc if v is not None]
    rec = records[r]
    return {
        "round": r + 1, "val_score": float(sel["val"]), "test_f1": float(sel["test"]), "thr": thr,
        "per_client": pc, "worst": round(min(ok), 4) if ok else None,
        "final_f1": None if last is None else float(last["test"]),
        "final_thr": None if last is None else [None if j is None else float(THRESHOLDS[j]) for j in last["thr_idx"]],
        "auroc": rec.get("auroc") if rec.get("auroc") == rec.get("auroc") else None,
        "per_client_auroc": [rec.get("per_client_auroc", {}).get(c) for c in cids],
    }


def select_local(val_grids, test_grids) -> dict:
    """One hospital's local-only run: grids are checkpoints x thresholds. Picks the
    (checkpoint, cutoff) with the best VAL macro-F1 and reads TEST there."""
    val = np.asarray(val_grids, dtype=np.float64)
    test = np.asarray(test_grids, dtype=np.float64)
    k, j = _argmax2d(val)
    return {"ckpt": k, "thr": float(THRESHOLDS[j]), "val_score": float(val[k, j]),
            "test_f1": float(test[k, j])}
