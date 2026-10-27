"""Offline tests for validation-based round/cutoff selection (federated/selection.py).

Pure numpy + sklearn — no torch / flwr / data needed:  python tests/test_val_selection.py

What it guards:
- macro_f1_grid reproduces sklearn macro-F1 at every cutoff (incl. single-class and
  all-one-prediction edge cases), and its 0.5 column equals the legacy `>= 0.5` score
- grid_record size-weights val and test grids separately and tolerates NaN/empty hospitals
- select_federated chooses the round+cutoff from VALIDATION only (test peak elsewhere is ignored)
- select_local does the same per hospital; old-format history (no val grids) returns None
- the client->server JSON payload (NaN included) survives a round trip
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import f1_score

from amr_fed.federated import selection as S


def test_grid_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(20):
        n = int(rng.integers(5, 400))
        y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
        p = np.clip(rng.normal(0.3 + 0.4 * y, 0.25), 0, 1)
        cases.append((y, p))
    cases.append((np.zeros(30, int), rng.random(30)))            # single class (all negative)
    cases.append((np.ones(30, int), rng.random(30)))             # single class (all positive)
    cases.append((rng.integers(0, 2, 50), np.full(50, 0.5)))      # constant scores, ties at 0.5
    cases.append((np.array([0, 1]), np.array([0.2, 0.9])))        # tiny
    for y, p in cases:
        g = S.macro_f1_grid(y, p)
        want = np.array([f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
                         for t in S.THRESHOLDS])
        assert np.allclose(g, want, atol=1e-12), (g - want)
    y, p = cases[0]
    assert abs(S.macro_f1_grid(y, p)[S.IDX_05] - f1_score(y, (p >= 0.5).astype(int), average="macro")) < 1e-12
    assert S.THRESHOLDS[S.IDX_05] == 0.5 and len(S.THRESHOLDS) == 29
    assert np.isnan(S.macro_f1_grid([], [])).all()
    print("OK: macro_f1_grid matches sklearn on", len(cases), "cases")


def _client_metrics(cid, n_val, val, test):
    return {"cid": cid, "n_val": n_val, "val_grid": json.dumps(list(val)),
            "test_grid": json.dumps(list(test))}


def test_grid_record_weights() -> None:
    T = len(S.THRESHOLDS)
    a_val, a_test = np.full(T, 0.6), np.full(T, 0.5)
    b_val, b_test = np.full(T, 0.8), np.full(T, 0.7)
    rec = S.grid_record([(300, _client_metrics(0, 100, a_val, a_test)),
                         (100, _client_metrics(1, 300, b_val, b_test))])
    # val weighted by n_val (100:300), test by n_test (300:100) — different weights on purpose
    assert np.allclose(rec["val_grid"], 0.75 * 0.8 + 0.25 * 0.6)
    assert np.allclose(rec["test_grid"], 0.75 * 0.5 + 0.25 * 0.7)
    assert rec["n_val"] == {"0": 100, "1": 300} and rec["n_test"] == {"0": 300, "1": 100}
    # an empty-val hospital (NaN grid, n_val=0) is skipped, not propagated
    nan = [float("nan")] * T
    rec2 = S.grid_record([(300, _client_metrics(0, 100, a_val, a_test)),
                          (100, _client_metrics(1, 0, nan, b_test))])
    assert np.allclose(rec2["val_grid"], 0.6) and np.allclose(rec2["test_grid"], 0.75 * 0.5 + 0.25 * 0.7)
    # JSON round trip with NaN (what Flower carries as a str)
    assert np.isnan(json.loads(json.dumps(nan))).all()
    print("OK: grid_record weights val/test separately and skips empty hospitals")


def _records(val_by_round, test_by_round, n_clients=2):
    """History records whose grids are constant across cutoffs except where overridden."""
    recs = []
    for v, t in zip(val_by_round, test_by_round):
        recs.append({"macro_f1": float(t[S.IDX_05]), "auroc": 0.8,
                     "val_grid": list(v), "test_grid": list(t),
                     "per_client_test_grid": {str(c): [x - 0.01 * c for x in t] for c in range(n_clients)},
                     "per_client_auroc": {str(c): 0.8 for c in range(n_clients)}})
    return recs


def test_select_federated_uses_val_only() -> None:
    T = len(S.THRESHOLDS)
    base = np.full(T, 0.60)
    v = [base.copy() for _ in range(4)]
    t = [base.copy() for _ in range(4)]
    v[1][20] = 0.70                        # val peaks at round 2, cutoff index 20
    t[1][20] = 0.65                        # ...where test is only 0.65
    t[2][S.IDX_05] = 0.99                  # test peaks (a lie) at round 3 / 0.5 — must be ignored
    t[3][S.IDX_05] = 0.62                  # final round @0.5
    v[3][5] = 0.66                         # final round's own best val cutoff is index 5
    t[3][5] = 0.61
    sel = S.select_federated(_records(v, t), 2)
    assert sel["round"] == 2 and abs(sel["thr"] - S.THRESHOLDS[20]) < 1e-12
    assert abs(sel["test_f1"] - 0.65) < 1e-12 and abs(sel["val_score"] - 0.70) < 1e-12
    assert abs(sel["test_f1_at05"] - 0.60) < 1e-12
    assert abs(sel["final_f1"] - 0.61) < 1e-12 and abs(sel["final_thr"] - S.THRESHOLDS[5]) < 1e-12
    assert sel["per_client"] == [0.65, 0.64] and sel["worst"] == 0.64
    assert S.select_federated([{"macro_f1": 0.7, "per_client": {}}], 2) is None   # old format
    assert S.select_federated([], 2) is None
    print("OK: select_federated picks round+cutoff on val; the test peak is ignored")


def test_select_local() -> None:
    T = len(S.THRESHOLDS)
    val, test = np.full((8, T), 0.5), np.full((8, T), 0.5)
    val[5, 3], test[5, 3] = 0.7, 0.6
    test[7, S.IDX_05] = 0.9                # a better test number the val rule must not see
    sel = S.select_local(val, test)
    assert sel["ckpt"] == 5 and abs(sel["thr"] - S.THRESHOLDS[3]) < 1e-12 and sel["test_f1"] == 0.6
    try:
        S.select_local(np.full((2, T), np.nan), np.full((2, T), 0.5))
        raise AssertionError("expected ValueError for all-NaN validation")
    except ValueError:
        pass
    print("OK: select_local chooses on val; all-NaN val raises")


if __name__ == "__main__":
    test_grid_matches_sklearn()
    test_grid_record_weights()
    test_select_federated_uses_val_only()
    test_select_local()
    print("OK: validation-selection tests passed")
