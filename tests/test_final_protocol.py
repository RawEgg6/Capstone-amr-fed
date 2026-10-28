"""Offline tests for the final-protocol pieces (no torch / flwr / data needed):
    python tests/test_final_protocol.py

Guards: confusion counts reproduce the F1 grid; per-hospital-cutoff selection picks each hospital's
cutoff on ITS validation rows and the round on weighted validation only; v3's similarity matrix is
row-stochastic with the diagonal largest, tends to identity / uniform at the two temperature limits,
and the shuffled control keeps the diagonal while scrambling neighbours; train-only fingerprints use
exactly the rows build_arrays trains on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amr_fed.federated import selection as S
from amr_fed import topology as T


def test_counts_match_grid() -> None:
    rng = np.random.default_rng(1)
    for _ in range(15):
        n = int(rng.integers(5, 300))
        y = (rng.random(n) < rng.uniform(0.05, 0.7)).astype(int)
        p = np.clip(rng.normal(0.3 + 0.4 * y, 0.25), 0, 1)
        cm = S.confusion_counts(y, p)
        assert cm.shape == (29, 4) and (cm.sum(axis=1) == n).all()
        assert np.allclose(S.macro_f1_from_counts(cm), S.macro_f1_grid(y, p), atol=1e-12)
        h = S.prob_hist(y, p)
        assert sum(h[0]) == y.sum() and sum(h[1]) == n - y.sum() and len(h[0]) == S.N_HIST_BINS
    print("OK: confusion counts / histograms reproduce the F1 grid")


def _rec(vals, tests, n_val, n_test, auroc=0.8):
    """One round: per-hospital val/test grids that are constant except where overridden."""
    T_ = len(S.THRESHOLDS)
    rec = {"per_client_val_grid": {}, "per_client_test_grid": {}, "n_val": {}, "n_test": {},
           "auroc": auroc, "per_client_auroc": {}}
    for c, (v, t) in enumerate(zip(vals, tests)):
        rec["per_client_val_grid"][str(c)] = list(v)
        rec["per_client_test_grid"][str(c)] = list(t)
        rec["n_val"][str(c)] = n_val[c]
        rec["n_test"][str(c)] = n_test[c]
        rec["per_client_auroc"][str(c)] = auroc
    return rec


def test_phc_selection() -> None:
    T_ = len(S.THRESHOLDS)
    base = np.full(T_, 0.5)
    def grid(peak_idx=None, val=None):
        g = base.copy()
        if peak_idx is not None:
            g[peak_idx] = val
        return g
    # round 1: hospital 0 peaks at cutoff 3, hospital 1 at cutoff 20 -> different cutoffs per hospital
    v1 = [grid(3, 0.7), grid(20, 0.8)]
    t1 = [grid(3, 0.60), grid(20, 0.70)]
    # round 2: worse validation everywhere, but a (misleading) great test number the rule must not see
    v2 = [grid(3, 0.6), grid(20, 0.6)]
    t2 = [grid(3, 0.95), grid(20, 0.95)]
    recs = [_rec(v1, t1, [100, 300], [200, 100]), _rec(v2, t2, [100, 300], [200, 100])]
    sel = S.select_federated_phc(recs, 2)
    assert sel["round"] == 1
    assert sel["thr"] == [float(S.THRESHOLDS[3]), float(S.THRESHOLDS[20])]
    assert sel["per_client"] == [0.6, 0.7]
    want = (0.60 * 200 + 0.70 * 100) / 300           # test weighted by n_test, not n_val
    assert abs(sel["test_f1"] - want) < 1e-12
    assert abs(sel["val_score"] - (0.7 * 100 + 0.8 * 300) / 400) < 1e-12
    assert sel["worst"] == 0.6
    assert S.select_federated_phc([{"macro_f1": 0.7}], 2) is None
    # a hospital with no validation rows is skipped, not propagated
    nan = np.full(T_, np.nan)
    sel2 = S.select_federated_phc([_rec([grid(3, 0.7), nan], [grid(3, 0.6), nan], [100, 0], [200, 0])], 2)
    assert sel2["thr"][1] is None and abs(sel2["test_f1"] - 0.6) < 1e-12
    print("OK: per-hospital cutoffs chosen on each hospital's validation rows; round on weighted validation")


def _fps(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [T.TopologyFingerprint(rng.normal(size=len(T.FEATURE_NAMES)) + 5) for _ in range(n)]


def test_personalized_weights() -> None:
    fps = _fps(5)
    for temp in (0.25, 0.5, 1.0, 2.0):
        W = T.personalized_weights(fps, temperature=temp)
        assert W.shape == (5, 5) and np.allclose(W.sum(axis=1), 1.0) and (W > 0).all()
        assert (W.argmax(axis=1) == np.arange(5)).all(), "diagonal must be the largest entry"
    assert np.allclose(T.personalized_weights(fps, temperature=float("inf")), 0.2)
    assert np.diag(T.personalized_weights(fps, temperature=1e-3)).min() > 0.999
    # sharper temperature -> more weight on self
    assert np.diag(T.personalized_weights(fps, 0.25)).mean() > np.diag(T.personalized_weights(fps, 2.0)).mean()
    # size features are ignored: changing only log_patients / log_tested_edges leaves W unchanged
    base = np.vstack([f.features for f in fps])
    alt = base.copy()
    for name in T.SIZE_FEATURES:
        alt[:, T.FEATURE_NAMES.index(name)] += np.arange(5) * 3.0
    Wb = T.personalized_weights([T.TopologyFingerprint(r) for r in base], 1.0)
    Wa = T.personalized_weights([T.TopologyFingerprint(r) for r in alt], 1.0)
    assert np.allclose(Wb, Wa), "size features must not influence v3 similarity"
    # shuffled control: fingerprints handed to the wrong hospitals; diagonal still dominant
    perm = T.derangement(5, 7)
    Ws = T.personalized_weights([fps[o] for o in perm], 1.0)
    assert (Ws.argmax(axis=1) == np.arange(5)).all() and not np.allclose(Ws, Wb)
    # one hospital
    assert np.allclose(T.personalized_weights(fps[:1], 1.0), 1.0)
    print("OK: v3 weights are row-stochastic, diagonal-dominant, hit both temperature limits, ignore size")


def test_train_frame_matches_build_arrays() -> None:
    from amr_fed.graph_build import _patient_grouped_split
    from amr_fed.data_loader import PK
    rng = np.random.default_rng(3)
    df = pd.DataFrame({PK: rng.integers(0, 400, 3000).astype(str), "x": rng.random(3000)})
    for seed in (42, 43):
        tr = T.train_frame(df, seed)
        code = _patient_grouped_split(df[PK].to_numpy(), seed)
        want = df[df[PK].map(code).to_numpy() == 0]
        assert tr.index.equals(want.index) and 0.55 < len(tr) / len(df) < 0.85
    print("OK: train_frame uses exactly the rows build_arrays trains on")


if __name__ == "__main__":
    test_counts_match_grid()
    test_phc_selection()
    test_personalized_weights()
    test_train_frame_matches_build_arrays()
    print("OK: final-protocol tests passed")
