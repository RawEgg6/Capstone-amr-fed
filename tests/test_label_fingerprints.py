"""Offline tests for label fingerprints, split diagnostics, the era split and budget slicing (no torch):
python tests/test_label_fingerprints.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amr_fed import config
from amr_fed.data_loader import ABX, ORG, PK
from amr_fed.federated import budget as B
from amr_fed.federated import budget_runs as BR
from amr_fed.federated.selection import THRESHOLDS, confusion_counts, select_federated_phc, select_local
from amr_fed.label_fingerprints import (antibiogram_distance, casemix_distance, cross_antibiogram_loss,
                                        score_against, transfer_gap, weights_from_dissimilarity)
from amr_fed.partition import era_cluster_split, patient_era
from amr_fed.split_diagnostics import heterogeneity, heterogeneity_vs_null

TK = config.KEYS["time"]


def _hospital(rng, n, rates, pairs=(("oA", "a0"), ("oA", "a1"), ("oB", "a0"), ("oB", "a1"))):
    rows = []
    for i in range(n):
        o, a = pairs[i % len(pairs)]
        rows.append({ORG: o, ABX: a, "label": int(rng.random() < rates[(o, a)])})
    return pd.DataFrame(rows)


def _pattern(flip: bool):
    base = {("oA", "a0"): 0.1, ("oA", "a1"): 0.6, ("oB", "a0"): 0.5, ("oB", "a1"): 0.2}
    if flip:   # same overall rate, opposite pattern
        base = {("oA", "a0"): 0.6, ("oA", "a1"): 0.1, ("oB", "a0"): 0.2, ("oB", "a1"): 0.5}
    return base


def test_fingerprints_find_the_same_pattern() -> None:
    rng = np.random.default_rng(0)
    frames = [_hospital(rng, 4000, _pattern(f)) for f in (False, False, True, True)]
    for fn in (antibiogram_distance, lambda fr: antibiogram_distance(fr, centred=True), cross_antibiogram_loss):
        D = fn(frames)
        assert D.shape == (4, 4) and np.allclose(np.diag(D), 0.0, atol=1e-9)
        assert D[0, 1] < D[0, 2] and D[2, 3] < D[2, 0], "same-pattern hospitals must look closer"
    # case mix is identical by construction -> casemix cannot tell them apart
    C = casemix_distance(frames)
    assert np.abs(C).max() < 1e-3
    # a pure base-rate shift is invisible to the CENTRED antibiogram but not to the raw one
    shifted = {k: min(v + 0.2, 0.95) for k, v in _pattern(False).items()}
    fr2 = [_hospital(rng, 6000, _pattern(False)), _hospital(rng, 6000, shifted)]
    assert antibiogram_distance(fr2, centred=True)[0, 1] < antibiogram_distance(fr2)[0, 1]
    print("OK: label fingerprints separate resistance patterns; casemix is label-free; centring removes base-rate shift")


def test_weights_and_scoring() -> None:
    D = np.array([[0, 1, 4], [1, 0, 4], [4, 4, 0]], dtype=float)
    W = weights_from_dissimilarity(D, temperature=0.5)
    assert np.allclose(W.sum(1), 1) and W[0, 1] > W[0, 2] and np.all(np.diag(W) >= W.max(1) - 1e-12)
    assert np.allclose(weights_from_dissimilarity(D, np.inf), 1 / 3)
    assert np.allclose(weights_from_dissimilarity(D * 7, 0.5), W), "scale-free"
    rng = np.random.default_rng(1)
    G = rng.random((8, 8)); np.fill_diagonal(G, 0)
    s = score_against(G + 0.01 * rng.random((8, 8)), G, n_perm=200)
    assert s["pooled"] > 0.9 and s["p_perm"] < 0.05
    L = np.array([[1.0, 2.0], [3.0, 1.5]])
    assert np.allclose(transfer_gap(L), [[0, 1], [1.5, 0]])
    print("OK: dissimilarity -> weights, scoring against a transfer gap")


def _era_frame(n_pat=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        t = pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(rng.integers(0, 4000)))
        late = t >= pd.Timestamp("2015-06-01")
        for o, a in (("oA", "a0"), ("oA", "a1")):
            rate = (0.6 if a == "a0" else 0.1) if late else (0.1 if a == "a0" else 0.6)
            rows.append({PK: f"p{p}", ORG: o, ABX: a, TK: str(t), "label": int(rng.random() < rate)})
    return pd.DataFrame(rows)


def test_era_split_and_heterogeneity() -> None:
    df = _era_frame()
    a = era_cluster_split(df, n_clients=4, n_eras=2, seed=3)
    assert a.index.is_unique and len(a) == df[PK].nunique() and set(a.unique()) == {0, 1, 2, 3}
    assert a.equals(era_cluster_split(df, n_clients=4, n_eras=2, seed=3)), "deterministic"
    era = patient_era(df, 2)
    assert ((a // 2) == era.reindex(a.index)).all(), "era_mix=0: every patient in its own era's group"
    mixed = era_cluster_split(df, n_clients=4, n_eras=2, era_mix=0.5, seed=3)
    assert 0.3 < ((mixed // 2) != era.reindex(mixed.index)).mean() < 0.7
    for bad in (dict(n_clients=5, n_eras=2), dict(n_clients=4, n_eras=2, era_mix=1.5)):
        try:
            era_cluster_split(df, **bad)
            raise AssertionError(f"should reject {bad}")
        except ValueError:
            pass
    h = heterogeneity_vs_null(df, a, n_null=3, min_pair_tests=10)
    assert h["concept_excess"] > 0 and h["concept_ratio"] > 2, h
    hr = heterogeneity(df, mixed.sample(frac=1, random_state=0).set_axis(mixed.index), min_pair_tests=10)
    assert hr["concept"] < h["concept"]
    print("OK: era split is 1:1, deterministic, era_mix dials the shift; diagnostic sees the concept shift")


def _fake_run(rng, n_clients=2, rounds=4, ckpts=(4, 8, 12)):
    """A run dict in the saved format: per-round slim records (confusion counts) + local-only checkpoints."""
    def cms(p_good):
        y = rng.random(300) < 0.3
        p = np.where(y, rng.random(300) * p_good + (1 - p_good), rng.random(300) * (1 - p_good))
        return confusion_counts(y, p).tolist()
    recs = [{"per_client_val_cm": {str(c): cms(0.5 + 0.1 * r) for c in range(n_clients)},
             "per_client_test_cm": {str(c): cms(0.5 + 0.1 * r) for c in range(n_clients)},
             "n_val": {str(c): 300 for c in range(n_clients)}, "n_test": {str(c): 300 for c in range(n_clients)}}
            for r in range(rounds)]
    detail = [[{"epoch": e, "val_cm": cms(0.5 + 0.02 * e), "test_cm": cms(0.5 + 0.02 * e)} for e in ckpts]
              for _ in range(n_clients)]
    return {"records": recs, "local_detail": detail, "n_triples": [{"test": 300}] * n_clients,
            "finetune": {"per_client": [0.7, 0.8], "epoch": [8, 6]}}


def test_budget_slicing_matches_selection() -> None:
    from amr_fed.federated.budget import _with_grids, _grid
    rng = np.random.default_rng(2)
    run = _fake_run(rng)
    full = select_federated_phc([_with_grids(r) for r in run["records"]], 2)
    assert abs(B.fed_upto(run, 4)["f1"] - full["test_f1"]) < 1e-9
    assert B.fed_upto(run, 2)["round"] <= 2
    h = run["local_detail"][0]
    one = select_local([_grid(x["val_cm"]) for x in h[:2]], [_grid(x["test_cm"]) for x in h[:2]])
    assert abs(B.local_upto(run, 8)["per_client"][0] - round(one["test_f1"], 4)) < 1e-4
    assert max(B.local_upto(run, 8)["epochs"]) <= 8
    assert abs(B.finetune_score(run)["mean"] - 0.75) < 1e-9
    try:
        B.fed_upto(run, 5)
        raise AssertionError("more rounds than the run has")
    except ValueError:
        pass
    assert BR.LONG_LOCAL == BR.LONG_ROUNDS * BR.LOCAL_EPOCHS + 8
    assert set(BR.PART_B_ARMS) == {"fedavg", "v3_topo", "v3_ab", "v3_ab_shuffled", "oracle", "simft"}
    assert all(a["finetune_epochs"] == 8 for a in BR.PART_B_ARMS.values()), "Part B is budget-matched"
    assert BR.era_group(7) == 1 and BR.era_group(4) == 0
    print("OK: budget slices reproduce the selection rule; arm lists as pre-registered")


if __name__ == "__main__":
    test_fingerprints_find_the_same_pattern()
    test_weights_and_scoring()
    test_era_split_and_heterogeneity()
    test_budget_slicing_matches_selection()
    print("OK: label fingerprint / era split / budget tests passed")
