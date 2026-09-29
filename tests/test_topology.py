"""Offline unit tests for the v0 topology fingerprint + similarity module.

No data / no GPU / no torch needed — pure pandas + numpy. Run:
    python tests/test_topology.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.topology import (
    FEATURE_NAMES, TopologyFingerprint, aggregation_weights,
    compute_fingerprint, cosine_similarity_matrix, distance_to_centroid,
    fingerprint_matrix, zscore,
)

D = len(FEATURE_NAMES)  # 14


def _frame() -> pd.DataFrame:
    """Deterministic tiny per-test frame: 8 patients, 2 organisms, 3 antibiotics.
    Mixed labels + timestamps (the history feature needs the time column)."""
    rows = []
    for i in range(8):
        for j, o in enumerate(["oA", "oB"]):
            for k, a in enumerate(["a0", "a1", "a2"]):
                if (i + j + k) % 3 == 0:
                    rows.append((f"p{i}", o, a, 1 if (i + k) % 2 == 0 else 0,
                                 f"2020-01-{(i % 5) + 1:02d}"))
    return pd.DataFrame(rows, columns=["anon_id", "organism", "antibiotic",
                                       "label", "order_time_jittered_utc"])


def _fps(n: int, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    return [TopologyFingerprint(rng.normal(size=D)) for _ in range(n)]


def test_fingerprint_shape_names_and_dict():
    fp = compute_fingerprint(_frame(), include_history=False)
    assert isinstance(fp, TopologyFingerprint)
    assert len(fp) == D == 14
    assert fp.names == FEATURE_NAMES
    assert np.all(np.isfinite(fp.features))
    d = fp.as_dict()
    assert list(d.keys()) == list(FEATURE_NAMES)
    assert np.allclose(list(d.values()), fp.features)


def test_fingerprint_deterministic():
    a = compute_fingerprint(_frame(), include_history=True)
    b = compute_fingerprint(_frame(), include_history=True)
    assert np.array_equal(a.features, b.features)   # exact, no RNG involved


def test_history_flag_only_touches_history_feature():
    off = compute_fingerprint(_frame(), include_history=False)
    on = compute_fingerprint(_frame(), include_history=True)
    idx = FEATURE_NAMES.index("history_prior_rate")
    assert np.allclose(off.features[:idx], on.features[:idx])
    assert np.allclose(off.features[idx + 1:], on.features[idx + 1:])
    assert np.isfinite(on.features[idx])
    assert off.features[idx] == 0.0


def test_fingerprint_rejects_empty_frame():
    try:
        compute_fingerprint(_frame().iloc[0:0])
        raise AssertionError("empty frame should raise")
    except ValueError:
        pass


def test_zscore_standardizes_and_handles_constant_column():
    f = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    z = zscore(f)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-12)
    # a zero-variance column must become zeros, not NaN
    zc = zscore(np.array([[5.0, 1.0], [5.0, 2.0]]))
    assert np.allclose(zc[:, 0], 0.0) and np.isfinite(zc).all()


def test_cosine_similarity_properties():
    f = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    s = cosine_similarity_matrix(f)
    assert np.allclose(s, s.T)                       # symmetric
    assert np.allclose(np.diag(s)[:4], 1.0)          # nonzero rows self-similarity = 1
    assert np.isclose(s[2, 3], 0.0)                  # orthogonal -> 0
    assert np.allclose(s[0, 1], 1.0)                 # identical rows -> 1
    assert np.allclose(s[4, :], 0.0)                 # zero vector -> 0, no NaN


def test_distance_to_centroid_flags_outlier():
    base = np.zeros(D)
    rows = np.array([base + 9.0, base + 0.1, base - 0.1, base + 0.05])
    d = distance_to_centroid(rows)
    assert np.all(np.isfinite(d))
    assert d[0] == d.max()                           # the big row is furthest


def test_aggregation_weights_temperature_inf_is_uniform():
    w = aggregation_weights(_fps(4), temperature=np.inf)
    assert np.allclose(w, 0.25)                      # exact FedAvg anchor
    assert np.isclose(w.sum(), 1.0)


def test_aggregation_weights_consensus_vs_distinctiveness():
    base = np.zeros(D)
    rows = np.array([base + 9.0, base + 0.1, base - 0.1, base + 0.05])
    fps = [TopologyFingerprint(r) for r in rows]
    w_c = aggregation_weights(fps, temperature=0.5, mode="consensus")
    w_d = aggregation_weights(fps, temperature=0.5, mode="distinctiveness")
    assert np.isclose(w_c.sum(), 1.0) and np.isclose(w_d.sum(), 1.0)
    assert w_c[0] < 0.25                             # outlier down-weighted under consensus
    assert w_d[0] > 0.25                             # outlier up-weighted under distinctiveness
    assert w_c[0] < w_d[0]


def test_aggregation_weights_temperature_controls_peakedness():
    fps = _fps(5)
    sharp = aggregation_weights(fps, temperature=0.1)
    soft = aggregation_weights(fps, temperature=10.0)
    assert np.isclose(sharp.sum(), 1.0) and np.isclose(soft.sum(), 1.0)
    assert sharp.var() > soft.var()                  # small T -> more concentrated
    assert soft.var() < 0.01                          # large T -> near uniform


def test_aggregation_weights_guards():
    fps = _fps(2)
    for bad in ({"mode": "nope"}, {"temperature": 0.0}, {"temperature": -1.0}):
        try:
            aggregation_weights(fps, **bad)
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass
    assert np.allclose(aggregation_weights(_fps(1)), 1.0)   # one client -> weight 1


def test_fingerprint_matrix_shape():
    m = fingerprint_matrix([compute_fingerprint(_frame(), include_history=False)] * 3)
    assert m.shape == (3, D)


if __name__ == "__main__":
    test_fingerprint_shape_names_and_dict()
    test_fingerprint_deterministic()
    test_history_flag_only_touches_history_feature()
    test_fingerprint_rejects_empty_frame()
    test_zscore_standardizes_and_handles_constant_column()
    test_cosine_similarity_properties()
    test_distance_to_centroid_flags_outlier()
    test_aggregation_weights_temperature_inf_is_uniform()
    test_aggregation_weights_consensus_vs_distinctiveness()
    test_aggregation_weights_temperature_controls_peakedness()
    test_aggregation_weights_guards()
    test_fingerprint_matrix_shape()
    print("OK: topology unit tests passed.")
