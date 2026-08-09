"""Offline unit tests for partition.py pure logic. Run: python tests/test_partition.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.data_loader import PK, ORG, ABX
from amr_fed.partition import WARD_COL, _apportion, assign_home_ward, dirichlet_ward_mixture


def test_assign_home_ward_priority():
    df = pd.DataFrame({
        PK:       ["pA", "pA", "pB", "pC", "pC"],
        WARD_COL: ["OP", "ICU", "ER", "NONE", "NONE"],   # pA: OP+ICU->ICU ; pC: only NONE
    })
    home = assign_home_ward(df)
    assert home["pA"] == "ICU"      # highest acuity wins
    assert home["pB"] == "ER"
    assert home["pC"] == "NONE"


def test_apportion_sums_to_n():
    for n in (0, 1, 7, 100):
        c = _apportion(n, np.array([0.5, 0.3, 0.2]))
        assert c.sum() == n and (c >= 0).all()


def test_dirichlet_assigns_every_patient_once():
    df = pd.DataFrame({
        PK:       [f"p{i}" for i in range(200)],
        WARD_COL: (["ICU", "ER", "IP", "OP"] * 50),
    })
    a = dirichlet_ward_mixture(df, n_clients=5, alpha=0.5, seed=42)
    a2 = dirichlet_ward_mixture(df, n_clients=5, alpha=0.5, seed=42)
    assert a.equals(a2)                              # deterministic
    assert a.index.is_unique and len(a) == 200       # 1:1, no leakage
    assert set(a.unique()) <= set(range(5)) and (a >= 0).all()


from amr_fed.partition import _rank_bucket_split


def test_rank_bucket_split_quantile_dial():
    score = pd.Series({"pA": 5.0, "pB": 1.0, "pC": 9.0, "pD": 3.0, "pE": 7.0, "pF": 2.0})
    a = _rank_bucket_split(score, 2)
    assert a.index.is_unique and len(a) == 6
    assert set(a.unique()) == {0, 1}
    # balanced sizes: 3 / 3
    assert list(a.value_counts().sort_index()) == [3, 3]
    # hospital 0 holds the lowest scores, hospital 1 the highest
    assert set(a.index[a == 0]) == {"pB", "pD", "pF"}
    assert set(a.index[a == 1]) == {"pA", "pC", "pE"}
    # deterministic
    assert _rank_bucket_split(score, 2).equals(a)


def test_rank_bucket_split_guards():
    score = pd.Series({"pA": 1.0, "pB": 2.0})
    try:
        _rank_bucket_split(score, 0)
        raise AssertionError("should have raised for n_clients < 1")
    except ValueError:
        pass
    try:
        _rank_bucket_split(score, 5)
        raise AssertionError("should have raised for n_clients > #patients")
    except ValueError:
        pass


from amr_fed.partition import _patient_homophily


def test_patient_homophily_clustered_vs_scattered():
    # o_clust: every tested edge majority-RESISTANT -> homophilic (dev > 0)
    # o_mix:   half-S half-R -> heterophilic (dev < 0)
    rows = []
    for abx in ["a0", "a1", "a2"]:
        for p in ["h1", "h2", "h3", "h4"]:
            rows.append((p, "o_clust", abx, 1))
    for p in ["x1", "x2", "x3", "x4"]:
        rows.append((p, "o_mix", "a3", 0))
        rows.append((p, "o_mix", "a4", 1))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    score = _patient_homophily(df)
    assert (score.loc[["h1", "h2", "h3", "h4"]] > 0).all()          # homophilic
    assert (score.loc[["x1", "x2", "x3", "x4"]] < 0).all()          # heterophilic
    assert score.index.is_unique and not score.isna().any()


from amr_fed.partition import homophily_split


def test_homophily_split_separates_spectrum():
    rows = []
    for abx in ["a0", "a1", "a2"]:
        for p in ["h1", "h2", "h3", "h4"]:
            rows.append((p, "o_clust", abx, 1))
    for p in ["x1", "x2", "x3", "x4"]:
        rows.append((p, "o_mix", "a3", 0))
        rows.append((p, "o_mix", "a4", 1))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    a = homophily_split(df, n_clients=2)
    assert a.index.is_unique and len(a) == 8
    assert set(a.unique()) == {0, 1}
    # hospital 0 = heterophilic patients, hospital 1 = homophilic patients
    assert set(a.index[a == 0]) == {"x1", "x2", "x3", "x4"}
    assert set(a.index[a == 1]) == {"h1", "h2", "h3", "h4"}
    assert homophily_split(df, n_clients=2).equals(a)          # deterministic


from amr_fed.partition import _patient_hubness, degree_skew_split


def test_degree_skew_split_sparse_vs_hub():
    rows = []
    for p in ["c1", "c2", "c3", "c4"]:                       # common-bug patients
        for abx in ["a0", "a1", "a2", "a3"]:
            rows.append((p, "o_common", abx, 0))
    for p in ["r1", "r2", "r3", "r4"]:                       # rare-bug patients
        rows.append((p, "o_rare", "a0", 0))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    hub = _patient_hubness(df)
    # breadth = log1p(# distinct antibiotics); common patients see a0..a3 (4), rare see a0 (1)
    assert np.isclose(hub.loc[["c1", "c2", "c3", "c4"]], np.log1p(4)).all()
    assert np.isclose(hub.loc[["r1", "r2", "r3", "r4"]], np.log1p(1)).all()
    a = degree_skew_split(df, n_clients=2)
    assert set(a.index[a == 0]) == {"r1", "r2", "r3", "r4"}  # sparse -> hospital 0
    assert set(a.index[a == 1]) == {"c1", "c2", "c3", "c4"}  # hubs -> hospital 1
    assert degree_skew_split(df, n_clients=2).equals(a)      # deterministic


def test_hubness_tracks_breadth_not_max_org_degree():
    # No-saturation regression: o_max dominates EVERY tested edge, so the old
    # triple-weighted mean (== o_max's tested-degree == 20) scores every patient the
    # same -> the degree-skew dial plateaus and hospitals are indistinguishable.
    # Breadth must separate: P_b (22 distinct antibiotics) > P_a (20 distinct).
    rows = []
    for i in range(20):
        rows.append(("P_a", "o_max", f"a{i}", 0))   # P_a grows only o_max (20 triples)
    for i in range(20):
        rows.append(("P_b", "o_max", f"a{i}", 0))   # o_max's 20 antibiotics ...
    rows.append(("P_b", "o_extra", "a20", 0))       # ... plus 2 more from o_extra
    rows.append(("P_b", "o_extra", "a21", 0))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    hub = _patient_hubness(df)
    assert np.isclose(hub["P_a"], np.log1p(20))
    assert np.isclose(hub["P_b"], np.log1p(22))
    assert hub["P_b"] > hub["P_a"]          # old weighted-mean gave P_b < P_a (saturated)
    assert hub.index.is_unique and not hub.isna().any()
    # the dial separates hospitals on a max-degree-dominated frame: span >> epsilon
    a = degree_skew_split(df, n_clients=2)
    assert set(a.index[a == 0]) == {"P_a"}   # narrow repertoire -> hospital 0
    assert set(a.index[a == 1]) == {"P_b"}   # broad repertoire -> hospital 1
    assert degree_skew_split(df, n_clients=2).equals(a)
    d = df.copy()
    d["client"] = d[PK].map(a)
    d["score"] = d[PK].map(hub)
    mean_score = d.groupby("client")["score"].mean()
    assert mean_score.iloc[-1] - mean_score.iloc[0] > 1e-3


if __name__ == "__main__":
    test_assign_home_ward_priority()
    test_apportion_sums_to_n()
    test_dirichlet_assigns_every_patient_once()
    test_rank_bucket_split_quantile_dial()
    test_rank_bucket_split_guards()
    test_patient_homophily_clustered_vs_scattered()
    test_homophily_split_separates_spectrum()
    test_degree_skew_split_sparse_vs_hub()
    test_hubness_tracks_breadth_not_max_org_degree()
    print("OK: partition unit tests passed.")
