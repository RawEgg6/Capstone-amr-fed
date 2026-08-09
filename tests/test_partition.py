"""Offline unit tests for partition.py pure logic. Run: python tests/test_partition.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.data_loader import PK, ORG, ABX
from amr_fed.partition import (
    WARD_COL, _apportion, _greedy_pack, assign_home_ward, dirichlet_ward_mixture,
    louvain_split, organism_community, _quadrant_assign, _residualize, topology_split,
)


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


def _corner_scores():
    """4 groups x 4 patients at extreme (hom, hub) corners -> cells 0..3.
    s = scattered+sparse, h = scattered+hub, c = clustered+sparse, b = clustered+hub."""
    idx = [f"{g}{i}" for g in "shcb" for i in range(4)]
    hom = pd.Series([-1.0] * 8 + [1.0] * 8, index=idx)          # s,h low ; c,b high
    hub = pd.Series([0.0] * 4 + [3.0] * 4 + [0.0] * 4 + [3.0] * 4, index=idx)
    return hom, hub


def test_quadrant_assign_corners_1_to_1():
    hom, hub = _corner_scores()
    a = _quadrant_assign(hom, hub, 4, 0.0, 42)
    assert a.index.is_unique and len(a) == 16
    # each extreme group lands exactly in its corner id (1:1)
    for group, cell in {"s": 0, "h": 1, "c": 2, "b": 3}.items():
        assert set(a.index[a == cell]) == {f"{group}{i}" for i in range(4)}
    # balanced: 4 patients per corner
    assert list(a.value_counts().sort_index()) == [4, 4, 4, 4]
    # deterministic
    assert _quadrant_assign(hom, hub, 4, 0.0, 42).equals(a)


def test_quadrant_assign_n8():
    hom, hub = _corner_scores()
    a = _quadrant_assign(hom, hub, 8, 0.0, 42)
    assert a.index.is_unique and len(a) == 16
    assert set(a.unique()) == set(range(8))
    assert list(a.value_counts().sort_index()) == [2] * 8        # 8 balanced hospitals
    # each corner's 4 patients split across hospitals {2*cell, 2*cell+1}
    for group, cell in {"s": 0, "h": 1, "c": 2, "b": 3}.items():
        assert set(a.loc[[f"{group}{i}" for i in range(4)]]) == {2 * cell, 2 * cell + 1}


def test_quadrant_assign_purity_deterministic():
    hom, hub = _corner_scores()
    clean = _quadrant_assign(hom, hub, 4, 0.0, 42)
    a1 = _quadrant_assign(hom, hub, 4, 1.0, 42)
    a1b = _quadrant_assign(hom, hub, 4, 1.0, 42)
    assert not a1.equals(clean)                   # full noise scrambles the corner labels
    assert a1.equals(a1b)                         # deterministic across same seed
    assert set(a1.unique()) <= set(range(4))      # ids stay in [0, n_clients)


def test_quadrant_assign_guards():
    hom, hub = _corner_scores()
    for bad_n in (3, 6):                          # <4 and not a multiple of 4
        try:
            _quadrant_assign(hom, hub, bad_n, 0.0, 42)
            raise AssertionError(f"n_clients={bad_n} should raise")
        except ValueError:
            pass
    try:
        _quadrant_assign(hom, hub, 4, 1.5, 42)
        raise AssertionError("purity=1.5 should raise")
    except ValueError:
        pass


def _quadrant_frame():
    """8 patients in the four (homophily, hubness) quadrants.
    o_clust edges all-resistant (dev +0.25); o_mix half/half (dev -0.35). Sparse patients
    see 1 distinct antibiotic (hub log1p(1)), hubs see 2 (hub log1p(2))."""
    rows = [
        # s = scattered + sparse (1 abx on mixed organism)
        ("s1", "o_mix", "aS1", 0), ("s2", "o_mix", "aS2", 0),
        # h = scattered + hub (2 abx on mixed organism)
        ("h1", "o_mix", "aH1", 0), ("h1", "o_mix", "aH2", 1),
        ("h2", "o_mix", "aH3", 1), ("h2", "o_mix", "aH4", 1),
        # c = clustered + sparse (1 abx on clustered organism)
        ("c1", "o_clust", "aC1", 1), ("c2", "o_clust", "aC2", 1),
        # b = clustered + hub (2 abx on clustered organism)
        ("b1", "o_clust", "aB1", 1), ("b1", "o_clust", "aB2", 1),
        ("b2", "o_clust", "aB3", 1), ("b2", "o_clust", "aB4", 1),
    ]
    return pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])


def test_topology_split_wrapper():
    df = _quadrant_frame()
    a = topology_split(df, n_clients=4)
    assert a.index.is_unique and len(a) == 8
    assert set(a.unique()) == {0, 1, 2, 3}
    for group, cell in {"s": 0, "h": 1, "c": 2, "b": 3}.items():
        assert set(a.index[a == cell]) == {f"{group}{i}" for i in (1, 2)}
    assert topology_split(df, n_clients=4).equals(a)            # deterministic


def test_residualize_decorrelates():
    # hub correlates with hom, so raw median-split b_sign tracks a_sign; residualizing
    # hub on hom removes the shared trend -> the decorrelated b_sign reorders patients.
    idx = [f"p{i}" for i in range(16)]
    hom = pd.Series(np.arange(16, dtype=float), index=idx)
    hub = pd.Series([1, 2, 3, 4, 5, 6, 8, 9, 4, 5, 6, 7, 8, 9, 11, 12], index=idx, dtype=float)
    raw = _quadrant_assign(hom, hub, 4, 0.0, 42)
    dec = _quadrant_assign(hom, _residualize(hub, hom), 4, 0.0, 42)
    raw_b = (raw % 2).to_numpy()      # hub quadrant iff hospital id is odd (m=1 -> cell)
    dec_b = (dec % 2).to_numpy()
    assert not np.array_equal(raw_b, dec_b)     # decorrelation changed the hub ordering
    assert _quadrant_assign(hom, _residualize(hub, hom), 4, 0.0, 42).equals(dec)


def _nx_or_skip():
    try:
        import networkx  # noqa: F401
        return True
    except ImportError:
        print("SKIP: louvain_split requires networkx")
        return False


def test_greedy_pack_balances_loads():
    counts = pd.Series({"c1": 100, "c2": 80, "c3": 50, "c4": 30, "c5": 10})
    assignment = _greedy_pack(counts, 3)
    assert set(assignment) == set(counts.index)              # every item assigned once
    assert all(0 <= v < 3 for v in assignment.values())
    # exact greedy trace: c1->0, c2->1, c3->2, c4->2, c5->1
    assert dict(assignment) == {"c1": 0, "c2": 1, "c3": 2, "c4": 2, "c5": 1}
    loads = [0, 0, 0]
    for item, size in counts.items():
        loads[assignment[item]] += size
    assert max(loads) - min(loads) <= int(counts.max())      # balanced within max item size


def test_organism_community_disjoint_bug_buckets():
    # two organisms -> each lands in its own hospital (greedy, largest first)
    df = pd.DataFrame({
        PK: ["pA1", "pA2", "pA3", "pB1", "pB2"],
        ORG: ["oA", "oA", "oA", "oB", "oB"],
        ABX: ["a0", "a1", "a2", "a3", "a4"],
    })
    a = organism_community(df, n_clients=2)
    assert a.index.is_unique and len(a) == 5               # 1:1, no leakage
    assert set(a.loc[["pA1", "pA2", "pA3"]]) == {0}         # oA patients -> hospital 0
    assert set(a.loc[["pB1", "pB2"]]) == {1}                # oB patients -> hospital 1
    assert organism_community(df, n_clients=2).equals(a)    # deterministic


def test_louvain_split_separates_disjoint_components():
    if not _nx_or_skip():
        return
    rows = []
    for p in ["pA1", "pA2", "pA3"]:                       # cluster A: orgs oA1,oA2 x aA1..aA3
        for a in ["aA1", "aA2", "aA3"]:
            rows.append((p, "oA1", a))
    for p in ["pA4", "pA5", "pA6"]:
        for a in ["aA1", "aA2", "aA3"]:
            rows.append((p, "oA2", a))
    for p in ["pB1", "pB2"]:                              # cluster B: orgs oB1,oB2 x aB1..aB3
        for a in ["aB1", "aB2", "aB3"]:
            rows.append((p, "oB1", a))
    for p in ["pB3", "pB4"]:
        for a in ["aB1", "aB2", "aB3"]:
            rows.append((p, "oB2", a))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX])
    a = louvain_split(df, n_clients=2)
    assert a.index.is_unique and len(a) == 10              # 1:1, no leakage
    hospA = set(a.loc[["pA1", "pA2", "pA3", "pA4", "pA5", "pA6"]])
    hospB = set(a.loc[["pB1", "pB2", "pB3", "pB4"]])
    assert len(hospA) == len(hospB) == 1                   # each cluster in a single hospital
    assert hospA & hospB == set()                           # disjoint hospitals
    assert louvain_split(df, n_clients=2).equals(a)         # deterministic


def test_partition_imports_without_networkx():
    """partition.py must import with NO top-level networkx (lazy, call-time import)."""
    import importlib
    mod = importlib.import_module("amr_fed.partition")
    assert callable(getattr(mod, "louvain_split", None))   # the lazy-import function exists
    assert not hasattr(mod, "nx")                          # networkx never a module attribute


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
    test_quadrant_assign_corners_1_to_1()
    test_quadrant_assign_n8()
    test_quadrant_assign_purity_deterministic()
    test_quadrant_assign_guards()
    test_topology_split_wrapper()
    test_residualize_decorrelates()
    test_greedy_pack_balances_loads()
    test_organism_community_disjoint_bug_buckets()
    test_louvain_split_separates_disjoint_components()
    test_partition_imports_without_networkx()
    print("OK: partition unit tests passed.")
