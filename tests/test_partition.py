"""Offline unit tests for partition.py pure logic. Run: python tests/test_partition.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.data_loader import PK
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


if __name__ == "__main__":
    test_assign_home_ward_priority()
    test_apportion_sums_to_n()
    test_dirichlet_assigns_every_patient_once()
    test_rank_bucket_split_quantile_dial()
    test_rank_bucket_split_guards()
    print("OK: partition unit tests passed.")
