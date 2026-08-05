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


if __name__ == "__main__":
    test_assign_home_ward_priority()
    test_apportion_sums_to_n()
    test_dirichlet_assigns_every_patient_once()
    print("OK: partition unit tests passed.")
