"""Fast, offline unit tests for graph_build's pure helpers.

The full build_arrays + to_hetero_data path is verified on real data / Colab; these
guard the non-trivial pure logic. Run: python tests/test_graph_build.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.graph_build import _age_to_ordinal, _patient_grouped_split, _smoothed_rate


def test_age_to_ordinal():
    s = pd.Series(["18-24 years", "65-74 years", "above 90", None])
    out = _age_to_ordinal(s)
    assert out.tolist()[:3] == [18.0, 65.0, 90.0]
    assert pd.isna(out.iloc[3])  # unknown -> NaN (imputed later)


def test_smoothed_rate_shrinks_low_counts():
    prior = 0.2
    # high count: 90/100 stays near 0.9; low count: 1/1 pulled toward prior
    r = _smoothed_rate(pd.Series([90.0, 1.0]), pd.Series([100, 1]), prior, alpha=20.0)
    assert abs(r.iloc[0] - (90 + 20 * 0.2) / (100 + 20)) < 1e-9
    assert prior < r.iloc[1] < 1.0            # shrunk down from 1.0, still above prior
    assert r.iloc[1] < r.iloc[0]


def test_patient_grouped_split_disjoint_and_deterministic():
    pats = np.repeat(np.arange(100), 5)       # 100 patients, 5 tests each
    code = _patient_grouped_split(pats, seed=42)
    code2 = _patient_grouped_split(pats, seed=42)
    assert code == code2                       # deterministic under same seed
    groups = {0: set(), 1: set(), 2: set()}
    for p, c in code.items():
        groups[c].add(p)
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert sum(len(g) for g in groups.values()) == 100
    assert len(groups[0]) == 70               # 70/15/15 of 100


if __name__ == "__main__":
    test_age_to_ordinal()
    test_smoothed_rate_shrinks_low_counts()
    test_patient_grouped_split_disjoint_and_deterministic()
    print("OK: graph_build unit tests passed.")
