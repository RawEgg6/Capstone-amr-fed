"""Offline tests for the injected-shift experiments of notebook 22 (no torch): python tests/test_shift_runs.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amr_fed import config
from amr_fed.data_loader import ABX, ORG, PK
from amr_fed.federated import shift_runs as S

TK = config.KEYS["time"]


def _frame(n_pat=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        t = pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(rng.integers(0, 4000)))
        for a in ("a0", "a1", "a2", "a3"):
            rows.append({PK: f"p{p}", ORG: "o", ABX: a, TK: str(t), "label": int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_shift_inverts_late_era_only() -> None:
    df = _frame()
    from amr_fed.partition import patient_era
    late = df[PK].map(patient_era(df, 2)).to_numpy() == 1
    for frac, n_abx in ((0.25, 1), (0.5, 2), (1.0, 4)):
        s = S.shifted_df(df, frac)
        changed = (s["label"] != df["label"]).to_numpy()
        assert not changed[~late].any(), "early era must be untouched"
        assert set(s.loc[changed, ABX]) <= set(df[ABX]) and len(set(s.loc[changed, ABX])) == n_abx
        assert changed[late & df[ABX].isin(set(s.loc[changed, ABX])).to_numpy()].all()
    assert S.shifted_df(df, 0.0)["label"].equals(df["label"])
    assert S.shifted_df(df, 0.5).equals(S.shifted_df(df, 0.5)), "deterministic"
    print("OK: concept shift inverts only late-era rows of the chosen antibiotics")


def test_panel_split() -> None:
    df = _frame()
    p = S.panel_df(df, 0.5)
    grp = S._patient_group(p[PK])
    a0, a1 = set(p.loc[grp == 0, ABX]), set(p.loc[grp == 1, ABX])
    assert len(a0 & a1) == 2 and len(a0) == 3 and len(a1) == 3 and a0 != a1
    part = S.panel_partition(p, 42)
    assert part.index.is_unique and set(part.unique()) <= set(range(10))
    assert (part // 5).to_numpy().tolist() == S._patient_group(pd.Series(part.index)).tolist()
    assert part.equals(S.panel_partition(p, 42)) and not part.equals(S.panel_partition(p, 43))
    print("OK: panel split drops off-panel rows; groups follow the patient hash")


def test_build_and_arms() -> None:
    specs = S.arm_specs()
    assert set(specs) == set(S.ARMS) and all(a["finetune_epochs"] == 8 for a in specs.values())
    assert specs["fedavg"]["compute_local"] is False
    df = _frame(100)
    for e in S.EXPERIMENTS:
        d, fn = S.build(e, df)
        a = fn(d, 42)
        assert a.index.is_unique and len(a) == d[PK].nunique(), e
    g = S.client_groups("shift100", df, 42)
    assert sorted(set(g)) == [0, 1] and len(g) == 10
    print("OK: every experiment builds a 10-hospital split; arms budget-matched")


if __name__ == "__main__":
    test_shift_inverts_late_era_only()
    test_panel_split()
    test_build_and_arms()
    print("OK: shift_runs tests passed")
