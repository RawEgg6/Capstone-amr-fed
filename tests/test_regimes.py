"""Offline tests for the data regimes of notebook 18 (pandas/numpy only):  python tests/test_regimes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amr_fed.data_loader import PK
from amr_fed.federated import analysis as A
from amr_fed.federated import regimes as R
from amr_fed.graph_build import _patient_grouped_split


def _frame(n_pat=400, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"p{i}" for i in range(n_pat)])
    rows = rng.integers(0, n_pat, 3000)
    return pd.DataFrame({PK: ids[rows], "x": rng.random(3000)})


def _kinds(sub, code):
    return pd.Series(sub[PK].map(code).to_numpy()).value_counts().to_dict(), \
           {c: set(sub.loc[sub[PK].map(code) == c, PK]) for c in (0, 1, 2)}


def test_subsample_keeps_test_and_scales_train() -> None:
    df = _frame()
    full = _patient_grouped_split(df[PK].to_numpy(), 42)
    test_full = {p for p, c in full.items() if c == 2}
    n_tr = sum(c == 0 for c in full.values())
    sizes = []
    for f in (0.5, 0.1, 0.03):
        sub, code = R.subsample_frame(df, seed=42, cid=3, frac=f)
        _, sets = _kinds(sub, code)
        assert sets[2] == test_full, "test patients must be identical at every data level"
        assert len(sets[0]) == max(2, round(f * n_tr)) and len(sets[1]) >= 2
        assert set(code.values()) <= {0, 1, 2} and set(sub[PK]) == sets[0] | sets[1] | sets[2]
        assert sets[0] <= {p for p, c in full.items() if c == 0}, "kept train patients must be original train patients"
        sizes.append(len(sets[0]))
    assert sizes[0] > sizes[1] > sizes[2]
    again, _ = R.subsample_frame(df, seed=42, cid=3, frac=0.1)
    again2, _ = R.subsample_frame(df, seed=42, cid=3, frac=0.1)
    assert again.index.equals(again2.index), "deterministic in (seed, cid, frac)"
    other, _ = R.subsample_frame(df, seed=42, cid=4, frac=0.1)
    assert not other.index.equals(again.index), "different hospitals draw different patients"
    try:
        R.subsample_frame(df, 42, 0, 0.0)
        raise AssertionError("frac=0 must raise")
    except ValueError:
        pass
    print("OK: subsample keeps the test patients, scales train/val, is deterministic")


def test_slice_frame() -> None:
    df = _frame(800)
    full = _patient_grouped_split(df[PK].to_numpy(), 7)
    n_va_full = sum(c == 1 for c in full.values())
    for k in (25, 100):
        sub, code = R.slice_frame(df, seed=7, n_train=k)
        _, sets = _kinds(sub, code)
        assert len(sets[0]) == k and len(sets[1]) == min(max(30, k // 2), n_va_full)
        assert sets[2] == {p for p, c in full.items() if c == 2}
    sub, code = R.slice_frame(df, seed=7, n_train=None)
    assert len(sub) == len(df), "k=None keeps the whole hospital"
    big, _ = R.slice_frame(df, seed=7, n_train=10 ** 6)
    assert len(big) == len(df)
    print("OK: slice_frame gives k train patients, a proportionate validation slice, and the full test set")


def test_holdout_rule() -> None:
    sizes = [50, 40, 30, 20, 10, 9, 8, 7, 6, 5]
    assign = pd.Series(np.repeat(np.arange(10), sizes), index=[f"p{i}" for i in range(sum(sizes))])
    assert R.holdout_hospital(assign, "median") == 4      # 5th largest
    assert R.holdout_hospital(assign, "smallest") == 9
    print("OK: hold-out rules pick the 5th largest / smallest hospital")


def _fake_scarcity(f, gap, seed_n=5, n=4):
    rng = np.random.default_rng(int(f * 100))
    runs = []
    for s in range(seed_n):
        base = 0.65 + rng.normal(0, 0.003, n)
        runs.append({"val_local_per_client": list(base), "phc_per_client": list(base + gap + rng.normal(0, 0.002, n)),
                     "finetune": {"per_client": list(base + gap + 0.005)}, "n_triples": [{"train": 300 * (i + 1), "test": 100 * (i + 1)} for i in range(n)],
                     "assignment_sizes": [200, 100, 50, 25]})
    return {"seeds": list(range(seed_n)), "runs": runs}


def test_analysis_reports() -> None:
    res = {"scarcity_f1": _fake_scarcity(1.0, 0.0), "scarcity_f0.25": _fake_scarcity(0.25, 0.02), "_meta": {}}
    txt = A.scarcity_report(res)
    assert "FedAvg - local-only" in txt and txt.index("data kept = 1") < txt.index("data kept = 0.25")
    v = A.scarcity_values(res, "scarcity_f0.25")
    assert abs(np.mean(v["fedavg"] - v["local"]) - 0.02) < 0.004
    runs = []
    for s in range(5):
        k = {m: {"f1": f, "thr": 0.5, "val": f, "epoch": 0, "auroc": 0.8, "n_val": 10, "n_test": 100}
             for m, f in (("zero_shot", 0.60), ("finetune", 0.66), ("scratch", 0.55 + 0.001 * s))}
        k["n_train_patients"] = 25
        runs.append({"seed": 42 + s, "rule": "median", "holdout_patients": 1000, "ks": {"25": k, "all": k}})
    ns = A.newsite_report({"newsite_median": {"seeds": list(range(5)), "runs": runs}, "_meta": {}})
    assert "fine-tune - scratch" in ns and "k=25" in ns
    print("OK: scarcity / new-site reports run on synthetic results")


if __name__ == "__main__":
    test_subsample_keeps_test_and_scales_train()
    test_slice_frame()
    test_holdout_rule()
    test_analysis_reports()
    print("OK: regime tests passed")
