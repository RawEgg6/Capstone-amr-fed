"""Offline tests for final_protocol.choose_temperature and the analysis helpers (no torch / data):
    python tests/test_final_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from amr_fed.federated import analysis as A
from amr_fed.federated import final_protocol as P


def _run(f1, val=None, n=3, w=None, size=(100, 50, 20), local=None, ft=None):
    r = {"phc_f1": f1, "phc_worst": f1 - 0.02, "val_fed": f1 - 0.001, "val_worst_fed": f1 - 0.02,
         "phc_auroc": 0.84, "phc_round": 8, "phc_val_score": f1 if val is None else val,
         "phc_per_client": [f1 + 0.01 * i for i in range(n)], "sizes": list(size),
         "agg_weights": w}
    if local is not None:
        r.update(val_local=local, val_worst_local=local - 0.02, val_local_auroc=0.83,
                 val_local_per_client=[local] * n)
    if ft is not None:
        r.update(ft_f1=ft, ft_worst=ft - 0.02, finetune={"per_client": [ft] * n})
    return r


def _arm(f1s, seeds=(42, 43, 44, 45, 46), **kw):
    return {"seeds": list(seeds), "runs": [_run(f, **{k: (v[i] if isinstance(v, list) else v)
                                                      for k, v in kw.items()}) for i, f in enumerate(f1s)]}


def _res():
    rng = np.random.default_rng(0)
    base = 0.70 + rng.normal(0, 0.002, 5)
    eye = np.eye(3) * 0.7 + 0.1          # arrays (not lists) are passed through unchanged
    res = {
        "fedavg": _arm(list(base), local=list(base - 0.003), ft=list(base + 0.002)),
        "v3_T0.25": _arm(list(base + 0.006), val=0.700),
        "v3_T0.5": _arm(list(base + 0.008 + rng.normal(0, 0.001, 5)), val=0.710, w=eye),
        "v3_T1": _arm(list(base + 0.004), val=0.705),
        "v3_shuffled": _arm(list(base + 0.001), w=eye),
        "uniform": _arm(list(base)),
        "topo": _arm(list(base + 0.001), w=np.array([0.2, 0.3, 0.5])),
        "_meta": {},
    }
    return res


def test_choose_temperature_uses_validation_only() -> None:
    res = _res()
    T, scores = P.choose_temperature(res)
    assert T == 0.5 and scores[0.5] > scores[0.25] > 0, (T, scores)
    # the test scores favour T=0.25 here, yet validation decides
    assert np.mean([r["phc_f1"] for r in res["v3_T0.25"]["runs"]]) < np.mean([r["phc_f1"] for r in res["v3_T0.5"]["runs"]])
    res2 = {k: v for k, v in res.items() if k != "v3_T1"}
    try:
        P.choose_temperature(res2)
        raise AssertionError("expected KeyError when a temperature is missing")
    except KeyError:
        pass
    print("OK: temperature chosen on validation, needs the full grid")


def test_stage1_specs() -> None:
    lo = [k for k, _, _ in P.stage1_specs("louvain", include_trimmed=False)]
    og = [k for k, _, _ in P.stage1_specs("org10", include_trimmed=True)]
    assert lo[0] == "fedavg" and "fedtrimmedavg" not in lo and "fedtrimmedavg" in og
    assert len(lo) == 8 and len(og) == 9, (len(lo), len(og))       # + v3_shuffled = 9 / 10 per seed
    assert not {"topo_shuffled", "invsize"} & set(lo)
    assert lo[1:4] == ["v3_T0.25", "v3_T0.5", "v3_T1"]
    fa = dict((k, e) for k, _, e in P.stage1_specs("x", False))
    assert fa["fedavg"]["compute_local"] and fa["fedavg"]["finetune_epochs"] == P.FT_EPOCHS
    print("OK: arm list has the pruned size (9 federated runs per seed incl. v3-shuffled)")


def test_analysis() -> None:
    res = _res()
    T, scores = P.choose_temperature(res)
    res["_meta"]["chosen_T"] = T
    assert A.resolve(res, "v3") == "v3_T0.5"
    d = A.paired(res, "v3", "local")
    assert d["n"] == 5 and d["mean"] > 0 and d["lo"] < d["mean"] < d["hi"] and d["n_pos"] == 5
    assert A.have(res, "v3") and A.have(res, "local") and A.have(res, "fedavg_ft") and not A.have(res, "invsize")
    out = A.summary_table(res) + A.comparison_report(res) + A.verdict(res)
    assert "PRE-REGISTERED VERDICT" in A.verdict(res) and "Local-only" in out
    tab = A.per_hospital_table(res, rank_by_size=True)
    assert "hospitals beating local-only" in tab and "#1" in tab
    assert "self-weights" in A.weights_report(res) and "weights" in A.weights_report(res)
    # incomplete results do not crash the verdict
    assert A.verdict({"fedavg": res["fedavg"]}).startswith("INCOMPLETE")
    print("OK: analysis helpers (paired stats, tables, verdict) run on a synthetic result")


if __name__ == "__main__":
    test_choose_temperature_uses_validation_only()
    test_stage1_specs()
    test_analysis()
    print("OK: final-analysis tests passed")
