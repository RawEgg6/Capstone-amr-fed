"""Fast, offline unit tests for graph_build's pure helpers.

The full build_arrays + to_hetero_data path is verified on real data / Colab; these
guard the non-trivial pure logic. Run: python tests/test_graph_build.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed.graph_build import (
    ABX, CDESC, COM, ORG, TK, _age_to_ordinal, _aggregate_labvital_to_patient,
    _build_comorbidity_arrays, _build_prior_exposure_edges, _history_raw, _medication_to_node,
    _patient_grouped_split, _prescription_features, _smoothed_rate, _specimen_features,
)
from amr_fed.data_loader import CK, PK


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


def test_build_comorbidity_arrays():
    pat_map = {"pA": 0, "pB": 1, "pC": 2}
    # train rates: pA all-resistant (1.0), pB all-susceptible (0.0); pC is val/test (absent)
    pat_rate = pd.Series({0: 1.0, 1: 0.0})
    edges = pd.DataFrame({
        PK:  ["pA", "pB", "pC", "pA", "pZ"],       # pZ not in pat_map -> dropped
        COM: ["diab", "diab", "htn", "htn", "diab"],
    })
    names, x, ei = _build_comorbidity_arrays(edges, pat_map, pat_rate, global_rate=0.2)
    assert list(names) == ["diab", "htn"], names
    assert ei.shape == (2, 4), ei.shape          # pZ edge dropped -> 4 edges, not 5
    assert x.shape == (2, 2) and np.isfinite(x).all()
    # diab (pA=1.0, pB=0.0 -> mean 0.5) should carry higher resistance signal than
    # htn (pA=1.0 in train, pC absent -> ~1.0) ... both finite; just assert ordering sane
    assert set(ei[1].tolist()) == {0, 1}


def test_medication_to_node_mapping():
    abx_map = {"Ciprofloxacin": 0, "Levofloxacin": 1, "Trimethoprim/Sulfamethoxazole": 2,
               "Vancomycin": 3, "Nitrofurantoin": 4}
    m2n = _medication_to_node(
        ["Ciprofloxacin Hcl", "Levofloxacin In", "Vancomycin In Dextrose",
         "Cipro", "Bactrim Ds", "Macrobid", "Sulfamethoxazole-Trimethoprim",
         "Zithromax", "Rifaximin"],
        abx_map)
    assert m2n["Ciprofloxacin Hcl"] == "Ciprofloxacin"      # salt stripped
    assert m2n["Levofloxacin In"] == "Levofloxacin"          # formulation stripped
    assert m2n["Vancomycin In Dextrose"] == "Vancomycin"
    assert m2n["Cipro"] == "Ciprofloxacin"                   # brand alias
    assert m2n["Bactrim Ds"] == "Trimethoprim/Sulfamethoxazole"
    assert m2n["Macrobid"] == "Nitrofurantoin"
    assert m2n["Sulfamethoxazole-Trimethoprim"] == "Trimethoprim/Sulfamethoxazole"  # combo stem
    assert m2n["Zithromax"] is None and m2n["Rifaximin"] is None  # not tested nodes -> drop


def test_build_prior_exposure_edges():
    pat_map = {"pA": 0, "pB": 1}
    abx_map = {"Ciprofloxacin": 0, "Vancomycin": 1}
    exp = pd.DataFrame({
        PK: ["pA", "pA", "pB", "pZ"],                        # pZ not in cohort -> dropped
        "medication_name": ["Ciprofloxacin Hcl", "Vancomycin", "Cipro", "Vancomycin"],
    })
    ei = _build_prior_exposure_edges(exp, pat_map, abx_map)
    edges = set(map(tuple, ei.T.tolist()))
    assert edges == {(0, 0), (0, 1), (1, 0)}, edges          # pA->cipro,vanco ; pB->cipro
    assert ei.dtype.kind == "i"


def test_aggregate_labvital_to_patient():
    import numpy as np
    patients = np.array(["pA", "pB", "pC"])          # pC has no lab/vital rows
    cp = pd.DataFrame({CK: [1, 2, 3], PK: ["pA", "pA", "pB"]})  # pA has 2 cultures
    lv = pd.DataFrame({CK: [1, 2], "labs_median_wbc": [10.0, 12.0], "vitals_median_hr": [80.0, np.nan]})
    names, x = _aggregate_labvital_to_patient(cp, lv, patients)
    assert names[-2:] == ["labs_measured", "vitals_measured"]
    assert x.shape == (3, len(names)) and np.isfinite(x).all()
    meas = dict(zip(names, range(len(names))))
    # pA measured labs+vitals=1 ; pC (no rows) measured=0
    assert x[0, meas["labs_measured"]] == 1.0 and x[2, meas["labs_measured"]] == 0.0
    assert x[2, meas["vitals_measured"]] == 0.0


def test_history_raw_is_leakage_safe():
    # patient pA: 3 cultures in time order with labels [1, 0, 1]
    df = pd.DataFrame({
        PK:  ["pA", "pA", "pA"],
        ORG: ["E", "E", "E"],
        ABX: ["Cip", "Cip", "Cip"],
        TK:  ["2020-01-01", "2020-01-02", "2020-01-03"],
        "label": [1, 0, 1],
    })
    raw = _history_raw(df, global_rate=0.2, alpha=10.0)
    # col0 = log1p(prior count): 0, 1, 2 cultures before
    assert np.allclose(raw[:, 0], [np.log1p(0), np.log1p(1), np.log1p(2)])
    # col1 = prior resistance rate, EB-shrunk, CURRENT label excluded:
    #   row0: no prior -> pure global 0.2
    #   row1: 1 prior (label 1) -> (1 + 10*0.2)/(1+10) = 3/11
    #   row2: 2 prior (labels 1,0 -> sum 1) -> (1 + 2)/(2+10) = 3/12
    # If it leaked the current/future label, row0 would NOT equal 0.2.
    assert np.allclose(raw[:, 1], [0.2, 3 / 11, 3 / 12], atol=1e-6), raw[:, 1]


def test_specimen_features_fixed_onehot():
    df = pd.DataFrame({CDESC: ["URINE", "respiratory", "BLOOD", "URINE"]})
    x = _specimen_features(df)
    assert x.shape == (4, 3)                       # fixed 3-way one-hot regardless of subset
    assert x.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]  # case-insensitive


def test_prescription_features():
    df = pd.DataFrame({CK: [1, 2, 3]})                 # CK=3 has no prior exposure
    agg = pd.DataFrame({CK: [1, 2], "presc_n": [3, 1],
                        "presc_nclass": [2, 1], "presc_recency": [5.0, 30.0]})
    x = _prescription_features(df, agg)
    assert x.shape == (3, 4) and np.isfinite(x).all()
    assert x[:, -1].tolist() == [1.0, 1.0, 0.0]        # has-prior-exposure flag


if __name__ == "__main__":
    test_age_to_ordinal()
    test_smoothed_rate_shrinks_low_counts()
    test_patient_grouped_split_disjoint_and_deterministic()
    test_build_comorbidity_arrays()
    test_medication_to_node_mapping()
    test_build_prior_exposure_edges()
    test_aggregate_labvital_to_patient()
    test_history_raw_is_leakage_safe()
    test_specimen_features_fixed_onehot()
    test_prescription_features()
    print("OK: graph_build unit tests passed.")
