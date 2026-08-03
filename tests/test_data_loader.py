"""Fast, offline unit tests for the pure transforms in data_loader.

No Drive / no real CSVs needed — synthetic frames exercise the row filter,
binary label mapping, and ward-priority collapse. Run: python tests/test_data_loader.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_fed import config
from amr_fed.data_loader import CK, LABEL, ORG, WP, _collapse_ward, _filter_and_label


def test_filter_and_label():
    df = pd.DataFrame({
        WP:    [1, 1, 1, 0, 1, 1],
        LABEL: ["Susceptible", "Resistant", "Intermediate", "Null", "Inconclusive", "Synergism"],
        ORG:   ["E.coli", "E.coli", "K.pneu", "Null", "E.coli", "E.coli"],
    })
    out = _filter_and_label(df)
    # negative culture (was_positive=0) + Inconclusive/Synergism dropped -> only S/I/R kept
    assert len(out) == 3, out
    assert "Null" not in out[ORG].values
    # R and I -> 1, S -> 0
    labels = dict(zip(out[LABEL], out["label"]))
    assert labels == {"Susceptible": 0, "Resistant": 1, "Intermediate": 1}, labels


def test_collapse_ward_priority():
    fc = config.WARD_FLAG_COLUMNS
    ward = pd.DataFrame({
        CK:          [1, 2, 3, 4],
        fc["ICU"]:   [1, 0, 0, 0],
        fc["ER"]:    [1, 1, 0, 0],   # row 1 has ICU+ER -> ICU wins
        fc["IP"]:    [0, 0, 1, 0],
        fc["OP"]:    [0, 0, 1, 0],   # row 3 has IP+OP -> IP wins
    })
    out = _collapse_ward(ward).set_index(CK)["ward"].to_dict()
    assert out == {1: "ICU", 2: "ER", 3: "IP", 4: "NONE"}, out


if __name__ == "__main__":
    test_filter_and_label()
    test_collapse_ward_priority()
    print("OK: data_loader unit tests passed.")
