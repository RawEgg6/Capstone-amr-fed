"""Phase-1 data loader: one clean per-test frame for a single hospital.

Loads the ARMD cohort, applies the locked row filter + binary label, and joins
the three cheap 1:1-per-culture context tables (demographics, ADI, ward). The
fan-out tables (comorbidity, labs, vitals, procedures, resistance, abx-exposure)
are NOT loaded here — they become edges / edge-features in graph_build.

All schema constants come from config.py; nothing is hardcoded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --- schema pulled from config (single source of truth) ---
PK, CK = config.KEYS["patient"], config.KEYS["culture"]
TK = config.KEYS["time"]
ORG, ABX = config.COLUMNS["organism"], config.COLUMNS["antibiotic"]
WP = config.COLUMNS["was_positive"]
CD = config.COLUMNS["culture_description"]
AGE, GENDER = config.COLUMNS["age"], config.COLUMNS["gender"]
ADI = config.COLUMNS["adi_score"]
LABEL = config.LABEL_COLUMN


def _path(name: str) -> Path:
    return Path(config.DATA_DIR) / config.ARMD_TABLES[name]


def _filter_and_label(cohort: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: drop non-outcome rows, add the binary `label` column.

    Keeps `was_positive == 1` AND susceptibility in config.VALID_LABELS.
    label = 1 for Resistant/Intermediate, 0 for Susceptible.
    """
    keep = (cohort[WP] == 1) & cohort[LABEL].isin(config.VALID_LABELS)
    out = cohort.loc[keep].copy()
    out["label"] = out[LABEL].isin(config.RESISTANT_LABELS).astype("int8")
    return out


def _collapse_ward(ward: pd.DataFrame) -> pd.DataFrame:
    """Collapse the overlapping ward flag columns to one `ward` value per culture.

    Priority (sickest setting wins) from config.WARD_PRIORITY; no flag -> "NONE".
    """
    conds = [ward[config.WARD_FLAG_COLUMNS[w]] == 1 for w in config.WARD_PRIORITY]
    ward = ward.copy()
    ward["ward"] = np.select(conds, config.WARD_PRIORITY, default="NONE")
    return ward[[CK, "ward"]]


def load_cohort_frame(ward: str | None = None, sample_n: int | None = None) -> pd.DataFrame:
    """Return the filtered, labelled, context-joined per-test frame.

    Grain: one row per (culture, organism, antibiotic) susceptibility result.

    Args:
        ward: if given (one of config.WARD_PARTITIONS), restrict to that hospital.
        sample_n: if given, read only the first N cohort rows (quick dry runs).
    """
    if ward is not None and ward not in config.WARD_PARTITIONS:
        raise ValueError(f"ward must be one of {config.WARD_PARTITIONS} or None, got {ward!r}")

    cohort = pd.read_csv(
        _path("cohort"),
        usecols=[PK, CK, TK, ORG, ABX, LABEL, WP, CD, "ordering_mode"],
        nrows=sample_n,
        low_memory=False,  # else mixed-dtype cols trip a chunked-read bug (IndexError)
    )
    df = _filter_and_label(cohort)

    # --- join the three cheap 1:1-per-culture context tables on the culture key ---
    demo = pd.read_csv(_path("demographics"), usecols=[CK, AGE, GENDER], low_memory=False).drop_duplicates(CK)
    adi = pd.read_csv(_path("adi"), usecols=[CK, ADI], low_memory=False).drop_duplicates(CK)
    adi[ADI] = pd.to_numeric(adi[ADI], errors="coerce")  # pandas 3.0 reads it as arrow string
    wcols = [config.WARD_FLAG_COLUMNS[w] for w in config.WARD_PRIORITY]
    ward_tbl = _collapse_ward(pd.read_csv(_path("ward"), usecols=[CK] + wcols, low_memory=False))

    for aux in (demo, adi, ward_tbl):
        df = df.merge(aux, on=CK, how="left")

    if ward is not None:
        df = df[df["ward"] == ward]

    return df.reset_index(drop=True)


def _self_check() -> None:
    """Real-data sanity check (see docs/2026-07-19-next-steps.md, task 1)."""
    df = load_cohort_frame()
    n, n_org, n_abx = len(df), df[ORG].nunique(), df[ABX].nunique()
    n_null = int((df[ORG].astype(str).str.lower() == "null").sum())
    pos_rate = round(df["label"].mean(), 3)
    adi_cov = round(df[ADI].notna().mean(), 3)  # 'Null' sentinels -> NaN, so this is real coverage
    print(f"rows={n:,} | organisms={n_org} | antibiotics={n_abx} | 'Null' organisms={n_null}")
    print(f"label balance (positive rate)={pos_rate} | classes={df['label'].value_counts().to_dict()}")
    print(f"adi_score numeric coverage={adi_cov} (rest are 'Null' sentinels -> impute + measured-flag later)")
    print(f"columns={list(df.columns)}")
    print("ward sizes:", df["ward"].value_counts(dropna=False).to_dict())

    assert 1_590_000 <= n <= 1_610_000, f"row count {n:,} outside expected ~1.60M"
    assert n_org >= 300 and n_abx >= 50, f"cardinality too low: {n_org} org / {n_abx} abx"
    assert n_null == 0, "negative-culture 'Null' organisms survived the filter"
    assert set(df["label"].unique()) <= {0, 1}, "label is not binary"
    assert adi_cov > 0.7, f"adi_score numeric coverage {adi_cov} unexpectedly low"
    print("OK: data_loader self-check passed.")


if __name__ == "__main__":
    _self_check()
