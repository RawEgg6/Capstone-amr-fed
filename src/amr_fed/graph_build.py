"""Phase-1 graph builder: one hospital's per-test frame -> heterogeneous KG.

Target formulation (locked): per-test **triple** classification. The 3,848 unique
(organism, antibiotic) pairs are the `tested` message-passing edges; supervision is
the ~1.6M individual (patient, organism, antibiotic) test triples, each with a binary
label. A decoder later scores (h_patient, h_organism, h_antibiotic) -> P(resistant).

Two layers, on purpose:
  * build_arrays()   -- pure pandas/numpy (node maps, features, edges, split). No torch,
                        so it runs + is testable on any stack (incl. numpy 2 / Intel Mac).
  * to_hetero_data() -- lazy-imports torch + PyG, packs arrays into a HeteroData with
                        reverse edges. Run this on Colab (modern torch).

Core 3 edges only. Enrichment edges (procedure, comorbidity, prior_exposure) come later.
Leakage rule: structural counts/degrees may use all observed edges; any feature that
touches the LABEL (resistance rates) is computed from TRAIN triples only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import ABX, ADI, CK, ORG, PK, load_cohort_frame

# age bins are ordinal; rank them by their leading integer ("above 90" -> 90)
def _age_to_ordinal(age: pd.Series) -> pd.Series:
    lead = age.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(lead, errors="coerce")


def _smoothed_rate(n_pos: pd.Series, n_tot: pd.Series, prior: float, alpha: float = 20.0) -> pd.Series:
    """Empirical-Bayes shrink toward the global prior for low-count nodes."""
    return (n_pos + alpha * prior) / (n_tot + alpha)


def _zscore(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def _patient_grouped_split(patients: np.ndarray, seed: int, fracs=(0.70, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Assign each unique patient to train/val/test; return per-patient split code {0,1,2}."""
    uniq = np.unique(patients)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    code = {}
    for i, p in enumerate(uniq):
        code[p] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return code


def _load_known_resistant(org_map: dict, abx_map: dict) -> np.ndarray:
    """(organism, known_resistant, antibiotic) edges from the resistance prior table,
    filtered to organisms/antibiotics that exist as nodes. Returns [2, E] int array."""
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES["resistance"]
    res = pd.read_csv(fp, usecols=[ORG, ABX], low_memory=False).dropna().drop_duplicates()
    res = res[res[ORG].isin(org_map) & res[ABX].isin(abx_map)]
    if len(res) == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = res[ORG].map(org_map).to_numpy()
    dst = res[ABX].map(abx_map).to_numpy()
    return np.vstack([src, dst]).astype(np.int64)


def build_arrays(df: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Pure pandas/numpy build. `df` = a data_loader.load_cohort_frame() result.

    Returns a dict of node maps, float feature matrices, edge_index arrays, the
    supervision triples/labels, and the patient-grouped split codes.
    """
    # --- node index maps (string/id -> contiguous int) ---
    organisms = np.sort(df[ORG].unique())
    antibiotics = np.sort(df[ABX].unique())
    patients = np.sort(df[PK].unique())
    org_map = {v: i for i, v in enumerate(organisms)}
    abx_map = {v: i for i, v in enumerate(antibiotics)}
    pat_map = {v: i for i, v in enumerate(patients)}

    o_idx = df[ORG].map(org_map).to_numpy()
    a_idx = df[ABX].map(abx_map).to_numpy()
    p_idx = df[PK].map(pat_map).to_numpy()
    y = df["label"].to_numpy().astype(np.float32)

    # --- patient-grouped split (a patient's tests all land in one split) ---
    code = _patient_grouped_split(df[PK].to_numpy(), seed)
    split = df[PK].map(code).to_numpy()          # per-triple split code {0,1,2}
    is_train = split == 0
    global_rate = float(y[is_train].mean())

    # --- leakage-safe resistance rates (TRAIN triples only), smoothed ---
    tr = pd.DataFrame({"o": o_idx[is_train], "a": a_idx[is_train], "y": y[is_train]})
    o_pos = tr.groupby("o")["y"].sum().reindex(range(len(organisms)), fill_value=0.0)
    o_tot = tr.groupby("o")["y"].size().reindex(range(len(organisms)), fill_value=0)
    a_pos = tr.groupby("a")["y"].sum().reindex(range(len(antibiotics)), fill_value=0.0)
    a_tot = tr.groupby("a")["y"].size().reindex(range(len(antibiotics)), fill_value=0)
    org_rate = _smoothed_rate(o_pos, o_tot, global_rate).to_numpy()
    abx_rate = _smoothed_rate(a_pos, a_tot, global_rate).to_numpy()

    # --- structural counts/degrees (label-free -> may use all rows) ---
    org_prev = np.bincount(o_idx, minlength=len(organisms)).astype(np.float64)
    abx_prev = np.bincount(a_idx, minlength=len(antibiotics)).astype(np.float64)
    pair = df[[ORG, ABX]].drop_duplicates()
    org_deg = pair.groupby(ORG).size().reindex(organisms, fill_value=0).to_numpy()
    abx_deg = pair.groupby(ABX).size().reindex(antibiotics, fill_value=0).to_numpy()

    # --- patient features ---
    pdf = df.drop_duplicates(PK).set_index(PK).reindex(patients)
    age_ord = _age_to_ordinal(pdf[config.COLUMNS["age"]])
    age_known = age_ord.notna().to_numpy().astype(np.float64)
    age_ord = age_ord.fillna(age_ord.median())  # imputation uses no label -> not label leakage
    gender = pd.get_dummies(pdf[config.COLUMNS["gender"]].astype(str), dummy_na=False)
    adi = pd.to_numeric(pdf[ADI], errors="coerce")
    adi_known = adi.notna().to_numpy().astype(np.float64)
    adi = adi.fillna(adi.median()).to_numpy()
    pat_ncult = df.groupby(PK)[CK].nunique().reindex(patients, fill_value=0).to_numpy()
    pat_norg = df.groupby(PK)[ORG].nunique().reindex(patients, fill_value=0).to_numpy()

    organism_x = _zscore(np.column_stack([np.log1p(org_prev), org_deg.astype(float), org_rate]))
    antibiotic_x = _zscore(np.column_stack([np.log1p(abx_prev), abx_deg.astype(float), abx_rate]))
    patient_num = _zscore(np.column_stack([
        age_ord.to_numpy(), age_known, adi, adi_known,
        np.log1p(pat_ncult), np.log1p(pat_norg),
    ]))
    patient_x = np.hstack([patient_num, gender.to_numpy().astype(float)])

    # --- core edges (directed; reverse added in to_hetero_data) ---
    tested = np.vstack([pair[ORG].map(org_map).to_numpy(), pair[ABX].map(abx_map).to_numpy()]).astype(np.int64)
    grew_pairs = df[[PK, ORG]].drop_duplicates()
    grew = np.vstack([grew_pairs[PK].map(pat_map).to_numpy(), grew_pairs[ORG].map(org_map).to_numpy()]).astype(np.int64)
    known_resistant = _load_known_resistant(org_map, abx_map)

    return {
        "node_names": {"organism": organisms, "antibiotic": antibiotics, "patient": patients},
        "x": {"organism": organism_x.astype(np.float32),
              "antibiotic": antibiotic_x.astype(np.float32),
              "patient": patient_x.astype(np.float32)},
        "edges": {
            ("organism", "tested", "antibiotic"): tested,
            ("patient", "grew", "organism"): grew,
            ("organism", "known_resistant", "antibiotic"): known_resistant,
        },
        "triples": np.vstack([p_idx, o_idx, a_idx]).astype(np.int64),  # [3, N]
        "y": y,
        "split": split.astype(np.int8),                                # {0=train,1=val,2=test}
        "train_pos_weight": float((1 - global_rate) / max(global_rate, 1e-6)),
    }


def to_hetero_data(arrays: dict):
    """Pack build_arrays() output into a PyG HeteroData (adds reverse edges). Colab/torch."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for ntype, x in arrays["x"].items():
        data[ntype].x = torch.from_numpy(x)
    for (s, r, d), ei in arrays["edges"].items():
        data[s, r, d].edge_index = torch.from_numpy(ei)
        data[d, f"rev_{r}", s].edge_index = torch.from_numpy(ei[[1, 0]])  # reverse for message passing

    tri = torch.from_numpy(arrays["triples"])
    data.triple_index = tri                      # [3, N]: patient, organism, antibiotic
    data.triple_label = torch.from_numpy(arrays["y"])
    split = torch.from_numpy(arrays["split"])
    data.train_mask = split == 0
    data.val_mask = split == 1
    data.test_mask = split == 2
    data.train_pos_weight = torch.tensor(arrays["train_pos_weight"])
    return data


def build_graph(ward: str | None = None, sample_n: int | None = None, seed: int = config.SEED):
    """Full pipeline: load -> arrays -> HeteroData. Needs torch (run on Colab)."""
    df = load_cohort_frame(ward=ward, sample_n=sample_n)
    return to_hetero_data(build_arrays(df, seed=seed))


def _self_check_arrays(ward: str | None = None) -> dict:
    """torch-free verification of build_arrays on real data (runs anywhere)."""
    df = load_cohort_frame(ward=ward)
    A = build_arrays(df)
    n_tri = A["triples"].shape[1]
    tested = A["edges"][("organism", "tested", "antibiotic")]
    grew = A["edges"][("patient", "grew", "organism")]
    kr = A["edges"][("organism", "known_resistant", "antibiotic")]
    print(f"ward={ward or 'ALL'} | triples={n_tri:,}")
    print(f"nodes: patient={len(A['node_names']['patient']):,} organism={len(A['node_names']['organism'])} "
          f"antibiotic={len(A['node_names']['antibiotic'])}")
    print(f"edges: tested={tested.shape[1]:,} grew={grew.shape[1]:,} known_resistant={kr.shape[1]:,}")
    print(f"feature dims: patient={A['x']['patient'].shape[1]} organism={A['x']['organism'].shape[1]} "
          f"antibiotic={A['x']['antibiotic'].shape[1]}")
    for nt, x in A["x"].items():
        assert np.isfinite(x).all(), f"non-finite feature in {nt}"
    # split sanity + patient disjointness
    tri_pat = A["triples"][0]
    sp = A["split"]
    sizes = {int(k): int(v) for k, v in zip(*np.unique(sp, return_counts=True))}
    tr_p, va_p, te_p = (set(tri_pat[sp == c].tolist()) for c in (0, 1, 2))
    assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p), "patient leakage across splits"
    print(f"split triples: train={sizes.get(0,0):,} val={sizes.get(1,0):,} test={sizes.get(2,0):,} "
          f"| pos_weight={A['train_pos_weight']:.2f}")
    print("OK: build_arrays self-check passed.")
    return A


if __name__ == "__main__":
    _self_check_arrays()
