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

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import ABX, ADI, CK, ORG, PK, load_cohort_frame

COM = config.COLUMNS["comorbidity"]  # comorbidity_component
PROC = config.COLUMNS["procedure"]   # procedure_description
TK = config.KEYS["time"]             # order_time_jittered_utc

# High-volume brand/short names in abx_class_exposure.medication_name whose drug stem
# doesn't match a tested-antibiotic node name automatically. Maps stem -> node name.
# (Only brands whose drug IS one of our 54 tested antibiotics; e.g. Zithromax/Azithromycin
# is omitted because Azithromycin is not a tested node.)
_ABX_BRAND_ALIASES = {
    "cipro": "Ciprofloxacin",
    "levaquin": "Levofloxacin",
    "keflex": "Cephalexin/Cephalothin",
    "bactrim": "Trimethoprim/Sulfamethoxazole",
    "septra": "Trimethoprim/Sulfamethoxazole",
    "macrobid": "Nitrofurantoin",
    "macrodantin": "Nitrofurantoin",
    "flagyl": "Metronidazole",
}

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


def _stream_comorbidity_edges(patient_ids: set, chunksize: int = 2_000_000,
                              cache_path: str | None = None) -> pd.DataFrame:
    """Distinct (patient, comorbidity_component) pairs restricted to cohort patients.

    The comorbidity table is ~18 GB / 206M rows, so it is streamed in chunks and each
    chunk filtered to the cohort before dedup (bounds memory to the ~2.1M distinct
    edges). If cache_path exists it is loaded instead; if set, the result is cached.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES["comorbidity"]
    parts = []
    for chunk in pd.read_csv(fp, usecols=[PK, COM], chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk[PK].isin(patient_ids)].dropna()
        if len(chunk):
            parts.append(chunk.drop_duplicates())
    edges = (pd.concat(parts, ignore_index=True).drop_duplicates()
             if parts else pd.DataFrame(columns=[PK, COM]))
    if cache_path:
        edges.to_parquet(cache_path, index=False)
    return edges


def _build_patient_entity_arrays(edge_df: pd.DataFrame, ecol: str, pat_map: dict,
                                 pat_rate: pd.Series, global_rate: float):
    """Generic (entity node names, features, (patient,-,entity) edge_index).

    Used for both comorbidity and procedure enrichment nodes. Features (real signals,
    no identity one-hot): log prevalence (# distinct patients) and a leakage-safe
    resistance association — mean of connected TRAIN patients' resistance rates,
    shrunk toward the global rate. `pat_rate` is indexed by patient node id and
    defined for TRAIN patients only.
    """
    edge_df = edge_df[edge_df[PK].isin(pat_map)]
    names = np.sort(edge_df[ecol].unique())
    emap = {v: i for i, v in enumerate(names)}
    n = len(names)
    e_p = edge_df[PK].map(pat_map).to_numpy()
    e_e = edge_df[ecol].map(emap).to_numpy()

    ee = pd.DataFrame({"p": e_p, "e": e_e})
    ee["rate"] = ee["p"].map(pat_rate)                     # NaN for val/test patients
    grp = ee.dropna(subset=["rate"]).groupby("e")["rate"]
    n_tot = grp.count().reindex(range(n), fill_value=0)
    n_pos = grp.sum().reindex(range(n), fill_value=0.0)
    ent_rate = _smoothed_rate(n_pos, n_tot, global_rate).to_numpy()
    ent_prev = ee.groupby("e").size().reindex(range(n), fill_value=0).to_numpy()

    x = _zscore(np.column_stack([np.log1p(ent_prev), ent_rate])).astype(np.float32)
    return names, x, np.vstack([e_p, e_e]).astype(np.int64)


def _build_comorbidity_arrays(edge_df: pd.DataFrame, pat_map: dict, pat_rate: pd.Series,
                              global_rate: float):
    """(comorbidity node names, features, (patient,has,comorbidity) edge_index)."""
    return _build_patient_entity_arrays(edge_df, COM, pat_map, pat_rate, global_rate)


def _load_procedure_edges(patient_ids: set, cache_path: str | None = None) -> pd.DataFrame:
    """Distinct (patient, procedure_description) rows for cohort patients.

    priorprocedures is ~126 MB — read whole (usecols), not streamed.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES["procedures"]
    df = pd.read_csv(fp, usecols=[PK, PROC], low_memory=False)
    df = df[df[PK].isin(patient_ids)].dropna().drop_duplicates()
    if cache_path:
        df.to_parquet(cache_path, index=False)
    return df


def _med_stem(name: str) -> str:
    """First drug token of a medication string (drops salts/formulations/combos)."""
    return re.split(r"[ /-]", str(name).strip().lower())[0]


def _medication_to_node(med_names, abx_map: dict) -> dict:
    """Map each medication_name -> an antibiotic node name (or None if no match)."""
    stem2node: dict = {}
    for node in abx_map:                       # node stems from tested-antibiotic names
        for comp in node.split("/"):
            stem2node.setdefault(_med_stem(comp), node)
    stem2node.update(_ABX_BRAND_ALIASES)       # brand overrides win
    return {m: stem2node.get(_med_stem(m)) for m in med_names}


def _load_abx_exposure(patient_ids: set, cache_path: str | None = None) -> pd.DataFrame:
    """Distinct (patient, medication_name) exposure rows for cohort patients.

    abx_class_exposure is ~540 MB / 5.4M rows — read whole (usecols), not streamed.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES["abx_class_exp"]
    df = pd.read_csv(fp, usecols=[PK, "medication_name"], low_memory=False)
    df = df[df[PK].isin(patient_ids)].dropna().drop_duplicates()
    if cache_path:
        df.to_parquet(cache_path, index=False)
    return df


def _build_prior_exposure_edges(exp_df: pd.DataFrame, pat_map: dict, abx_map: dict) -> np.ndarray:
    """(patient, prior_exposure, antibiotic) edges — reuses the antibiotic node (no new type)."""
    m2n = _medication_to_node(exp_df["medication_name"].unique(), abx_map)
    df = exp_df.assign(node=exp_df["medication_name"].map(m2n)).dropna(subset=["node"])
    # keep only exposures to patients AND antibiotic nodes present in THIS graph
    # (a ward subset may lack some of the 54 antibiotics -> alias could point off-graph)
    df = df[df[PK].isin(pat_map) & df["node"].isin(abx_map)]
    pairs = df[[PK, "node"]].drop_duplicates()
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64)
    e_p = pairs[PK].map(pat_map).to_numpy()
    e_a = pairs["node"].map(abx_map).to_numpy()
    return np.vstack([e_p, e_a]).astype(np.int64)


def _read_median_cols(name: str) -> tuple[pd.DataFrame, list]:
    """Per-culture mean of a wide table's median_* columns (labs or vitals).

    Values are stored as strings with 'Null' sentinels; coerce to numeric. A culture
    may span several Period_Day rows -> collapse to one row per culture by mean.
    """
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES[name]
    med = [c for c in pd.read_csv(fp, nrows=0).columns if c.startswith("median_")]
    df = pd.read_csv(fp, usecols=[CK] + med, low_memory=False)
    df[med] = df[med].replace("Null", np.nan).apply(pd.to_numeric, errors="coerce")
    df = df.groupby(CK, as_index=False)[med].mean()
    renamed = [f"{name}_{c}" for c in med]
    df = df.rename(columns=dict(zip(med, renamed)))
    return df, renamed


def _load_labvital_per_culture(cache_path: str | None = None) -> pd.DataFrame:
    """Per-culture labs+vitals median summary (ward-independent -> cache once, reuse)."""
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    labs, _ = _read_median_cols("labs")
    vit, _ = _read_median_cols("vitals")
    out = labs.merge(vit, on=CK, how="outer")
    if cache_path:
        out.to_parquet(cache_path, index=False)
    return out


def _aggregate_labvital_to_patient(cp_df: pd.DataFrame, lv_df: pd.DataFrame,
                                   patients: np.ndarray):
    """Per-patient labs/vitals features + measured flags, aligned to node order.

    cp_df: unique (culture, patient). lv_df: per-culture labs+vitals summary.
    Averages a patient's cultures; NaN (never measured) -> column median impute,
    plus a labs_measured / vitals_measured 0/1 flag. Returns (names, float32 X).
    """
    feat_cols = [c for c in lv_df.columns if c != CK]
    per = cp_df.merge(lv_df, on=CK, how="left").groupby(PK)[feat_cols].mean()
    per = per.reindex(patients)
    labs_cols = [c for c in feat_cols if c.startswith("labs_")]
    vit_cols = [c for c in feat_cols if c.startswith("vitals_")]
    labs_meas = per[labs_cols].notna().any(axis=1).astype(float).to_numpy()
    vit_meas = per[vit_cols].notna().any(axis=1).astype(float).to_numpy()
    per = per.fillna(per.median()).fillna(0.0)          # impute (no label -> not leakage)
    x = _zscore(per[feat_cols].to_numpy())
    x = np.column_stack([x, labs_meas, vit_meas]).astype(np.float32)
    return feat_cols + ["labs_measured", "vitals_measured"], x


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


def _history_raw(df: pd.DataFrame, global_rate: float, alpha: float = 10.0) -> np.ndarray:
    """Raw (un-standardised) per-test patient-history features, in df row order.

    Columns: log prior-culture-count, prior resistance rate (overall / same-antibiotic /
    same-organism, EB-shrunk toward global), log days-since-previous-culture.
    Leakage-safe: each test sees only the patient's cultures STRICTLY BEFORE it.
    """
    d = df[[PK, ORG, ABX, TK, "label"]].copy()
    d["t"] = pd.to_datetime(d[TK], errors="coerce", utc=True)
    d = d.sort_values([PK, "t"], kind="stable")
    lab = d["label"].to_numpy().astype(float)

    def prior_rate(keys):
        g = d.groupby(keys, sort=False)
        n = g.cumcount().to_numpy().astype(float)          # #cultures before this one
        s = g["label"].cumsum().to_numpy() - lab           # sum of prior labels (excl. current)
        return n, (s + alpha * global_rate) / (n + alpha)  # EB-shrunk prior resistance rate

    n_all, r_all = prior_rate(PK)
    _, r_abx = prior_rate([PK, ABX])
    _, r_org = prior_rate([PK, ORG])
    days_since = (d["t"] - d.groupby(PK, sort=False)["t"].shift(1)).dt.total_seconds().to_numpy() / 86400.0
    days_since = np.nan_to_num(days_since, nan=0.0)         # first-ever culture -> 0

    feat = np.column_stack([np.log1p(n_all), r_all, r_abx, r_org,
                            np.log1p(np.clip(days_since, 0, None))])
    return pd.DataFrame(feat, index=d.index).reindex(df.index).to_numpy()  # back to df order


def _patient_history_features(df: pd.DataFrame, global_rate: float, alpha: float = 10.0) -> np.ndarray:
    """Standardised patient-history predictors (personalized-antibiogram signal; Corbin 2022)."""
    return _zscore(_history_raw(df, global_rate, alpha)).astype(np.float32)


def build_arrays(df: pd.DataFrame, seed: int = config.SEED, enrich: tuple = (),
                 comorbidity_cache: str | None = None, exposure_cache: str | None = None,
                 procedure_cache: str | None = None, rich_patient: bool = False,
                 labvital_cache: str | None = None, patient_history: bool = False) -> dict:
    """Pure pandas/numpy build. `df` = a data_loader.load_cohort_frame() result.

    Returns a dict of node maps, float feature matrices, edge_index arrays, the
    supervision triples/labels, and the patient-grouped split codes.

    `enrich` adds enrichment node/edge types on top of the core 3, e.g.
    enrich=("comorbidity",) adds the comorbidity node + (patient,has,comorbidity)
    edge. `comorbidity_cache` is a parquet path for the streamed edge list.
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
    # per-patient TRAIN resistance rate (indexed by patient node id; train patients only)
    pat_rate = pd.DataFrame({"p": p_idx[is_train], "y": y[is_train]}).groupby("p")["y"].mean()

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
    # fixed-width gender encoding (values are '0'/'1'/'Null'); a per-subset one-hot
    # would vary in width and break federated weight aggregation, so pin 2 columns.
    gser = pdf[config.COLUMNS["gender"]].astype(str)
    gender = np.column_stack([(gser == "1").to_numpy(dtype=float),            # is_1
                              (~gser.isin(["0", "1"])).to_numpy(dtype=float)])  # is_unknown/Null
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
    patient_x = np.hstack([patient_num, gender])

    if rich_patient:  # append labs/vitals node features (+ measured flags)
        lv = _load_labvital_per_culture(labvital_cache)
        _, lv_x = _aggregate_labvital_to_patient(df[[CK, PK]].drop_duplicates(), lv, patients)
        patient_x = np.hstack([patient_x, lv_x])

    # --- core edges (directed; reverse added in to_hetero_data) ---
    tested = np.vstack([pair[ORG].map(org_map).to_numpy(), pair[ABX].map(abx_map).to_numpy()]).astype(np.int64)
    grew_pairs = df[[PK, ORG]].drop_duplicates()
    grew = np.vstack([grew_pairs[PK].map(pat_map).to_numpy(), grew_pairs[ORG].map(org_map).to_numpy()]).astype(np.int64)
    known_resistant = _load_known_resistant(org_map, abx_map)

    out = {
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

    if "comorbidity" in enrich:
        ce = _stream_comorbidity_edges(set(patients), cache_path=comorbidity_cache)
        names, com_x, com_ei = _build_comorbidity_arrays(ce, pat_map, pat_rate, global_rate)
        out["node_names"]["comorbidity"] = names
        out["x"]["comorbidity"] = com_x
        out["edges"][("patient", "has", "comorbidity")] = com_ei

    if "prior_exposure" in enrich:
        ee = _load_abx_exposure(set(patients), cache_path=exposure_cache)
        out["edges"][("patient", "prior_exposure", "antibiotic")] = \
            _build_prior_exposure_edges(ee, pat_map, abx_map)

    if "procedure" in enrich:
        pe = _load_procedure_edges(set(patients), cache_path=procedure_cache)
        names, proc_x, proc_ei = _build_patient_entity_arrays(pe, PROC, pat_map, pat_rate, global_rate)
        out["node_names"]["procedure"] = names
        out["x"]["procedure"] = proc_x
        out["edges"][("patient", "underwent", "procedure")] = proc_ei

    if patient_history:  # per-test decoder features (prior-resistance predictors)
        out["triple_feat"] = _patient_history_features(df, global_rate)

    return out


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
    if "triple_feat" in arrays:                  # per-test patient-history decoder features
        data.triple_feat = torch.from_numpy(arrays["triple_feat"])
    split = torch.from_numpy(arrays["split"])
    data.train_mask = split == 0
    data.val_mask = split == 1
    data.test_mask = split == 2
    data.train_pos_weight = torch.tensor(arrays["train_pos_weight"])
    return data


def build_graph(ward: str | None = None, sample_n: int | None = None, seed: int = config.SEED,
                enrich: tuple = (), comorbidity_cache: str | None = None,
                exposure_cache: str | None = None, procedure_cache: str | None = None,
                rich_patient: bool = False, labvital_cache: str | None = None,
                patient_history: bool = False):
    """Full pipeline: load -> arrays -> HeteroData. Needs torch (run on Colab)."""
    df = load_cohort_frame(ward=ward, sample_n=sample_n)
    return to_hetero_data(build_arrays(df, seed=seed, enrich=enrich,
                                       comorbidity_cache=comorbidity_cache,
                                       exposure_cache=exposure_cache,
                                       procedure_cache=procedure_cache,
                                       rich_patient=rich_patient, labvital_cache=labvital_cache,
                                       patient_history=patient_history))


def _self_check_arrays(ward: str | None = None, enrich: tuple = (),
                       comorbidity_cache: str | None = None, rich_patient: bool = False) -> dict:
    """torch-free verification of build_arrays on real data (runs anywhere)."""
    df = load_cohort_frame(ward=ward)
    A = build_arrays(df, enrich=enrich, comorbidity_cache=comorbidity_cache,
                     rich_patient=rich_patient)
    n_tri = A["triples"].shape[1]
    tested = A["edges"][("organism", "tested", "antibiotic")]
    grew = A["edges"][("patient", "grew", "organism")]
    kr = A["edges"][("organism", "known_resistant", "antibiotic")]
    print(f"ward={ward or 'ALL'} | triples={n_tri:,}")
    print(f"nodes: patient={len(A['node_names']['patient']):,} organism={len(A['node_names']['organism'])} "
          f"antibiotic={len(A['node_names']['antibiotic'])}")
    print(f"edges: tested={tested.shape[1]:,} grew={grew.shape[1]:,} known_resistant={kr.shape[1]:,}")
    if ("patient", "has", "comorbidity") in A["edges"]:
        ce = A["edges"][("patient", "has", "comorbidity")]
        print(f"enrichment: comorbidity nodes={len(A['node_names']['comorbidity'])} "
              f"has_edges={ce.shape[1]:,} feat_dim={A['x']['comorbidity'].shape[1]}")
    if ("patient", "prior_exposure", "antibiotic") in A["edges"]:
        pe = A["edges"][("patient", "prior_exposure", "antibiotic")]
        n_abx_hit = len(np.unique(pe[1])) if pe.shape[1] else 0
        print(f"enrichment: prior_exposure edges={pe.shape[1]:,} "
              f"antibiotic nodes hit={n_abx_hit}/{len(A['node_names']['antibiotic'])}")
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
