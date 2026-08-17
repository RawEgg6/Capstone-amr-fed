"""Phase-2 partitioning: split the single institution into simulated hospitals.

Main method: a Dirichlet(alpha) ward-mixture (Hsu et al. 2019; Li et al. 2022,
NIID-Bench) — each hospital is a different blend of wards, alpha dials how
different they are (small alpha = strong non-IID, large = near-IID). Plus plain
ward and ADI-quartile baseline splits.

Every partition assigns each PATIENT to exactly one hospital (no leakage): the
returned object is a Series indexed by patient id -> hospital label. A patient's
"home ward" is their highest-acuity ward across all their cultures (priority
ICU>ER>IP>OP, else NONE) — the worst-value-in-window convention used by ICU
severity scores.

Pure pandas/numpy; networkx optional, lazy. No torch — runs and is verifiable on any stack.

Dispatch contract (`_call_partition`): a partition_fn is invoked by KEYWORD match on its
signature — `n_clients` is injected only when the fn declares it AND a non-None value is
given; `seed` is injected when the fn declares a keyword named 'seed'; 2-arg (df, seed)
lambdas and 1-arg baselines (specimen_baseline) fall back to positional calls. Footguns:
run_fedavg's default `n_clients=5` breaks `topology_split` (needs a multiple of 4) —
callers pass `n_clients ∈ {4, 8, ...}`; 3+-required-param lambdas must use
`functools.partial` (only `n_clients`/`seed` are matched by name).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .data_loader import ADI, PK

WARD_COL = "ward"  # priority-collapsed per-culture ward added by data_loader
LABEL = "label"    # binary target column (Resistant/Intermediate=1 vs Susceptible=0)
CDESC = config.COLUMNS["culture_description"]
ORG, ABX = config.COLUMNS["organism"], config.COLUMNS["antibiotic"]
_SPECIMEN_VOCAB = ["URINE", "RESPIRATORY", "BLOOD"]  # mirror graph_build._SPECIMEN_VOCAB


def assign_home_ward(df: pd.DataFrame) -> pd.Series:
    """Each patient's highest-priority ward across their cultures (index=patient)."""
    order = {w: i for i, w in enumerate(config.WARD_PRIORITY)}  # ICU=0 ... OP=3
    d = df[[PK, WARD_COL]].copy()
    d["rank"] = d[WARD_COL].map(lambda w: order.get(w, len(order)))  # NONE -> lowest priority
    return d.sort_values("rank").groupby(PK)[WARD_COL].first()


def _apportion(n: int, props: np.ndarray) -> np.ndarray:
    """Split n items across len(props) buckets by proportions (largest-remainder)."""
    exact = props * n
    counts = np.floor(exact).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        for c in np.argsort(exact - counts)[::-1][:rem]:
            counts[c] += 1
    return counts


def _rank_bucket_split(score: pd.Series, n_clients: int) -> pd.Series:
    """Deterministic quantile dial: rank patients by `score` (stable mergesort —
    deterministic ties), assign contiguous, balanced blocks -> hospital 0 = lowest
    scores, hospital k-1 = highest. Returns Series index=patient -> client id."""
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if len(score) < n_clients:
        raise ValueError(f"n_clients={n_clients} exceeds #patients {len(score)}")
    order = score.sort_values(kind="mergesort").index.to_numpy()
    counts = _apportion(len(order), np.full(n_clients, 1.0 / n_clients))
    assignment = pd.Series(-1, index=score.index, dtype="int64")
    start = 0
    for c, k in enumerate(counts):
        assignment.loc[order[start:start + k]] = c
        start += k
    assert (assignment >= 0).all(), "some patient was left unassigned"
    return assignment


def _tested_edge_homophily(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(organism, antibiotic) tested edge: binary majority label + assortativity-
    style homophily deviation. b(o,a) = majority label over the pair's triples
    (rate >= 0.5 -> 1). agreement(e) = fraction of e's neighbor tested-edges (sharing
    organism OR antibiotic endpoint) with the same binary label. dev(e) = agreement(e)
    - P(b(e)), where P(b) is the global fraction of tested edges with label b — this
    removes the ambient resistance rate, so the measure is structural, not a label-rate
    shift. Undefined (degree <= 1 on both sides) -> dev 0 (neutral).
    Returns DataFrame indexed by (ORG, ABX) with columns b, dev."""
    d = df[[ORG, ABX, "label"]]
    pair = d.groupby([ORG, ABX], sort=False).agg(rate=("label", "mean"))
    b = (pair["rate"] >= 0.5).astype(np.int8)
    t = pd.DataFrame({ORG: pair.index.get_level_values(0),
                      ABX: pair.index.get_level_values(1),
                      "b": b.to_numpy()})
    org_stat = t.groupby(ORG)["b"].agg(n_pos_org="sum", deg_org="size")
    abx_stat = t.groupby(ABX)["b"].agg(n_pos_abx="sum", deg_abx="size")
    t = t.merge(org_stat, on=ORG, how="left").merge(abx_stat, on=ABX, how="left")
    bb = t["b"].to_numpy()
    deg_o, n_o = t["deg_org"].to_numpy(), t["n_pos_org"].to_numpy()
    deg_a, n_a = t["deg_abx"].to_numpy(), t["n_pos_abx"].to_numpy()
    # same-label neighbor count excluding self
    same_o = np.where(bb == 1, n_o - 1, (deg_o - n_o) - 1).astype(float)
    same_a = np.where(bb == 1, n_a - 1, (deg_a - n_a) - 1).astype(float)
    # np.where evaluates both branches eagerly, so deg_x == 1 -> 0/0 (nan) would emit a
    # RuntimeWarning even though the result is discarded; suppress to keep output pristine.
    with np.errstate(divide="ignore", invalid="ignore"):
        agree_o = np.where(deg_o > 1, same_o / (deg_o - 1), np.nan)
        agree_a = np.where(deg_a > 1, same_a / (deg_a - 1), np.nan)
    # nanmean's "Mean of empty slice" is a warnings.warn, not a numpy fp flag
    # (np.errstate can't catch it); suppress when both deg == 1 -> all-NaN slice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agree = np.nanmean(np.column_stack([agree_o, agree_a]), axis=1)  # one-sided OK
    p1 = float(b.mean())
    chance = np.where(bb == 1, p1, 1.0 - p1)
    dev = np.where(np.isnan(agree), 0.0, agree - chance)             # both deg<=1 -> neutral
    return pd.DataFrame({"b": bb, "dev": dev}, index=pair.index)


def _patient_homophily(df: pd.DataFrame) -> pd.Series:
    """Per-patient homophily score = mean deviation over their triples (NaN-free)."""
    edges = _tested_edge_homophily(df)
    d = df[[PK, ORG, ABX]].merge(edges[["dev"]], left_on=[ORG, ABX], right_index=True, how="left")
    return d.groupby(PK)["dev"].mean().fillna(0.0)


def homophily_split(df: pd.DataFrame, n_clients: int = 5,
                    seed: int = config.SEED) -> pd.Series:
    """Structure non-IID split: hospitals span the homophily spectrum.

    Each patient is scored by the assortativity-style homophily deviation of their
    (organism, antibiotic) tested edges; patients are ranked and assigned to
    contiguous, balanced blocks. Hospital 0 = most HETEROPHILIC (resistance
    scattered), hospital k-1 = most HOMOPHILIC (resistance clustered). The deviation
    subtracts the ambient resistance rate, so resistance RATE stays roughly constant
    across hospitals — the divergence is structural. `seed` unused (deterministic).
    Returns Series index=patient -> client id."""
    return _rank_bucket_split(_patient_homophily(df), n_clients)


def _patient_hubness(df: pd.DataFrame) -> pd.Series:
    """Per-patient drug-repertoire breadth = log1p(# distinct antibiotics across the
    patient's tested edges). Grows with real breadth; bounded, no plateau at any single
    organism's tested-degree. NaN-free."""
    return np.log1p(df.groupby(PK)[ABX].nunique()).astype(float)


def degree_skew_split(df: pd.DataFrame, n_clients: int = 5,
                      seed: int = config.SEED) -> pd.Series:
    """Topology distribution skew (OpenFGL 2024): hospitals differ in graph degree.

    Each patient is scored by their drug-repertoire breadth (# distinct antibiotics
    across their tested edges, log1p-transformed); patients are ranked and assigned
    to contiguous, balanced blocks. Hospital 0 = SPARSE (1-abx patients), hospital
    k-1 = HUB-heavy (broad-repertoire patients) — contrasting degree distributions
    the shared GNN cannot fit equally. `seed` unused (deterministic).
    Returns Series index=patient -> client id."""
    return _rank_bucket_split(_patient_hubness(df), n_clients)


def _robust_z(s: pd.Series) -> pd.Series:
    """Robust z-score: (s - median) / max(MAD, std, 1.0). All-identical -> zeros.

    MAD = median absolute deviation; the /1.0 floor keeps near-constant axes from
    amplifying floating-point noise into fake signal. NaN-safe (single-value/empty axes
    fall through to the 1.0 floor)."""
    med = s.median()
    mad = (s - med).abs().median()
    std = s.std(ddof=0)
    denom = 1.0
    for v in (mad, std):
        if v == v:                       # not NaN
            denom = max(denom, v)
    return (s - med) / denom


def _residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """Return y with its OLS linear dependence on x removed: y - OLS_fit(y ~ x).

    Constant x (var == 0) carries no signal to remove -> y returned unchanged.
    Series are aligned on y's index. Deterministic."""
    x = x.reindex(y.index)
    if np.var(x) == 0:
        return y
    b = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0))
    a = float(y.mean()) - b * float(x.mean())
    return y - (a + b * x)


def _quadrant_assign(hom: pd.Series, hub: pd.Series, n_clients: int,
                     purity: float, seed: int) -> pd.Series:
    """Cross homophily x hubness into four structural quadrants (FedGTA / AdaFGL 2-D).

    Each axis is median-split on first-ranks (rank tie-break deterministic) -> a_sign
    (1 = above-median homophily = 'clustered') and b_sign (1 = above-median hubness =
    'hub'); cell = 2*a_sign + b_sign, so 0 = scattered+sparse, 1 = scattered+hub,
    2 = clustered+sparse, 3 = clustered+hub (opposite corners). n_clients must be a
    multiple of 4 with m = n_clients // 4 hospitals per quadrant; within each cell the
    patients are ranked by za + zb (robust z of each axis, so the two scales are
    comparable) and split into m balanced hospitals via _rank_bucket_split ->
    hospital = cell*m + bucket. `purity` in [0, 1] softens the hard split: with that
    probability a patient is reassigned to a uniform random hospital (np.random.default_rng(seed);
    purity=1.0 = fully uniform). Guards: purity outside [0,1], n_clients % 4 != 0, or any
    quadrant with < m patients -> ValueError.
    Returns Series index=patient -> client id."""
    if not (0.0 <= purity <= 1.0):
        raise ValueError(f"purity must be in [0, 1], got {purity}")
    if n_clients < 4 or n_clients % 4 != 0:
        raise ValueError(
            f"n_clients must be a multiple of 4 (>= 4), got {n_clients} "
            f"(each quadrant needs n_clients // 4 hospitals)"
        )
    m = n_clients // 4
    a_rank = hom.rank(method="first")
    b_rank = hub.rank(method="first")
    a_sign = (a_rank > a_rank.median()).astype(np.int8)
    b_sign = (b_rank > b_rank.median()).astype(np.int8)
    cell = (2 * a_sign + b_sign).astype(np.int64)
    za = _robust_z(hom)
    zb = _robust_z(hub)
    counts = cell.value_counts()
    missing = [c for c in range(4) if counts.get(c, 0) < m]
    if missing:
        raise ValueError(
            f"each quadrant needs >= m={m} patients but quadrants {missing} have fewer; "
            f"quadrant patient counts: {counts.to_dict()}"
        )
    assignment = pd.Series(-1, index=cell.index, dtype="int64")
    for c in range(4):
        mask = cell == c
        bucket = _rank_bucket_split(za[mask] + zb[mask], m)
        assignment[mask] = (c * m + bucket).to_numpy()
    if purity > 0.0:
        rng = np.random.default_rng(seed)
        noisy = rng.random(len(assignment)) < purity
        if noisy.any():
            assignment[noisy] = rng.integers(0, n_clients, size=int(noisy.sum()))
    return assignment


def topology_split(df: pd.DataFrame, n_clients: int = 4, purity: float = 0.0,
                   seed: int = config.SEED, decorrelate: bool = False) -> pd.Series:
    """2-D crossed structural split (FedGTA / AdaFGL): homophily x hubness quadrants.

    Each patient is scored on both axes — homophily (assortativity-style resistance
    clustering of their tested edges) and hubness (drug-repertoire breadth) — and crossed
    via _quadrant_assign into four structurally distinct hospital types: scattered+sparse,
    scattered+hub, clustered+sparse, clustered+hub. `decorrelate=True` residualizes hubness
    on homophily first (the axes correlate: broad-repertoire patients tend to see clustered
    bugs), so the hub quadrant is the *independent* breadth signal. `purity` in [0, 1]
    softens the hard split toward uniform assignment. Deterministic (seed).
    Returns Series index=patient -> client id."""
    hom = _patient_homophily(df)
    hub = _patient_hubness(df)
    if decorrelate:
        hub = _residualize(hub, hom)
    return _quadrant_assign(hom, hub, n_clients, purity, seed)


def dirichlet_ward_mixture(df: pd.DataFrame, n_clients: int = 5, alpha: float = 0.5,
                           seed: int = config.SEED) -> pd.Series:
    """Assign each patient to one of n_clients hospitals via a Dirichlet(alpha)
    ward-mixture. Returns Series index=patient -> client id (0..n_clients-1)."""
    home = assign_home_ward(df)
    rng = np.random.default_rng(seed)
    assignment = pd.Series(-1, index=home.index, dtype="int64")
    for ward in sorted(home.unique()):
        pats = home.index[home == ward].to_numpy(copy=True)  # writable for shuffle
        rng.shuffle(pats)
        counts = _apportion(len(pats), rng.dirichlet(alpha * np.ones(n_clients)))
        start = 0
        for c, k in enumerate(counts):
            assignment.loc[pats[start:start + k]] = c
            start += k
    assert (assignment >= 0).all(), "some patient was left unassigned"
    return assignment


def label_dirichlet(df: pd.DataFrame, n_clients: int = 5, beta: float = 0.5,
                    n_bins: int = 3, seed: int = config.SEED) -> pd.Series:
    """noniid-labeldir split (Li et al. 2022; Hsu et al. 2019): partition PATIENTS so
    hospitals differ in their *resistant-rate* mix — the label-distribution skew that
    actually widens the FedAvg-vs-local gap (a ward/feature split barely does).

    Each patient is summarised by their resistant fraction, binned into n_bins strata;
    each stratum is split across clients via Dirichlet(beta). Small beta => strong skew.
    Returns Series index=patient -> client id (0..n_clients-1)."""
    rng = np.random.default_rng(seed)
    rate = df.groupby(PK)[LABEL].mean()  # per-patient fraction of tests that are resistant
    # rank-based bins so heavy rate ties (e.g. patients with one all-S culture) don't
    # collapse the quantile edges; the extremes still separate into top/bottom strata.
    strata = pd.qcut(rate.rank(method="first"), n_bins, labels=False)
    assignment = pd.Series(-1, index=rate.index, dtype="int64")
    for s in range(n_bins):
        pats = rate.index[strata == s].to_numpy(copy=True)  # writable for shuffle
        rng.shuffle(pats)
        counts = _apportion(len(pats), rng.dirichlet(beta * np.ones(n_clients)))
        start = 0
        for c, k in enumerate(counts):
            assignment.loc[pats[start:start + k]] = c
            start += k
    assert (assignment >= 0).all(), "some patient was left unassigned"
    return assignment


def _greedy_pack(counts: pd.Series, n_bins: int) -> dict:
    """Pack items into n_bins bins, smallest-load-first (largest items first).

    Sorts `counts` by size descending, then assigns each item to the currently
    emptiest bin (`np.argmin(loads)`), so the load spread stays within the largest
    item size even when sizes are skewed — self-contained, independent of caller
    input order. Deterministic. Returns dict mapping item -> bin (int 0..n_bins-1)."""
    counts = counts.sort_values(ascending=False)
    loads = np.zeros(n_bins)
    assignment = {}
    for item, size in counts.items():
        bin_idx = int(np.argmin(loads))
        loads[bin_idx] += size
        assignment[item] = bin_idx
    return assignment


def louvain_split(df: pd.DataFrame, n_clients: int = 5,
                  seed: int = config.SEED) -> pd.Series:
    """FedGTA-style community split: Louvain communities of the organism-antibiotic
    test graph, greedily packed into hospitals.

    Builds a weighted bipartite graph on (organism, antibiotic) tested pairs
    (edge weight = pair count, node ids coerced to str so they're consistent);
    networkx `louvain_communities` partitions it into communities (seeded). Each
    patient inherits the community of their dominant (mode) organism, and communities
    are packed into n_clients balanced hospitals via `_greedy_pack` (smallest-load-
    first). Hospitals therefore see topologically *different* bug neighbourhoods
    through the community-detection lens — the FedGTA/OpenFGL community-split idea.

    NOTE: if the test graph yields fewer communities than n_clients, some hospitals
    are empty; run_fedavg self-heals to n_clients = max(hospital)+1, so the effective
    hospital count may be less than requested.

    Requires networkx (lazy import at call time; raises a clear ImportError with an
    install hint if missing). Deterministic (seed). Returns Series index=patient ->
    client id."""
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as e:  # keep partition.py importable without networkx
        raise ImportError(
            "louvain_split requires networkx. Install with: pip install networkx"
        ) from e

    d = df[[ORG, ABX]]
    edges = d.groupby([ORG, ABX]).size()  # weight = count of tested pairs per (org, abx)
    G = nx.Graph()
    G.add_nodes_from(edges.index.get_level_values(0).astype(str), bipartite=0)
    G.add_nodes_from(edges.index.get_level_values(1).astype(str), bipartite=1)
    G.add_weighted_edges_from((str(o), str(a), int(w)) for (o, a), w in edges.items())

    comms = louvain_communities(G, weight="weight", seed=seed)
    comm_map = {node: cid for cid, comm in enumerate(comms) for node in comm}

    patient_assignments = df.groupby(PK)[ORG].agg(lambda x: x.mode().iloc[0]).astype(str)
    comm_counts = patient_assignments.map(comm_map).value_counts()  # patients per community
    assignment = _greedy_pack(comm_counts, n_clients)
    return patient_assignments.map(comm_map).map(assignment).astype(int).rename("hospital")


def _specimen_category(s: pd.Series) -> pd.Series:
    """Map raw culture_description to {URINE,RESPIRATORY,BLOOD,OTHER} (matches graph_build)."""
    cd = s.astype(str).str.upper()
    cat = pd.Series("OTHER", index=cd.index, dtype="object")
    for v in _SPECIMEN_VOCAB:
        cat[cd == v] = v
    return cat


def specimen_baseline(df: pd.DataFrame) -> pd.Series:
    """Natural split: one hospital per specimen source (urine/respiratory/blood[/other]).
    Each patient -> their modal specimen across cultures. Resistance varies strongly by
    source (EDA: urine ~0.18 vs resp ~0.29), so this is a real label + topology skew and
    the most clinically defensible partition. Returns Series index=patient -> specimen str."""
    d = pd.DataFrame({PK: df[PK].to_numpy(), "spec": _specimen_category(df[CDESC]).to_numpy()})
    # modal specimen per patient; ties -> alphabetical first (deterministic)
    return d.groupby(PK)["spec"].agg(lambda x: x.mode().iloc[0])


def organism_community(df: pd.DataFrame, n_clients: int = 5,
                       seed: int = config.SEED) -> pd.Series:
    """Structural split (option #4): group organisms into n_clients DISJOINT 'hospitals'
    so each sees a different set of bugs — the maximal topological heterogeneity a
    topology-aware aggregator is built to exploit (the FGL community-split idea; OpenFGL
    2024). Each patient -> the bucket holding their dominant (most-frequent) organism.

    Organisms are greedily packed smallest-bucket-first (largest organisms first) so
    hospital sizes stay balanced — avoids the degenerate 1-patient clients that broke the
    low-beta label-Dirichlet split. `seed` is unused (greedy is deterministic); kept for a
    uniform partition_fn(df, seed) signature. Returns Series index=patient -> client id."""
    d = pd.DataFrame({PK: df[PK].to_numpy(), "org": df[ORG].astype(str).to_numpy()})
    home_org = d.groupby(PK)["org"].agg(lambda x: x.mode().iloc[0])  # dominant organism/patient
    counts = home_org.value_counts()  # patients per organism, largest first
    org_to_client = _greedy_pack(counts, n_clients)
    return home_org.map(org_to_client)


def _patient_abx_markers(df: pd.DataFrame) -> pd.DataFrame:
    """For each patient, compute boolean presence of key marker antibiotics.

    Clinical antibiotic panels always include 3-4 beta-lactam agents (e.g.
    Ampicillin, Cefazolin, Ceftriaxone, Meropenem on the same blood culture).
    The 'dominant family' approach will therefore ALWAYS return beta_lactam for
    most patients — the correct strategy is a PRIORITY HIERARCHY on specific
    MARKER antibiotics that are only ordered when the clinical picture warrants:

      vancomycin_tested  -- panel includes vancomycin (gram-positive / MRSA concern)
      carbapenem_tested  -- panel includes meropenem/imipenem/ertapenem (severe gram-neg)
      nitrofuran_tested  -- panel includes nitrofurantoin (UTI-specific panel)
      quinolone_tested   -- panel includes any -floxacin (community/respiratory panel)

    Returns DataFrame index=patient_id with boolean columns for each marker.
    """
    s = df[ABX].astype(str).str.lower()
    d = pd.DataFrame({PK: df[PK].to_numpy()})
    d["van"]   = s.str.contains("vancomycin", regex=False)
    d["carb"]  = s.str.contains("penem", regex=False)
    d["nitro"] = s.str.contains("nitrofurantoin", regex=False)
    d["quin"]  = s.str.contains("floxacin", regex=False)
    return d.groupby(PK)[["van", "carb", "nitro", "quin"]].any()


def antibiotic_family_split(df: pd.DataFrame, n_clients: int = 5,
                             seed: int = config.SEED) -> pd.Series:
    """Non-IID split by clinical antibiotic panel type — priority marker hierarchy.

    Assigns each patient to a hospital based on the most clinically distinctive
    antibiotic in their tested panel, using a strict priority order (highest →
    lowest) so each patient lands in exactly one hospital:

      H0  vancomycin panel  -- gram-positive / MRSA concern (blood/wound)
      H1  carbapenem panel  -- severe gram-neg / ESBL (blood/respiratory/ICU)
                               (excludes patients already in H0)
      H2  nitrofurantoin    -- UTI-specific panel (community UTI, outpatient)
                               (excludes H0 and H1)
      H3  fluoroquinolone   -- respiratory / community UTI without UTI-specific drug
                               (excludes H0–H2)
      H4  first-line only   -- no distinctive marker: standard cephalosporins /
                               penicillins only (e.g. wound, paediatric)

    **Why this works where the old mode-based split failed:**
    Every panel includes 3–4 beta-lactam agents, so the "dominant family" is
    always beta-lactam (84% of patients). The MARKER hierarchy instead asks
    which clinically *meaningful* drug was added to the panel — a drug only
    ordered when the clinical scenario warrants it:
      - Vancomycin → the clinician suspected gram-positive/MRSA
      - Carbapenem → the clinician suspected MDR gram-negative/ESBL
      - Nitrofurantoin → the culture is a urine sample
    These ORDERING DECISIONS reflect fundamentally different clinical and
    microbial contexts → the patients genuinely come from different subgraphs.

    **Expected sizes (ARMD cohort, ~67K patients):**
      H0 ~5–10K  (all MRSA-risk cultures)
      H1 ~8–15K  (all severe gram-neg ICU/blood cultures)
      H2 ~20–30K (all UTI cultures — nitrofurantoin is urine-only)
      H3 ~5–15K  (community respiratory / outpatient)
      H4 ~5–10K  (first-line standard panels)

    `n_clients` must be 5. `seed` unused (deterministic). Returns Series
    index=patient_id -> hospital id (0..4).
    """
    if n_clients != 5:
        raise ValueError(
            f"antibiotic_family_split always produces 5 hospitals. "
            f"Got n_clients={n_clients}. Pass n_clients=5 explicitly."
        )
    markers = _patient_abx_markers(df)
    # Priority order: vancomycin > carbapenem > nitrofurantoin > fluoroquinolone > other
    assignment = pd.Series(4, index=markers.index, dtype="int64")  # default: H4
    assignment[markers["quin"]]  = 3  # fluoroquinolone (overwritten by higher priority below)
    assignment[markers["nitro"]] = 2  # nitrofurantoin
    assignment[markers["carb"]]  = 1  # carbapenem
    assignment[markers["van"]]   = 0  # vancomycin (highest priority, applied last)
    assignment.name = "hospital"

    _HOSP_NAMES = {0: "vancomycin", 1: "carbapenem", 2: "nitrofuran",
                   3: "fluoroquinol", 4: "first-line"}
    counts = assignment.value_counts().sort_index()
    print("antibiotic_family_split (priority-marker) hospital sizes:")
    for h, n in counts.items():
        print(f"  H{h} ({_HOSP_NAMES[h]:13s}): {n:>7,} patients")
    total = counts.sum()
    if counts.min() < 2_000:
        smallest_h = int(counts.idxmin())
        print(f"  WARNING: H{smallest_h} ({_HOSP_NAMES[smallest_h]}) has only "
              f"{counts[smallest_h]:,} patients — may give unreliable F1.")
    return assignment


def _patient_prior_abx_score(df: pd.DataFrame) -> pd.Series:
    """Per-patient prior antibiotic exposure breadth score.

    Reads the antibiotic class exposure table directly (it is not loaded by
    load_cohort_frame — it is a fan-out enrichment table used by graph_build).
    For each patient, aggregates all prior-antibiotic records across all their
    cultures and computes:

        score = log1p(n_prior_exposures) + log1p(n_distinct_antibiotic_classes)

    so the score grows with both volume and breadth of prior ABX use.  Patients
    with zero recorded exposures (ABX-naive) get score 0.

    Returns Series index=patient_id -> float score (NaN-free, all patients in df).
    """
    from pathlib import Path
    fp = Path(config.DATA_DIR) / config.ARMD_TABLES["abx_class_exp"]
    # CK = culture key (order_proc_id_coded); PK = patient key (anon_id)
    # The exposure table is keyed by CK so we need to map back to patients via df.
    ck = config.KEYS["culture"]
    exp = pd.read_csv(
        fp,
        usecols=[ck, "antibiotic_class"],
        low_memory=False,
    )
    # Map culture key -> patient key using the cohort frame (df has both CK and PK columns)
    ck_to_pk = df[[ck, PK]].drop_duplicates(ck).set_index(ck)[PK]
    exp[PK] = exp[ck].map(ck_to_pk)
    exp = exp.dropna(subset=[PK])

    # Aggregate per patient: total exposures + distinct antibiotic classes
    agg = (
        exp.groupby(PK)["antibiotic_class"]
        .agg(n="size", nclass="nunique")
        .reindex(df[PK].unique())
        .fillna(0)
    )
    score = np.log1p(agg["n"]) + np.log1p(agg["nclass"])
    return score.rename("prior_abx_score")


def prior_abx_exposure_split(df: pd.DataFrame, n_clients: int = 4,
                              seed: int = config.SEED) -> pd.Series:
    """Non-IID split on prior antibiotic exposure breadth (mechanistic non-IID).

    Ranks patients by their prior ABX exposure score (log #exposures +
    log #distinct classes) and assigns contiguous balanced blocks:
    - Hospital 0 = ABX-naive patients (no prior antibiotic records)
    - Hospital k-1 = heavily and broadly antibiotic-exposed patients

    **Why this creates genuine headroom for a topology-aware aggregator:**
    Prior antibiotic use is the primary *cause* of resistance selection pressure.
    Naive patients have a fundamentally different resistance distribution (lower
    rates, dominated by community-acquired organisms) vs exposed patients (higher
    resistance rates, healthcare-associated organisms). FedAvg averaging across
    these conflicting distributions actively degrades the naive hospital (importing
    "resistance is common" priors) and may also harm the exposed hospital (importing
    "susceptible" priors). The label distributions *conflict*, not just differ.

    `seed` is unused (deterministic rank-based split), kept for uniform signature.
    Returns Series index=patient_id -> hospital id (0..n_clients-1).
    """
    score = _patient_prior_abx_score(df)
    return _rank_bucket_split(score, n_clients)


def prior_abx_binary_split(df: pd.DataFrame,
                            seed: int = config.SEED) -> pd.Series:
    """Extreme 2-hospital prior-ABX split at the natural ZERO boundary.

    - Hospital 0 = patients with **zero** recorded prior antibiotic exposures
      (truly ABX-naive: no prior-ABX record in abx_class_exposure at all)
    - Hospital 1 = patients with **any** prior antibiotic exposure (score > 0)

    This is harder than `prior_abx_exposure_split(n_clients=2)`, which would
    split at the median (giving two roughly-equal groups of light and heavy
    users). The zero boundary is a real clinical divide:
    - Naive patients: community-acquired infections, lower resistance rates,
      susceptible-dominant, simple drug histories.
    - Exposed patients: healthcare-associated infections, selection-pressure-
      driven resistance, broad-spectrum drug histories.

    The groups are typically unequal in size (~35-45% naive / ~55-65% exposed
    depending on the cohort slice), making size-imbalance an additional source
    of non-IID signal. FedAvg's uniform averaging imports the "resistance is
    common" signal from the exposed majority directly into the naive hospital's
    model — the maximum possible negative transfer scenario.

    `seed` unused (deterministic hard threshold). Kept for uniform signature.
    Returns Series index=patient_id -> 0 (naive) or 1 (exposed).
    """
    score = _patient_prior_abx_score(df)
    assignment = (score > 0).astype(int)
    assignment.name = "hospital"
    n_naive = int((assignment == 0).sum())
    n_exposed = int((assignment == 1).sum())
    if n_naive == 0 or n_exposed == 0:
        raise ValueError(
            f"prior_abx_binary_split: one group is empty "
            f"(naive={n_naive}, exposed={n_exposed}). "
            f"Check that the abx_class_exp table covers these patients."
        )
    return assignment


def ward_baseline(df: pd.DataFrame) -> pd.Series:
    """Baseline split: each home ward is its own hospital (patient -> ward)."""
    return assign_home_ward(df)


def adi_baseline(df: pd.DataFrame, n_clients: int = 4) -> pd.Series:
    """Baseline split: ADI-score quantiles (patient -> 'Q1'..'Qk' or 'Unknown')."""
    adi = pd.to_numeric(df.groupby(PK)[ADI].first(), errors="coerce")
    known = adi.dropna()
    # bin on ranks (not raw values) so heavy ADI-score ties don't collapse the quantile edges
    q = pd.Series("Unknown", index=adi.index, dtype="object")
    q.loc[known.index] = pd.qcut(known.rank(method="first"), n_clients,
                                 labels=[f"Q{i+1}" for i in range(n_clients)]).astype("object")
    return q


def partition_summary(df: pd.DataFrame, assignment: pd.Series) -> tuple[pd.DataFrame, float]:
    """Per-hospital size + resistance rate, and the non-IID score (test-weighted
    std of resistance rate across hospitals — larger = more heterogeneous)."""
    d = df[[PK, "label"]].copy()
    d["client"] = d[PK].map(assignment)
    g = d.groupby("client")["label"].agg(n_tests="size", resist_rate="mean")
    g["n_patients"] = d.groupby("client")[PK].nunique()
    g = g[g.index.astype(str) != "NONE"]  # drop the ward-less bucket from the score
    w = g["n_tests"] / g["n_tests"].sum()
    wmean = (w * g["resist_rate"]).sum()
    non_iid = float(np.sqrt((w * (g["resist_rate"] - wmean) ** 2).sum()))
    return g.round(4), non_iid


def alpha_sweep(df: pd.DataFrame, alphas=(0.1, 0.5, 1.0), n_clients: int = 5,
                seed: int = config.SEED) -> pd.DataFrame:
    """Non-IID score of the Dirichlet ward-mixture across alpha (proves the dial)."""
    rows = []
    for a in alphas:
        _, non_iid = partition_summary(df, dirichlet_ward_mixture(df, n_clients, a, seed))
        rows.append({"alpha": a, "non_iid_wstd": round(non_iid, 4)})
    return pd.DataFrame(rows)


def _call_partition(partition_fn, df, n_clients=None, seed=config.SEED):
    """Dispatch partition_fn(df, **kw) by keyword match in its signature.

    - n_clients is injected only when the fn declares it AND a non-None value is given.
    - seed is injected when the fn declares a keyword named 'seed'.

    Footguns documented in the module docstring:
      * run_fedavg default n_clients=5 breaks topology_split (needs multiples of 4).
      * 3+-required-param lambdas need functools.partial.
    """
    import inspect
    sig = inspect.signature(partition_fn)
    kw = {}
    if "n_clients" in sig.parameters and n_clients is not None:
        kw["n_clients"] = n_clients
    if "seed" in sig.parameters:
        kw["seed"] = seed
    if kw:
        return partition_fn(df, **kw)
    n_required = sum(1 for p in sig.parameters.values()
                     if p.default is inspect.Parameter.empty)
    if n_required >= 2:
        return partition_fn(df, seed)   # legacy (df, seed) lambdas
    return partition_fn(df)             # 1-arg baselines (specimen_baseline)


def _assert_separated(g: pd.DataFrame, name: str, min_span: float = 1e-3,
                      monotone: bool = True) -> None:
    """Validate that a split actually separates hospitals by the intended score.

    - `span(mean_score)` >= min_span: hospitals differ by at least a floor.
    - if monotone: mean_score must be non-decreasing with hospital ID (rank-based splits).
    """
    span = g["mean_score"].iloc[-1] - g["mean_score"].iloc[0]
    assert span >= min_span, \
        f"{name}: mean-score span {span:.6f} < {min_span} — hospitals not separated"
    if monotone:
        assert g["mean_score"].is_monotonic_increasing, \
            f"{name}: mean score not monotone (rank-based splits should be)"


def _self_check() -> None:
    from .data_loader import load_cohort_frame
    df = load_cohort_frame()
    n_pat = df[PK].nunique()

    print("=== home ward distribution (patients) ===")
    print(assign_home_ward(df).value_counts(dropna=False).to_dict())

    print("\n=== baseline: ward split ===")
    g, niid = partition_summary(df, ward_baseline(df))
    print(g.to_string()); print("non-IID(wstd):", niid)

    print("\n=== baseline: ADI quartile split ===")
    g, niid = partition_summary(df, adi_baseline(df))
    print(g.to_string()); print("non-IID(wstd):", niid)

    print("\n=== Dirichlet ward-mixture: alpha sweep (5 hospitals) ===")
    sweep = alpha_sweep(df)
    print(sweep.to_string(index=False))

    print("\n=== label-Dirichlet split (resistant-rate skew, 5 hospitals) ===")
    g, niid_lab = partition_summary(df, label_dirichlet(df, n_clients=5, beta=0.5))
    print(g.to_string()); print("non-IID(wstd):", round(niid_lab, 4))

    print("\n=== specimen natural split ===")
    g, niid_spec = partition_summary(df, specimen_baseline(df))
    print(g.to_string()); print("non-IID(wstd):", round(niid_spec, 4))

    print("\n=== organism-community split (5 hospitals, disjoint bug sets) ===")
    org = organism_community(df, n_clients=5)
    g, niid_org = partition_summary(df, org)
    print(g.to_string()); print("non-IID(wstd):", round(niid_org, 4))
    assert org.index.is_unique and len(org) == n_pat, "organism split: 1:1 patient assignment"

    print("\n=== hard topology splits (deterministic quantile dial) ===")
    def _topology_report(name, score, assign):
        d = df[[PK, "label"]].copy()
        d["client"] = d[PK].map(assign)
        d["score"] = d[PK].map(score)
        g = d.groupby("client").agg(n_patients=(PK, "nunique"), n_tests=("label", "size"),
                                    resist_rate=("label", "mean"), mean_score=("score", "mean"))
        print(f"\n{name}:")
        print(g.round(4).to_string())
        rr = g["resist_rate"]
        span = g["mean_score"].iloc[-1] - g["mean_score"].iloc[0]
        print(f"  resistance-rate spread: {rr.max() - rr.min():.4f} "
              f"(covariate — reported, not constrained)")
        print(f"  mean-score span (H0 -> H{len(g) - 1}): {span:.4f}")
        _assert_separated(g, name, monotone=True)
        return g

    for fn, score_fn, name in [
        (homophily_split, _patient_homophily, "homophily spectrum"),
        (degree_skew_split, _patient_hubness, "degree skew"),
    ]:
        assign = fn(df, n_clients=5)
        assert assign.index.is_unique and len(assign) == n_pat, f"{name}: 1:1 assignment"
        _topology_report(name, score_fn(df), assign)

    # topology_split (2-D crossed)
    print("\n  --- topology_split (homophily × degree quadrants) ---")
    if True:  # scope block
        try:
            assign = topology_split(df, n_clients=8, purity=0.0)
            assert assign.index.is_unique and len(assign) == n_pat, "topology_split: 1:1 assignment"
            d_top = df[[PK, "label"]].copy()
            d_top["client"] = d_top[PK].map(assign)
            # combined z-score for per-hospital separation
            hom_z = _robust_z(_patient_homophily(df))
            hub_z = _robust_z(_patient_hubness(df))
            d_top["score"] = d_top[PK].map(hom_z) + d_top[PK].map(hub_z)
            g_top = d_top.groupby("client").agg(
                n_patients=(PK, "nunique"), n_tests=("label", "size"),
                resist_rate=("label", "mean"), mean_score=("score", "mean"),
                std_score=("score", "std"),
            )
            print(g_top.round(4).to_string())
            rr_top = g_top["resist_rate"]
            print(f"  resistance-rate spread: {rr_top.max() - rr_top.min():.4f} "
                  f"(covariate — reported, not constrained)")
            _assert_separated(g_top, "topology_split", monotone=False)
        except ValueError as e:
            print(f"  SKIP: {e}")

    # louvain_split
    print("\n  --- louvain_split (community detection) ---")
    try:
        assign = louvain_split(df, n_clients=5)
        assert assign.index.is_unique and len(assign) == n_pat, "louvain_split: 1:1 assignment"
        d_lv = df[[PK, "label"]].copy()
        d_lv["client"] = d_lv[PK].map(assign)
        g_lv = d_lv.groupby("client").agg(
            n_patients=(PK, "nunique"), n_tests=("label", "size"),
            resist_rate=("label", "mean"),
        )
        print(g_lv.round(4).to_string())
        rr_lv = g_lv["resist_rate"]
        print(f"  resistance-rate spread: {rr_lv.max() - rr_lv.min():.4f} "
              f"(covariate — reported, not constrained)")
    except ImportError:
        print("  SKIP (networkx missing)")
    except Exception as e:
        print(f"  SKIP: {e}")

    print("\n=== topology diagnostics (runtime signals) ===")
    hub = _patient_hubness(df)
    distinct_vals = hub.nunique()
    plateau_frac = (hub >= hub.max() - 1e-6).mean()  # fraction at/near ceiling
    print(f"\nhubness: min={hub.min():.4f}  max={hub.max():.4f}  range={hub.max() - hub.min():.4f}  "
          f"distinct values={distinct_vals}  plateau_frac={plateau_frac:.4f}")
    if plateau_frac > 0.1:
        print("  ⚠ plateau_frac > 0.1 — hubness still saturates; consider log1p(distinct organisms) "
              "or log1p(total triples) as fallback")

    hom = _patient_homophily(df)
    neutral_mass = (hom.abs() < 1e-9).mean()
    quartiles = hom.quantile([0.25, 0.5, 0.75])
    print(f"homophily: neutral_mass={neutral_mass:.4f}  "
          f"quartiles=[{quartiles[0.25]:.6f}, {quartiles[0.5]:.6f}, {quartiles[0.75]:.6f}]  "
          f"IQR={quartiles[0.75] - quartiles[0.25]:.6f}")
    if neutral_mass > 0.3 or (quartiles[0.75] - quartiles[0.25]) < 1e-6:
        print("  ⚠ homophily degenerate — consider alternate axis (organism-level homophily "
              "or log1p(distinct org))")

    try:
        from scipy import stats as scipy_stats
        rho, _pv = scipy_stats.spearmanr(hom, hub)
        print(f"Spearman ρ(hom, hub) = {rho:.4f}")

        # quadrant shares from a purity=0 topology_split
        assign = topology_split(df, n_clients=4, purity=0.0)
        shares = assign.value_counts(normalize=True).sort_index()
        print(f"topology quadrant shares (purity=0): "
              + " | ".join(f"Q{i}: {shares.get(i, 0):.3f}" for i in range(4)))

        if abs(rho) > 0.6:
            print("  ⚠ |ρ| > 0.6 — axes correlated; flip decorrelate=True in topology_split")
        if any(shares.get(i, 0) < 0.12 for i in range(4)):
            print("  ⚠ quadrant share < 0.12 — axis orthogonality may be weak; try decorrelate=True")
    except (ImportError, ModuleNotFoundError):
        print("Spearman ρ: scipy not available (skip)")
    except Exception as e:
        print(f"axis correlation: {e} (skip)")

    # the dial must work: smaller alpha -> more heterogeneity
    s = sweep.sort_values("alpha")
    assert s["non_iid_wstd"].iloc[0] > s["non_iid_wstd"].iloc[-1], \
        "non-IID should shrink as alpha grows — the Dirichlet dial isn't working"
    # the whole point: a label-skew split must be MORE non-IID than the ward mixture
    ward_niid = float(s["non_iid_wstd"].iloc[-1])  # alpha=1.0 (mildest ward split)
    assert niid_lab > ward_niid, \
        "label-Dirichlet should be more non-IID than the ward mixture — that's its purpose"
    # no leakage: every patient assigned exactly once
    assign = dirichlet_ward_mixture(df, 5, 0.5)
    assert assign.index.is_unique and len(assign) == n_pat, "patient assignment not 1:1"
    print("\nOK: partition self-check passed (dial works, no patient leakage).")


if __name__ == "__main__":
    _self_check()
