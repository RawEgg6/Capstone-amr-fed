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
              f"(label-Dirichlet is ~0.3+; should stay small = structural, not a label shift)")
        print(f"  mean-score span (H0 -> H{len(g) - 1}): {span:.4f}")
        assert g["mean_score"].is_monotonic_increasing, f"{name}: mean score not monotone"
        assert rr.max() - rr.min() < 0.20, f"{name}: resistance-rate spread too large"
        return g

    for fn, score_fn, name in [
        (homophily_split, _patient_homophily, "homophily spectrum"),
        (degree_skew_split, _patient_hubness, "degree skew"),
    ]:
        assign = fn(df, n_clients=5)
        assert assign.index.is_unique and len(assign) == n_pat, f"{name}: 1:1 assignment"
        _topology_report(name, score_fn(df), assign)

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
