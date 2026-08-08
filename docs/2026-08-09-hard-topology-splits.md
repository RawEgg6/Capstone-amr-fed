# Hard Topology Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two deterministic "hard" topology splits (`homophily_split`, `degree_skew_split`) to `src/amr_fed/partition.py` so FedAvg has measurable headroom for the Phase-5 topology-aware aggregator.

**Architecture:** Both splits compute a per-patient topology score from the cohort frame (pure pandas, no torch), then assign patients to hospitals with a shared rank-based quantile dial (`_rank_bucket_split`) — hospital 0 is the most extreme end of the spectrum, hospital k−1 the opposite extreme. Both plug into the existing `run_fedavg(partition_fn=...)` contract unchanged.

**Tech Stack:** Python, pandas, numpy. No torch/PyG. Tests run with `python tests/test_partition.py` (plain asserts; pytest also works via `pyproject.toml`).

## Global Constraints

- Signatures: every split is `(df: pd.DataFrame, n_clients: int = 5, seed: int = config.SEED) -> pd.Series`, `Series(index=patient -> client id 0..k-1)`.
- Assignment is 1:1 (no patient in two hospitals, none left unassigned) and deterministic (`seed` kept for a uniform `partition_fn(df, seed)` contract but unused — the dial is seed-independent).
- Resistance rate must stay roughly constant across hospitals (the divergence is structural, not a label-rate shift).
- Pure pandas/numpy — `partition.py` has no torch dependency; do not import torch.
- No new graph schema / per-client features (feature dims must stay identical for FedAvg weight alignment).

---
## Task 1: Shared rank-based quantile dial — `_rank_bucket_split`

**Files:**
- Modify: `src/amr_fed/partition.py` (insert helper after `_apportion`, ~line 48)
- Test: `tests/test_partition.py`

**Interfaces:**
- Produces: `_rank_bucket_split(score: pd.Series, n_clients: int) -> pd.Series` — `score` indexed by patient, returns `Series(index=patient -> client id 0..n_clients-1)`. Hospital 0 = lowest scores, hospital k−1 = highest. Balanced block sizes via existing `_apportion`; raises `ValueError` if `n_clients < 1` or `n_clients > len(score)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_partition.py` (before the `if __name__ == "__main__":` block) and register it in the runner:

```python
from amr_fed.partition import _rank_bucket_split


def test_rank_bucket_split_quantile_dial():
    score = pd.Series({"pA": 5.0, "pB": 1.0, "pC": 9.0, "pD": 3.0, "pE": 7.0, "pF": 2.0})
    a = _rank_bucket_split(score, 2)
    assert a.index.is_unique and len(a) == 6
    assert set(a.unique()) == {0, 1}
    # balanced sizes: 3 / 3
    assert list(a.value_counts().sort_index()) == [3, 3]
    # hospital 0 holds the lowest scores, hospital 1 the highest
    assert set(a.index[a == 0]) == {"pB", "pD", "pF"}
    assert set(a.index[a == 1]) == {"pA", "pC", "pE"}
    # deterministic
    assert _rank_bucket_split(score, 2).equals(a)


def test_rank_bucket_split_guards():
    score = pd.Series({"pA": 1.0, "pB": 2.0})
    try:
        _rank_bucket_split(score, 0)
        raise AssertionError("should have raised for n_clients < 1")
    except ValueError:
        pass
    try:
        _rank_bucket_split(score, 5)
        raise AssertionError("should have raised for n_clients > #patients")
    except ValueError:
        pass
```

Register in the runner block:

```python
    test_rank_bucket_split_quantile_dial()
    test_rank_bucket_split_guards()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_partition.py`
Expected: FAIL — `ImportError: cannot import name '_rank_bucket_split'`

- [ ] **Step 3: Write minimal implementation**

Insert after the `_apportion` function in `src/amr_fed/partition.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_partition.py`
Expected: PASS — `OK: partition unit tests passed.`

- [ ] **Step 5: Commit**

```bash
git add tests/test_partition.py src/amr_fed/partition.py
git commit -m "Add rank-based quantile dial for deterministic topology splits"
```

---
## Task 2: Homophily score — `_tested_edge_homophily` + `_patient_homophily`

**Files:**
- Modify: `src/amr_fed/partition.py` (add after `_rank_bucket_split`)
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: `_rank_bucket_split` (Task 1)
- Produces:
  - `_tested_edge_homophily(df: pd.DataFrame) -> pd.DataFrame` — indexed by `(ORG, ABX)`, columns `b` (binary majority label) and `dev` (assortativity-style homophily deviation; NaN-free).
  - `_patient_homophily(df: pd.DataFrame) -> pd.Series` — `Series(index=patient -> mean deviation over their triples)`, NaN-free.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_partition.py` and register:

```python
from amr_fed.partition import _patient_homophily


def test_patient_homophily_clustered_vs_scattered():
    # o_clust: every tested edge majority-RESISTANT -> homophilic (dev > 0)
    # o_mix:   half-S half-R -> heterophilic (dev < 0)
    rows = []
    for abx in ["a0", "a1", "a2"]:
        for p in ["h1", "h2", "h3", "h4"]:
            rows.append((p, "o_clust", abx, 1))
    for p in ["x1", "x2", "x3", "x4"]:
        rows.append((p, "o_mix", "a3", 0))
        rows.append((p, "o_mix", "a4", 1))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    score = _patient_homophily(df)
    assert (score.loc[["h1", "h2", "h3", "h4"]] > 0).all()          # homophilic
    assert (score.loc[["x1", "x2", "x3", "x4"]] < 0).all()          # heterophilic
    assert score.index.is_unique and not score.isna().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_partition.py`
Expected: FAIL — `ImportError: cannot import name '_patient_homophily'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/amr_fed/partition.py`:

```python
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
    agree_o = np.where(deg_o > 1, same_o / (deg_o - 1), np.nan)
    agree_a = np.where(deg_a > 1, same_a / (deg_a - 1), np.nan)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_partition.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_partition.py src/amr_fed/partition.py
git commit -m "Add assortativity-style homophily score for tested edges"
```

---
## Task 3: `homophily_split`

**Files:**
- Modify: `src/amr_fed/partition.py` (add after `_patient_homophily`)
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: `_rank_bucket_split` (Task 1), `_patient_homophily` (Task 2)
- Produces: `homophily_split(df: pd.DataFrame, n_clients: int = 5, seed: int = config.SEED) -> pd.Series` — hospital 0 = most heterophilic, hospital k−1 = most homophilic.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_partition.py` and register:

```python
from amr_fed.partition import homophily_split


def test_homophily_split_separates_spectrum():
    rows = []
    for abx in ["a0", "a1", "a2"]:
        for p in ["h1", "h2", "h3", "h4"]:
            rows.append((p, "o_clust", abx, 1))
    for p in ["x1", "x2", "x3", "x4"]:
        rows.append((p, "o_mix", "a3", 0))
        rows.append((p, "o_mix", "a4", 1))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    a = homophily_split(df, n_clients=2)
    assert a.index.is_unique and len(a) == 8
    assert set(a.unique()) == {0, 1}
    # hospital 0 = heterophilic patients, hospital 1 = homophilic patients
    assert set(a.index[a == 0]) == {"x1", "x2", "x3", "x4"}
    assert set(a.index[a == 1]) == {"h1", "h2", "h3", "h4"}
    assert homophily_split(df, n_clients=2).equals(a)          # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_partition.py`
Expected: FAIL — `ImportError: cannot import name 'homophily_split'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/amr_fed/partition.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_partition.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_partition.py src/amr_fed/partition.py
git commit -m "Add homophily spectrum split (structure non-IID)"
```

---
## Task 4: `_patient_hubness` + `degree_skew_split`

**Files:**
- Modify: `src/amr_fed/partition.py` (add after `homophily_split`)
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: `_rank_bucket_split` (Task 1)
- Produces:
  - `_patient_hubness(df: pd.DataFrame) -> pd.Series` — `Series(index=patient -> triple-weighted mean of their organisms' tested-degree)`.
  - `degree_skew_split(df: pd.DataFrame, n_clients: int = 5, seed: int = config.SEED) -> pd.Series` — hospital 0 = sparse/rare bugs, hospital k−1 = hub/common bugs.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_partition.py` and register:

```python
from amr_fed.partition import _patient_hubness, degree_skew_split


def test_degree_skew_split_sparse_vs_hub():
    rows = []
    for p in ["c1", "c2", "c3", "c4"]:                       # common-bug patients
        for abx in ["a0", "a1", "a2", "a3"]:
            rows.append((p, "o_common", abx, 0))
    for p in ["r1", "r2", "r3", "r4"]:                       # rare-bug patients
        rows.append((p, "o_rare", "a0", 0))
    df = pd.DataFrame(rows, columns=[PK, ORG, ABX, "label"])
    hub = _patient_hubness(df)
    assert (hub.loc[["c1", "c2", "c3", "c4"]] == 4).all()
    assert (hub.loc[["r1", "r2", "r3", "r4"]] == 1).all()
    a = degree_skew_split(df, n_clients=2)
    assert set(a.index[a == 0]) == {"r1", "r2", "r3", "r4"}  # sparse -> hospital 0
    assert set(a.index[a == 1]) == {"c1", "c2", "c3", "c4"}  # hubs -> hospital 1
    assert degree_skew_split(df, n_clients=2).equals(a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_partition.py`
Expected: FAIL — `ImportError: cannot import name '_patient_hubness'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/amr_fed/partition.py`:

```python
def _patient_hubness(df: pd.DataFrame) -> pd.Series:
    """Per-patient hubness = triple-weighted mean of their organisms' tested-degree
    (# distinct antibiotics each bug was tested against). NaN-free."""
    d = df[[PK, ORG, ABX]]
    org_deg = d.groupby(ORG)[ABX].nunique().rename("deg")
    w = (d.groupby([PK, ORG]).size().rename("w").reset_index()
          .merge(org_deg, on=ORG))
    w["prod"] = w["w"] * w["deg"]
    g = w.groupby(PK).agg(w=("w", "sum"), prod=("prod", "sum"))
    return (g["prod"] / g["w"]).astype(float)


def degree_skew_split(df: pd.DataFrame, n_clients: int = 5,
                      seed: int = config.SEED) -> pd.Series:
    """Topology distribution skew (OpenFGL 2024): hospitals differ in graph degree.

    Each patient is scored by the triple-weighted mean tested-degree of their
    organisms; patients are ranked and assigned to contiguous, balanced blocks.
    Hospital 0 = SPARSE (rare bugs, few tested edges), hospital k-1 = HUB-heavy
    (common bugs tested against most antibiotics) — contrasting degree distributions
    the shared GNN cannot fit equally. `seed` unused (deterministic).
    Returns Series index=patient -> client id."""
    return _rank_bucket_split(_patient_hubness(df), n_clients)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_partition.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_partition.py src/amr_fed/partition.py
git commit -m "Add degree-skew split (topology distribution skew)"
```

---
## Task 5: Real-data self-check reporting

**Files:**
- Modify: `src/amr_fed/partition.py` (`_self_check`, after the organism-community block ~line 209)

**Interfaces:**
- Consumes: `homophily_split`, `degree_skew_split`, `_patient_homophily`, `_patient_hubness`, `partition_summary` (all previous tasks)

- [ ] **Step 1: Add the report block to `_self_check`**

After the organism-community section (line `assert org.index.is_unique ...`) and before the existing dial assertions, insert:

```python
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
        print(f"  mean-score span (H0 -> H{k}): {span:.4f}")
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
```

- [ ] **Step 2: Run the self-check on real data**

Run: `python -m amr_fed.partition` (requires `ARMD_DIR` pointing at the shared data mount)
Expected: per-hospital tables for both splits; `monotone` + `spread < 0.20` assertions pass; final `OK: partition self-check passed.`

If `spread < 0.20` fails on real data, loosen to a printed diagnostic (do not silently change the measure).

- [ ] **Step 3: Commit**

```bash
git add src/amr_fed/partition.py
git commit -m "Self-check: report per-hospital topology stats for hard splits"
```

---
## Task 6: Robust `partition_fn` dispatch in `run_fedavg`

**Files:**
- Modify: `src/amr_fed/federated/run.py:131` (the `partition_fn(df, seed)` call) and add a module-level helper near `POOLED_REFERENCE`.

**Interfaces:**
- Consumes: nothing new
- Produces: `_call_partition(partition_fn, df, seed)` — calls `partition_fn` correctly for any arity: 1-arg baselines (`specimen_baseline(df)`), 2-arg lambdas (`lambda d, s: ...`), and 3-arg splits with a `seed` kwarg (`homophily_split`, `degree_skew_split`, `organism_community`, `label_dirichlet`).

- [ ] **Step 1: Add the helper**

Insert after `POOLED_REFERENCE = 0.71` in `src/amr_fed/federated/run.py`:

```python
def _call_partition(partition_fn, df, seed):
    """Call a partition function with the right arity. Public contract is
    partition_fn(df, seed), but baselines vary: specimen_baseline(df) takes only df,
    the notebook uses 2-arg lambdas, and the topology splits take (df, n_clients, seed).
    Dispatch on the signature so all of them work identically."""
    import inspect
    sig = inspect.signature(partition_fn)
    params = list(sig.parameters.values())
    n_required = sum(1 for p in params if p.default is inspect.Parameter.empty)
    if "seed" in sig.parameters:
        return partition_fn(df, seed=seed)
    if n_required >= 2:
        return partition_fn(df, seed)
    return partition_fn(df)
```

- [ ] **Step 2: Use it in `run_fedavg`**

Replace the `raw = (...)` line in `run_fedavg`:

```python
    raw = (dirichlet_ward_mixture(df, n_clients=n_clients, alpha=alpha, seed=seed)
           if partition_fn is None else _call_partition(partition_fn, df, seed))
```

- [ ] **Step 3: Verify imports + no regressions**

Run: `python -c "from amr_fed.federated.run import _call_partition, run_fedavg; print('ok')"`
Expected: prints `ok`. Also verify the three dispatch branches with a one-liner:

```bash
python - <<'PY'
from amr_fed.federated.run import _call_partition
import pandas as pd
df = pd.DataFrame({
    "anon_id": [f"p{i}" for i in range(8)],
    "organism": ["o1", "o2"] * 4,
    "antibiotic": ["a0", "a1", "a2", "a3"] * 2,
    "label": [1, 0, 1, 0, 1, 0, 1, 0],
    "culture_description": ["URINE", "BLOOD"] * 4,
    "ward": ["ICU", "OP"] * 4,
})
# 1-arg baseline
s = _call_partition(lambda d: d["anon_id"], df, 42); assert len(s) == 8
# 2-arg lambda (the notebook style)
s = _call_partition(lambda d, seed: d["anon_id"], df, 42); assert len(s) == 8
# seed-kwarg splits (default n_clients=5 needs >= 5 patients); specimen_baseline has no seed -> 1-arg path
from amr_fed.partition import specimen_baseline, homophily_split
print(_call_partition(specimen_baseline, df, 42).to_dict())
print(_call_partition(homophily_split, df, 42).to_dict())
print("dispatch ok")
PY
```

Expected: prints `dispatch ok` (the toy frame produces tiny splits; this only checks the dispatch branches, not the split quality).

- [ ] **Step 4: Commit**

```bash
git add src/amr_fed/federated/run.py
git commit -m "Robust partition_fn dispatch in run_fedavg (fixes arity bug)"
```

---
## Task 7: Notebook experiment cells

**Files:**
- Modify: `notebooks/04_federated.ipynb` (append two code cells after cell 8)

**Interfaces:**
- Consumes: `homophily_split`, `degree_skew_split` (Tasks 3, 4), `run_multiseed`

- [ ] **Step 1: Append cell 9 (homophily)**

Add a new code cell at the end of the notebook:

```python
# 9) HARD SPLIT #1: HOMOPHILY spectrum -- hospitals span heterophilic <-> homophilic.
# Resistance RATE is held ~constant; the divergence is structural (resistance clustering).
# The shared GNN should degrade on the extremes -> headroom for the Phase-5 topology aggregator.
import sys
for _m in [m for m in list(sys.modules) if m.startswith('amr_fed')]:
    del sys.modules[_m]
from amr_fed.federated.run import run_multiseed
from amr_fed.data_loader import load_cohort_frame
from amr_fed.partition import homophily_split

df = load_cohort_frame()
run_multiseed(seeds=(42, 43, 44), df=df, label="homophily", partition_fn=homophily_split)
```

- [ ] **Step 2: Append cell 10 (degree skew)**

Add another new code cell:

```python
# 10) HARD SPLIT #2: DEGREE SKEW -- hospitals span sparse <-> hub-heavy organism graphs.
# Contrasting degree distributions; the shared GNN can't fit both regimes equally.
import sys
for _m in [m for m in list(sys.modules) if m.startswith('amr_fed')]:
    del sys.modules[_m]
from amr_fed.federated.run import run_multiseed
from amr_fed.data_loader import load_cohort_frame
from amr_fed.partition import degree_skew_split

df = load_cohort_frame()
run_multiseed(seeds=(42, 43, 44), df=df, label="degree-skew", partition_fn=degree_skew_split)
```

- [ ] **Step 3: Validate the notebook is still parseable**

Run: `python -c "import json; json.load(open('notebooks/04_federated.ipynb')); print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_federated.ipynb
git commit -m "Notebook: experiment cells for homophily + degree-skew splits"
```

---
## Final Verification

- [ ] Run `python tests/test_partition.py` — all unit tests pass.
- [ ] Run `python -m amr_fed.partition` (with `ARMD_DIR`) — self-check passes with monotone topology + small resistance-rate spread.
- [ ] Run the run.py dispatch one-liner from Task 6 — prints `dispatch ok`.
- [ ] Notebook parses (`json.load` ok).
