# Hard topology splits for Phase-5 headroom

Date: 2026-08-09
Status: approved in principle (assignment strategy = deterministic quantile dial)

## Goal

Create two controlled, deterministic patient→hospital partitions whose induced client
graphs diverge **topologically** (not just in label rate), so FedAvg's single shared GNN
degrades on the extremes and the Phase-5 topology-aware aggregator has measurable
headroom. Headline metric = worst-hospital and per-hospital deltas; mean F1 is context,
not the claim (FedAvg already matches the pooled ceiling on mean F1).

The peer-reviewed recipe this follows: **structure non-IID / homophily divergence**
(AdaFGL 2024), **topology distribution skew** (OpenFGL 2024), community-style splits
(FedGTA, VLDB 2024).

## Splits

Both return the standard contract used by `run_fedavg(partition_fn=...)`:
`Series(index=patient -> client id 0..k-1)`, 1:1 assignment (no leakage), balanced sizes.

### 1. `homophily_split(df, n_clients=5, seed=config.SEED)`

Structure non-IID. Steps:

1. Binary label per `(organism, antibiotic)` tested edge: `b(o,a) = (rate(o,a) >= 0.5)`
   over that pair's triples.
2. Local agreement of edge `e=(o,a)` = fraction of its *neighbor* tested-edges (sharing
   organism `o` OR antibiotic `a`) with the same binary label.
3. Deviation `a(e) = agreement(e) − P(b(e))`, where `P(b)` = global fraction of tested
   edges with label `b` (~0.19 for resistant). This is an assortativity-style deviation:
   it removes the ambient resistance rate, so the contrast is **structural**, not "one
   hospital is just more resistant."
4. Per-patient score = mean deviation over the patient's triples.
5. Assign by rank quantile blocks (below): hospital 0 = most heterophilic (resistance
   scattered), hospital k−1 = most homophilic (resistance clustered).

A single shared GNN cannot fit both regimes: in the homophilic hospital message-passing
amplifies resistance, in the heterophilic hospital it dilutes it.

### 2. `degree_skew_split(df, n_clients=5, seed=config.SEED)`

Topology distribution skew. Steps:

1. Per-organism tested-degree = number of distinct antibiotics the organism was tested
   against (degree in the `(organism, tested, antibiotic)` bipartite subgraph).
2. Per-patient score = triple-weighted mean of their organisms' tested-degree.
3. Assign by rank quantile blocks: hospital 0 = sparse / rare bugs (low degree),
   hospital k−1 = hub / common bugs (tested against most antibiotics, dense well-connected
   subgraph).

Contrasting degree distributions → the shared weights fit one regime but not the other
(e.g. sparse graphs barely message-pass on `tested`; the model leans on node + triple
features there).

## Assignment: deterministic quantile dial (chosen)

Rank patients by score (stable mergesort — deterministic ties), split into `n_clients`
contiguous, balanced blocks (largest-remainder via the existing `_apportion`). Hospital 0
is the most extreme end of the spectrum, hospital k−1 the opposite extreme. Deterministic,
seed-independent, reproducible. This guarantees monotone topology contrast — exactly the
controlled dial the headroom experiment needs. (`seed` kept in the signature for a uniform
`partition_fn(df, seed)` contract; unused.)

## Integration

- **`src/amr_fed/partition.py`**: add `homophily_split`, `degree_skew_split`, shared
  `_rank_bucket_split(score, n_clients)`, plus small private score helpers
  (`_tested_edge_homophily`, `_patient_hubness`) reused by the self-check.
- **Self-check** (extend `_self_check()`): per-hospital table
  `{n_patients, n_tests, resist_rate, mean_score}`; assert (a) 1:1 assignment,
  (b) mean score is strictly monotone across hospital ids, (c) resistance-rate spread
  stays small (structural, not label shift — compare against label-Dirichlet's spread).
- **`tests/test_partition.py`**: offline unit tests on tiny synthetic frames — every
  patient assigned exactly once; balanced sizes; monotone mean score; resistance-rate
  constancy. No data dependency (runs anywhere, matches existing style).
- **`src/amr_fed/federated/run.py`**: fix the `partition_fn(df, seed)` positional call
  (latent bug: it would bind `seed`→`n_clients` for 3-arg signatures; the notebook
  currently works around it with lambdas). Replace with signature-aware dispatch.
- **`notebooks/04_federated.ipynb`**: add cells 9 & 10 running both splits through
  `run_multiseed(seeds=(42,43,44))`, mirroring the organism-community cell.

## Headroom verification

Run `run_multiseed` on both splits. Expected signature of success:
- `local-only` drops on the extreme hospitals (the topology is genuinely hard there),
- FedAvg recovers some of it, worst-hospital gain ≥ organism-community's +0.037,
- resistance rates roughly constant across hospitals (proving the divergence is structural).

That recovered-but-not-full gap is the headroom the Phase-5 topology-aware aggregator
exploits.

## Non-goals

- No per-client graph schema / feature changes (breaks FedAvg weight alignment — feature
  dims must match for state-dict averaging).
- No FedProx / FedGTA implementation yet — this is the dataset + partition groundwork for
  Phase 5.
- No personalization experiments yet (Phase 5 territory).
