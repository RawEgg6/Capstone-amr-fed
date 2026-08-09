# Headroom Calibration Record — 2026-08-09

## Gate definition

`headroom_gate(partition_fn, ...)` in `src/amr_fed/federated/run.py` is the acceptance test
for whether a split is "hard enough" — i.e., FedAvg measurably trails pooled, giving the
Phase-5 topology-aware aggregator real headroom.

Three conditions must ALL hold (mean over 3 seeds):

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| `fed_helps` | `FedAvg_best > local_only` | FedAvg must still help vs training alone |
| `gap_ok` | `(worst_fed ≤ pooled_worst − 0.02)` OR `(FedAvg_best ≤ pooled − 0.01)` | FedAvg trails pooled at the worst hospital OR on average |
| `pooled_ok` | `pooled_worst ≥ 0.60` | Pooled stays strong — the GNN can learn all regimes jointly |

## Calibration knobs

| Knob | Effect |
|------|--------|
| `purity` (topology_split) | **FedAvg-lifting dial**: 0.0 = pure quadrants (max divergence); 0.2 = mixed, lifts FedAvg off floor. Pooled barely moves. |
| `hidden` | Narrower (64→32) = more FedAvg-pooled gap. Wider (128) = restores pooled strength. |
| `rounds × local_epochs` | More budget = pooled trains better, closes the headroom gap. Less = FedAvg's 8× gradient-step advantage grows. |
| `n_clients` | More hospitals = averaging drifts further, FedAvg degrades faster than pooled. |
| `decorrelate` | True if homophily/hubness axes correlated → purer quadrants. Check `_self_check` diagnostic. |

## Protocol history

### v1 (2026-08-09, hidden=64, 6r × 3e = 18 budget)

| Split | n | Pooled (mean±std) | FedAvg-best | Local-only | Worst local→Fed | vs Pooled | PASS? |
|-------|---|-------------------|-------------|------------|------------------|-----------|-------|
| homophily | 8 | 0.6732±.0065 | 0.7014±.0025 | 0.6918±.0017 | 0.6695→0.6930 | **+0.028** | ❌ |
| degree-skew | 8 | 0.6785±.0056 | 0.6928±.0085 | 0.6862±.0014 | 0.6371→0.6790 | **+0.014** | ❌ |
| topology-corners | 8 | 0.6881±.0016 | 0.7036±.0014 | 0.6900±.0076 | 0.6044→0.6751 | **+0.015** | ❌ |
| louvain | 3* | 0.6596±.0096 | 0.6976±.0009 | 0.6877±.0062 | 0.6722→0.6614 | **+0.038** | ❌ |

*Louvain: requested 5 hospitals, got 3 communities (sizes ~73%/21%/5%). Worst hospital got WORSE with FedAvg — right shape, pooled was too weak.*

**Root cause:** hidden=64 + 18-epoch matched budget under-trained pooled (0.66–0.69 vs old Phase 1 reference 0.71). Narrow model didn't hurt FedAvg's averaging more than pooled's joint GD — FedAvg's 8× gradient-step advantage dominated at short budgets.

### v2 (current, hidden=128, 8r × 4e = 32 budget)

Wider model + more budget to restore pooled strength (~0.71 target). FedAvg's 8× gradient-step advantage should shrink as total budget grows.

| Split | purity | n | rounds | loc_ep | hidden | Pooled | FedAvg-best | Local | Worst-Fed | Worst-Pooled | PASS? |
|-------|--------|---|--------|--------|--------|--------|-------------|-------|-----------|-------------|-------|
| topology | 0.0 | 8 | 8 | 4 | 128 | — | — | — | — | — | — |
| louvain | — | 5* | 8 | 4 | 128 | — | — | — | — | — | — |

## Splits removed from calibration runs

**homophily** and **degree-skew** (single-axis rank splits) were run at v1 and showed zero
headroom — FedAvg beat pooled by +0.028 and +0.014 respectively. Even at hidden=64/18-budget,
single-axis divergence is too mild. These splits are useful for Phase 5 sensitivity analysis
(does topology-aware aggregation help even when FedAvg already wins?) but are not calibration targets.

## Headroom attribution story (Phase 5)

When the gate passes:

1. **Topology divergence is measurable** — homophily × degree quadrants; Louvain communities on the test graph
2. **Resistance rate is a reported covariate** — per-hospital rate documented alongside topology stats, not held constant
3. **The headroom is real** — pooled stays strong (≥0.60 worst hospital) while FedAvg trails
4. **Phase 5 story:** topology-aware aggregation recovers structurally different AND label-shifted worst hospitals

## Runtime diagnostics

> **🟡 PENDING — run `python -m amr_fed.partition` on Colab/data machine.**

- Hubness plateau fraction: —
- Homophily neutral mass: —
- Spearman ρ(hom, hub): —
- Quadrant shares (purity=0): —
- Decision: `decorrelate=True`? (|ρ| > 0.6 or any share < 0.12 → yes)
