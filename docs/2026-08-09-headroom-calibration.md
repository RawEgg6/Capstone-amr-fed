# Headroom Calibration Record — 2026-08-09

## Gate definition

`headroom_gate(partition_fn, ...)` in `src/amr_fed/federated/run.py` is the acceptance test
for whether a split is "hard enough" — i.e., FedAvg measurably trails pooled, giving the
Phase-5 topology-aware aggregator real headroom.

Three conditions must ALL hold (mean over seeds):

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| `fed_helps` | `FedAvg_best > local_only` | FedAvg must still help vs training alone |
| `gap_ok` | `(worst_fed ≤ pooled_worst − 0.02)` OR `(FedAvg_best ≤ pooled − 0.01)` | FedAvg trails pooled at the worst hospital OR on average |
| `pooled_ok` | `pooled_worst ≥ 0.60` | Pooled stays strong — the GNN can learn all regimes jointly |

## Calibration knobs

| Knob | Default | Direction to widen gap | Direction to lift FedAvg |
|------|---------|----------------------|--------------------------|
| `purity` (topology_split) | 0.0 | Lower → purer quadrants, wider FedAvg-pooled gap | Raise (0.2) → mixes patients across quadrants, lifts FedAvg off the floor |
| `hidden` | 64 | Lower (32) — narrow model hurts FedAvg's averaging more than pooled's joint GD | Raise (128) — wider model closes the gap |
| `rounds` × `local_epochs` | 6 × 3 = 18 | Less budget (4 × 2 = 8) — FedAvg wastes budget on client drift | More budget (8 × 4 = 32) |
| `n_clients` | 8 | More clients (up to surgical minimum) — averaging drifts further | Fewer clients (4) — larger hospitals, closer to pooled |
| `decorrelate` | False | True — if homophily/hubness axes are correlated, residualize to create purer quadrants | N/A |

`purity` is the **FedAvg-lifting dial**: pooled trains on the union regardless of partition,
so mixing a few patients across quadrants lifts FedAvg while barely affecting pooled.

## Protocol

Starting calibrated protocol (from the "Calibrate protocol" decision in the plan):

- `n_clients`: 8 (topology) / 5 (louvain) — more than the old 5
- `rounds`: 6, `local_epochs`: 3 → matched budget = 18 epochs (old: 10 × 6 = 60)
- `hidden`: 64 (old: 128) — inverted widening-theorem lever
- Seeds: 42, 43, 44

## Winning configuration

> **🟡 PENDING — run on Colab with ARMD data mounted.**

### topology_split (2-D crossed quadrants)

| Run | purity | n_clients | rounds | local_epochs | hidden | Pooled (mean±std) | FedAvg-best (mean±std) | Local-only (mean±std) | Worst-Fed (mean±std) | Worst-Pooled (mean±std) | PASS/FAIL |
|-----|--------|-----------|--------|--------------|--------|-------------------|------------------------|----------------------|---------------------|------------------------|-----------|
| 1 | 0.0 | 8 | 6 | 3 | 64 | — | — | — | — | — | — |

### louvain_split (community detection)

| Run | n_clients | rounds | local_epochs | hidden | Pooled (mean±std) | FedAvg-best (mean±std) | Local-only (mean±std) | Worst-Fed (mean±std) | Worst-Pooled (mean±std) | PASS/FAIL |
|-----|-----------|--------|--------------|--------|-------------------|------------------------|----------------------|---------------------|------------------------|-----------|
| 1 | 5 | 6 | 3 | 64 | — | — | — | — | — | — |

## Headroom attribution story (Phase 5)

When the gate passes:

1. **Topology divergence is measurable** — the split creates structurally different hospitals
   (homophily × degree quadrants; Louvain communities on the test graph).
2. **Resistance rate is a reported covariate** — per-hospital rate is documented alongside
   topology stats, not held constant. The Phase-5 story: topology-aware aggregation recovers
   **structurally different AND label-shifted worst hospitals**.
3. **The headroom is real** — pooled stays strong (≥0.60 worst hospital) while FedAvg
   trails by a measurable margin, giving the Phase-5 aggregator a target to beat.
4. **The protocol is calibrated** — more clients + shorter matched budget + narrower model
   expose heterogeneity that the old 5-client/128-width/60-epoch protocol hid.

## Runtime diagnostics (from Task 6 _self_check)

> **🟡 PENDING — run `python -m amr_fed.partition` on Colab/data machine.**

- Hubness plateau fraction: —
- Homophily neutral mass: —
- Spearman ρ(hom, hub): —
- Quadrant shares (purity=0): —
- Decision: `decorrelate=True`? (|ρ| > 0.6 or any share < 0.12 → yes)
