# Phase 5 results — topology-aware vs FedAvg (Path-B gate evaluation)

Recorded 2026-10-18. **Provenance:** numbers transcribed from the `07_topology_comparison.ipynb`
JSON output pasted into chat after the Colab run (~2026-09-28). They are **not yet
re-verified** against the saved file — re-check against Drive
`amr_topology_comparison.json` before quoting any of these in slides/paper.

**Protocol (both splits):** distinctiveness mode, T = 1.0, 3 seeds (42/43/44),
matched budgets, hidden 128, full 1.6M-row cohort, 5 clients (organism-community) /
3 clients (specimen). Gate (Path-B, pre-registered): topology-aware beats FedAvg-best
by ≥ 0.005 on worst-hospital **or** mean macro-F1.

## Verdict

| Split | FedAvg best / worst | Topology best / worst | Δ mean | Δ worst | Gate |
|---|---|---|---|---|---|
| organism-community | 0.7074 ± 0.0091 / 0.6897 ± 0.0071 | 0.7084 ± 0.0077 / 0.6951 ± 0.0078 | +0.0010 | **+0.0054** | **PASS** |
| specimen | 0.7111 ± 0.0056 / 0.6915 ± 0.0111 | 0.7124 ± 0.0014 / 0.6965 ± 0.0049 | +0.0013 | **+0.0050** | **PASS** |

Both splits pass on worst-hospital (the headline metric); mean deltas are noise-level.

## Aggregation weights actually used (notebook 06 fingerprints)

- organism-community (5 clients): **[0.15, 0.02, 0.07, 0.74, 0.02]** — H3 (narrowest
  repertoire, lowest homophily) gets ~74% where FedAvg's size weights give ~5%.
- specimen (3 clients): **[0.36, 0.40, 0.24]** — the two small hospitals get ~64%
  combined instead of ~22% by size.

## Secondary findings

1. **Stability.** Specimen seed-42: FedAvg decayed 0.717 → 0.680 across rounds while
   topology held 0.714 → 0.714; `fedavg_final` 0.6910 → 0.7028. Topology arm's
   best-round std is tighter (specimen 0.0014 vs 0.0056).
2. **Determinism confirmed.** Pooled/local-only columns now match across arms to ~1e-4
   (organism local 0.6996 vs 0.6996; specimen pooled 0.6904 vs 0.6904) — the seed+cid
   fix (`0af90d0`) verified end-to-end; residual ~1e-4 is GPU-kernel nondeterminism.
3. **Per-seed heterogeneity (honest caveat).** organism seed-43: topo worst +0.019;
   organism seed-42: −0.012. Worst-delta std (~0.016–0.021) **dwarfs** the +0.005
   effect → the PASS is fragile at n=3. One more seed could flip it.
4. **Machine-checked aspects:** fingerprint decode/fallback, `temperature=inf ≡ FedAvg`,
   weighting math — all covered by `tests/test_topology_strategy.py` + `test_topology.py`
   (17/17 + strategy/seeding tests; not run locally — Colab/CI only).

## Reconciliation needed

`res.txt` (2026-09-28, committed log) records a **specimen FedAvg baseline** with
pooled 0.6903 ± 0.005, local-only 0.6962 ± 0.0026, FedAvg best **0.7147 ± 0.002**,
worst 0.6957 ± 0.0013 — not identical to the FedAvg arm in the comparison table
(0.7111 ± 0.0056 / 0.6915 ± 0.0111). Both claim 3 seeds (42/43/44). Determine which
protocol/round of the run each reflects (pre- vs post-seeding-fix, or different
`compute_pooled`/arm settings) **before citing either number**.
