# Louvain topology arm — results (2026-10-23)

**Notebook:** `notebooks/09_louvain_topology.ipynb`
**Saved:** `/content/drive/MyDrive/amr_louvain_k5_verdict.json`, `amr_louvain_k10_verdict.json`

## Background

Phase 5 (organism-community / specimen splits) showed topology-aware aggregation beats FedAvg
by +0.005 on the worst hospital — real but fragile at n=3. The Louvain split was identified
earlier as the one case where FedAvg *underperforms local-only* (FedAvg 0.7308 < local 0.7423,
`docs/2026-08-11-splits-and-results.md` §7), but the topology arm had never been run there.
This experiment closes that gap.

Design fixed before running: same C1/C2/C3 criteria as Experiment #1, 5 seeds (42–46),
v2 protocol (rounds=8, local_epochs=4, local_only_epochs=32, hidden=128), topology arm =
distinctiveness T=1.0 (the validated Phase-5 mode).

## Results (k=5)

| Arm | Best-round F1 | Worst-hosp F1 | Final-round F1 |
|---|---|---|---|
| Local-only | — | 0.6742 ± 0.0220 | — |
| FedAvg | 0.6785 ± 0.0227 | 0.6528 ± 0.0298 | 0.6465 ± 0.0147 |
| Topology-aware | **0.6987 ± 0.0060** | **0.6712 ± 0.0098** | **0.6907 ± 0.0166** |

Tail hospital (highest client id) per seed:

| Seed | FedAvg | Topo |
|---|---|---|
| 42 | 0.7296 | 0.7380 |
| 43 | 0.6943 | 0.7192 |
| 44 | 0.6809 | 0.6762 |
| 45 | 0.7048 | 0.7475 |
| 46 | 0.7196 | 0.7197 |
| **Mean** | **0.7058** | **0.7201** |

Local-only tail mean: 0.7215.

### Criterion verdict (k=5)

| Criterion | Result | Value |
|---|---|---|
| **C1** — FedAvg hurts minority (`worst_fed ≤ local_worst − 0.01`) | **PASS** | 0.6528 ≤ 0.6742 − 0.01 = 0.6642 ✓ |
| **C2** — Path-B gate (`Δworst ≥ 0.005` or `Δmean ≥ 0.005`) | **PASS** | Δmean = +0.0202, Δworst = +0.0184 (both ≥ 0.005) |
| **C3** — topo raises the tail (tail_topo > tail_fedavg) | **PASS** | 0.7201 vs 0.7058 (+0.0143); topo nearly recovers to local (0.7215) |

## Results (k=10)

Numbers are essentially identical to k=5. **This is expected, not a bug:** `louvain_split`
lets NetworkX determine the number of communities naturally; on ARMD's organism-antibiotic
graph it finds ~5 communities regardless of the `n_clients` request. When Louvain yields fewer
communities than `n_clients`, `run_fedavg` self-heals to `max(hospital)+1`, so k=10 collapsed
back to the same 5-hospital assignment as k=5. The `docs/` note in `partition.py` documents
this. The k=10 arm is therefore a repeat of k=5 under a different label and contributes no new
evidence.

**Implication for future k=10 experiments:** use a split that explicitly creates 10 hospitals
(e.g. `label_dirichlet` / `organism_community` with `n_clients=10`) rather than Louvain, which
is resolution-limited by the graph's natural community structure.

## Summary and interpretation

This is the strongest topology-aware result in the project:

- **Δmean best-round: +0.0202** (vs Phase-5 +0.005 — 4× larger effect)
- **Δworst: +0.0184** (Phase-5 was +0.005)
- **Variance reduction:** topo std 0.006 vs FedAvg std 0.023 on best-round F1 — topology-aware
  is not only better but markedly more stable across seeds.
- **Tail recovery:** topo tail mean 0.7201 vs local-only 0.7215 — the resistant-minority
  hospital is nearly restored to what it would have achieved training alone.

**Why Louvain shows a larger effect than the Phase-5 splits:**
The Louvain communities correspond to structurally coherent subgraphs in the organism-antibiotic
test graph. Each hospital inherits a distinct set of bug-drug pairs; the GNN's message-passing
representations are genuinely different across hospitals in a way that scalar label skew
(antibiogram-conflict, label-Dirichlet) does not reproduce. Size-weighted FedAvg follows the
majority community and distorts the minority hospital's edge embeddings; the topology
fingerprint (which captures graph degree, homophily, and resistance-history structure) correctly
identifies the minority hospital as distinctive and up-weights it.

**Do-not-claim guardrails (from `research_outline.md` §J):**
- n=5 seeds; do not claim statistical significance.
- Mean improvement (+0.020) is large enough to report directionally, but report with std and
  note that seed 44 is an outlier (topo tail 0.6762 < FedAvg 0.6809 on that seed).
- Report final-round alongside best-round (final: topo 0.6907 vs FedAvg 0.6465, Δ +0.044 —
  even larger, but volatile).
- Do not claim "beats centralized" — pooled numbers not collected in this run; add if needed.

## Next steps

1. Run pooled arm on Louvain k=5 to complete the three-way comparison (local / pooled / FedAvg / topo).
2. Bootstrap CIs on the worst-hospital delta (n=5 too small for paired t-test).
3. Add these numbers to `docs/2026-08-11-splits-and-results.md` §3 (canonical results record).
4. Consider a forced k=10 split (e.g. organism_community with n_clients=10) to test whether
   larger client counts amplify the effect as FedGTA predicts.
