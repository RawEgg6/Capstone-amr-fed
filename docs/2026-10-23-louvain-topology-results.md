# Louvain topology arm — results (rewritten 2026-10-24)

**Notebooks:** `09_louvain_topology` (first run, aggregates only), `10_baseline_comparison`
(5-way, aggregates only), **`11_louvain_per_hospital`** (re-run with every per-hospital number —
the source of truth for this doc). Raw: `/content/drive/MyDrive/amr_louvain_k5_full.json`.

> **Correction notice (2026-10-24).** An earlier version of this file called the setup "k=5",
> described a "minority hospital", reported topology-aware final-round F1 as 0.6907, and claimed
> topology-aware "wins all three metrics". Per-hospital data (nb11) shows those statements were
> wrong or unsupported; they were removed. What remains below is only what the nb11 output backs.

## 1. What the split actually is

`louvain_split(n_clients=5)` returns **3 hospitals**, not 5 — Louvain on the organism–antibiotic
graph finds 3 communities after greedy packing (`n_clients` is only an upper bound; `run_fedavg`
self-heals to the real count). Asking for 10 gives the same 3. The partition depends on the seed,
giving **3 distinct partitions across seeds 42–46**:

| Seeds | Hospital sizes (patients) |
|---|---|
| 42, 46 | 49,348 / 14,204 / 3,456 |
| 43, 45 | 49,113 / 14,190 / 3,705 |
| 44 | 46,960 / 5,857 / 14,191 |

Always one giant hospital (~49k, ≈73%), one ~14k (≈21%; ~6k in seed 44), one ~4k (≈5%).
Hospital ids are **not** ordered by size (seed 44's H1 is the smallest). "Largest / middle /
smallest" below mean by patient count. Protocol: v2 (rounds=8, local_epochs=4,
local_only_epochs=32, hidden=128), 5 seeds (42–46), macro-F1; FedAvg/Topo use the best round
unless stated. Topology arm = distinctiveness, T=1.0.

## 2. Per-hospital results (mean ± std over 5 seeds)

| Hospital | n (typical) | Local-only | Pooled | FedAvg | Topo-aware |
|---|---|---|---|---|---|
| Largest | ~48.8k | 0.6741 ± 0.022 | 0.6766 | 0.6773 ± 0.022 | **0.6986 ± 0.005** |
| Middle | ~14.2k | **0.6932 ± 0.022** | 0.6260 | 0.6529 ± 0.030 | 0.6700 ± 0.010 |
| Smallest | ~4.0k | **0.7434 ± 0.015** | 0.7033 | 0.7230 ± 0.025 | 0.7384 ± 0.017 |

Change vs local-only, same hospital (positive = federation helped):

| Hospital | FedAvg | Topo-aware |
|---|---|---|
| Largest | +0.003 (better in 2/5 seeds) | **+0.024 (better in 5/5)** |
| Middle | −0.040 (1/5) | −0.023 (1/5) |
| Smallest | −0.020 (1/5) | −0.005 (1/5) |

## 3. Aggregate view

| Metric | Local-only | Pooled | FedAvg | Topo-aware |
|---|---|---|---|---|
| Weighted-avg F1 (best round) | 0.6825 ± 0.018 | 0.6732 ± 0.021 | 0.6785 ± 0.023 | **0.6987 ± 0.006** |
| Final-round F1 | n/a | n/a | 0.6457 ± 0.015 | 0.6786 ± 0.021 |
| Worst hospital (each seed's min) | 0.6741 ± 0.022 | 0.6260 ± 0.025 | 0.6529 ± 0.030 | 0.6700 ± 0.010 |
| Weighted AUROC | 0.8317 | 0.8281 | 0.8163 ± 0.010 | 0.8280 ± 0.007 |

Topo-aware beats FedAvg on weighted-avg F1 in 5/5 seeds and on worst-hospital in 3/5.

## 4. What this supports, and what it does not

**Supported**
- Topology-aware aggregation beats FedAvg on this split: +0.020 weighted best-round F1, with
  much lower seed variance (std 0.006 vs 0.023); AUROC is also close to local-only
  (0.8280 vs 0.8317) where FedAvg's is lower (0.8163).
- The clearest per-hospital effect is the **largest hospital**: topo beats its own local model
  in 5/5 seeds (+0.024); FedAvg does not (+0.003).
- On the two smaller hospitals **both** federated methods are worse than training alone; topo is
  closer to local than FedAvg. Neither "rescues" them.
- Pooled is not a ceiling here: it is below local-only on average (0.6732 vs 0.6825) and
  collapses on the middle hospital (0.626) — size-weighted loss under-trains it.

**Not supported — do not claim**
- ~~"Topo beats local-only for the minority hospital"~~ — it does not for any hospital except the largest.
- ~~"Worst-hospital Δ vs local = −0.021 (FedAvg) / −0.003 (topo)"~~ — a worst-vs-worst comparison
  mixes different hospitals (local's worst is the largest hospital in 5/5 seeds; federated
  methods' worst is the middle one in 4/5). Use the same-hospital deltas in §2.
- ~~"Topo wins final-round by +0.044"~~ — corrected: 0.6786 vs 0.6457 = **+0.033**. nb09/nb10's
  0.6907 did not reproduce in nb11 (best-round, worst-hospital and FedAvg numbers did match).
- ~~"Topo up-weights the structurally distinctive hospital"~~ — see §5: the weights are near-uniform.
- No significance claims: n=5 over 3 distinct partitions; local-only is itself unstable
  (seed 44: local on the largest hospital 0.632 vs 0.67–0.69 elsewhere, which inflates both
  federated methods' apparent gain in that seed).

## 5. Open issue: is it topology, or just uniform averaging?

Topology weights logged in nb11 sit in a narrow band (≈0.28–0.40 per hospital; seed 44 up to
0.22–0.50), i.e. close to **1/3 each**. FedAvg uses size weights ≈ 0.73 / 0.21 / 0.05. So the
topology arm largely moves weight *away* from the giant hospital toward the small ones. That
difference alone (uniform vs size weighting) could explain the gain. The needed control is the
`topo_temperature=inf` arm (uniform weights) under the same protocol — not yet run.
Also: `topo_weight_client_{i}` in `strategy.py` is indexed by *result order* in each round (Flower
returns results in arbitrary order), not by hospital id, so the logged per-client weights appear to
rotate between rounds. Aggregation itself is consistent within a round; only the logging labels are
unreliable.

## 6. Baselines (notebook 10 — aggregates only, per-hospital data not saved)

Same protocol, seeds 42–46. FedAvg and Topo rows reproduced exactly in nb11; the other three were
**not re-run** and are taken from the nb10 output.

| Method | Best-round F1 | Worst-hospital F1 | Final-round F1 |
|---|---|---|---|
| Local-only | n/a | 0.6741 ± 0.022 | n/a |
| FedAvg | 0.6785 ± 0.0227 | 0.6529 ± 0.030 | 0.6457 ± 0.015 |
| FedProx (μ=0.01) | 0.6823 ± 0.0186 | 0.6576 ± 0.0269 | 0.6490 ± 0.0155 |
| FedMedian | 0.6983 ± 0.0051 | 0.6690 ± 0.0047 | 0.6542 ± 0.0117 |
| FedTrimmedAvg | 0.6978 ± 0.0057 | 0.6685 ± 0.0080 | 0.6777 ± 0.0182 |
| **Topo-aware** | **0.6987 ± 0.0062** | 0.6700 ± 0.0095 | 0.6786 ± 0.0205 |

Reading: FedProx ≈ FedAvg. FedMedian, FedTrimmedAvg and Topo are statistically
indistinguishable on best-round and worst-hospital. On final-round only FedMedian trails;
Topo (0.6786) ≈ FedTrimmedAvg (0.6777). **Topo does not clearly win against the robust
baselines.** Pooled is not in this table's source run (use §3).

## 7. Next steps

1. Run the **uniform-weight (T=inf)** control on this split (§5).
2. Move to a split with a real many-hospital structure — `organism_community` with 10 hospitals
   (`notebooks/12_ten_hospitals.ipynb`, designed 2026-10-24; FedAvg, FedProx, FedMedian, FedTrimmedAvg, uniform control, Topo); Louvain cannot produce 10.
3. Personalization (local fine-tune after the last round) — federation currently loses to local
   on two of three hospitals.
4. Bootstrap CIs on per-hospital deltas once more partitions exist.
