# Louvain split — six-method results (3 hospitals, rank-matched)

**Notebook:** `notebooks/13_louvain_full_mac.ipynb` (local M2 CPU, five seeds, all arms). Raw:
`results/amr_louvain_full.json` (local; not in git). These numbers replace the earlier
index-matched Louvain results from notebooks 09–11, which mixed hospitals across seeds.

## 1. The split

`louvain_split(n_clients=5, seed)` returns **3 hospitals for every seed**, whatever `n_clients` is
set to. The partition changes with the seed:

| Seed | Hospital sizes (patients, largest first) |
|---|---|
| 42, 46 | 49,348 / 14,204 / 3,456 |
| 43, 45 | 49,113 / 14,190 / 3,705 |
| 44 | 46,960 / 14,191 / 5,857 |

Hospitals are ranked by size **within each seed**, so "largest / middle / smallest" always means the
same role. Seed 44's smallest hospital is a different group of patients from the smallest hospitals
in the other seeds, so rank matching is an approximation, not an exact match.

## 2. Protocol

- rounds = 8, local_epochs = 4, local_only_epochs = 32, hidden = 128, macro-F1, five seeds (42–46).
- **Arms:** local-only, pooled, FedAvg (size-weighted), FedProx, FedMedian, FedTrimmedAvg,
  uniform-weight control (T = ∞), topo (distinctiveness, T = 1.0).

## 3. Per-hospital F1 (mean over seeds, best round, ranked by size)

| Hospital | Patients | Local | Pooled | FedAvg | FedProx | FedMed | FedTrim | Uniform | Topo |
|---|---|---|---|---|---|---|---|---|---|
| Largest | ~48,776 | 0.6785 | 0.6886 | 0.6923 | 0.6914 | 0.7010 | 0.6997 | 0.6997 | **0.7011** |
| Middle | ~14,196 | **0.6818** | 0.6406 | 0.6719 | 0.6753 | 0.6783 | 0.6791 | 0.6792 | **0.6829** |
| Smallest | ~4,036 | **0.7357** | 0.7111 | 0.7328 | 0.7316 | 0.7416 | 0.7401 | 0.7400 | **0.7435** |

## 4. Summary (mean ± std over seeds)

| Method | Best-round F1 | Final-round F1 | Worst-hospital F1 | AUROC (best) |
|---|---|---|---|---|
| Local only | 0.6838 ± 0.0167 | — | 0.6680 ± 0.0133 | 0.8308 ± 0.0038 |
| Pooled | 0.6851 ± 0.0091 | — | 0.6406 ± 0.0096 | 0.8297 ± 0.0014 |
| FedAvg | 0.6935 ± 0.0032 | 0.6671 ± 0.0151 | 0.6719 ± 0.0119 | 0.8195 ± 0.0040 |
| FedProx | 0.6930 ± 0.0041 | 0.6681 ± 0.0175 | 0.6753 ± 0.0136 | 0.8176 ± 0.0083 |
| FedMedian | 0.7020 ± 0.0037 | 0.6875 ± 0.0270 | 0.6783 ± 0.0147 | 0.8316 ± 0.0038 |
| FedTrimmedAvg | 0.7010 ± 0.0069 | 0.6874 ± 0.0233 | 0.6791 ± 0.0130 | 0.8329 ± 0.0045 |
| Uniform control | 0.7009 ± 0.0068 | 0.6877 ± 0.0229 | 0.6792 ± 0.0130 | 0.8329 ± 0.0045 |
| **Topo (T=1)** | **0.7027 ± 0.0038** | **0.6990 ± 0.0093** | **0.6829 ± 0.0097** | 0.8336 ± 0.0031 |

Definitions: best-round = highest F1 over rounds (optimistic, since it picks the peak on test data);
final-round = after round 8; worst-hospital = lowest per-hospital F1 in each seed, averaged over seeds.

## 5. Topology weights (recomputed from fingerprints, T = 1)

| Seed | Largest | Middle | Smallest |
|---|---|---|---|
| 42 | 0.316 | 0.286 | 0.399 |
| 43 | 0.300 | 0.370 | 0.329 |
| 44 | 0.217 | 0.496 | 0.287 |
| 45 | 0.300 | 0.370 | 0.329 |
| 46 | 0.316 | 0.286 | 0.399 |

Uniform would be 0.333 each. Topo's weights are close to uniform on this split, which is why topo
and the uniform control perform almost the same.

## 6. What the numbers show

- **Topo vs FedAvg (best round):** +0.009 on average. The uniform control gets +0.007 of that gain,
  so most of topo's advantage over FedAvg comes from equal weighting rather than the topology signal.
- **Topo vs uniform (best round):** +0.002, within noise. On the final round, topo is +0.011
  (0.6990 against 0.6877), and the std is about 0.01–0.02, so that is weak evidence.
- **Topo vs FedAvg (final round):** +0.032. FedAvg's score falls during training; topo's doesn't.
- **Topo beats local-only on all three hospital ranks** (averaged over seeds). The largest and
  smallest gains are clear. The middle gain is +0.001, which is noise.
- **Federation vs local-only, by rank:** FedAvg beats local on the largest hospital only. Middle
  and smallest get worse (−0.010 and −0.003).
- **Robust methods are as good as topo.** FedMedian and FedTrimmedAvg match topo on best-round and
  worst-hospital F1 within noise.
- **Pooled is not a ceiling here.** Its average (0.6851) is barely above local (0.6838), and it is
  well below local on the middle hospital (0.6406). The pooled model takes only 32 full-batch
  optimizer steps (see Caveats), so it is probably undertrained.

## 7. Caveats

- **Pooled is undertrained.** `run_pooled` takes one full-batch step per epoch for 32 epochs, so it
  gets 32 optimizer steps. The 0.71 figure in `run.py` is a hardcoded constant from an earlier
  Phase-1 run; `run.py` doesn't record that run's training budget, so I haven't verified why it is
  higher. Pooled numbers in this doc should not be read as a ceiling until its training budget is
  fixed.
- **Rank matching is approximate.** Seed 44 has a different partition, so its "smallest" hospital
  is not the same patients as the smallest in the other seeds.
- **Best-round is optimistic.** The final-round columns are the fairer comparison.
- **Five seeds over three partitions.** Differences under about 0.01 are not reliable here.
- **FedAvg differs from the old doc.** The old Louvain doc reported FedAvg best-round 0.6785. This
  run gives 0.6935. The local-only numbers agree with the old doc, but I haven't found the cause of the
  FedAvg difference. It may be the metric definition or a change in the code.

## 8. Next steps

1. Fix the pooled training budget (more optimizer steps, or mini-batches) before using pooled as a
   ceiling.
2. Match hospitals by patient membership, not just size rank, to remove the seed-44 approximation.
3. Rerun the topo and uniform arms on the same machine to remove the cross-machine caveat.
4. Check whether topo's advantage over the robust methods holds with more seeds.
