# Notebook 16 results — final protocol on the Louvain split (3 hospitals)

**Run:** local (Mac CPU), `notebooks/16_louvain_final.ipynb` logic via `final_protocol.run_all`; seeds 42–46 (three distinct partitions);
rounds 8, local epochs 4, local-only epochs 32, hidden 128. **Raw:** `results/amr_louvain_final_all.json` (local; not in git).
**Pre-registration:** `2026-10-28-final-protocol-preregistration.md`. Primary metric `phc_f1`: test macro-F1, size-weighted, every
hospital at its own validation-chosen cutoff, round chosen on validation. v3 temperature chosen on validation: **T = 0.5** (mean val F1
0.7103 / 0.7115 / 0.7110 at T = 0.25 / 0.5 / 1.0 — nearly flat).

## 1. Summary (mean ± std over 5 seeds)

| Method | Per-hospital cutoffs (primary) | Shared cutoff | Worst hospital | AUROC |
|---|---|---|---|---|
| Local-only | 0.7085 ± 0.0050 | 0.7085 ± 0.0050 | 0.7021 ± 0.0044 | 0.8308 ± 0.0038 |
| FedAvg | 0.7099 ± 0.0042 | 0.7068 ± 0.0039 | 0.7052 ± 0.0029 | 0.8361 ± 0.0030 |
| **FedAvg + fine-tune** | **0.7156 ± 0.0046** | — | **0.7103 ± 0.0038** | — |
| v3 (T = 0.5) | 0.7119 ± 0.0041 | 0.7086 ± 0.0038 | 0.7066 ± 0.0030 | 0.8372 ± 0.0030 |
| v3-shuffled (T = 0.5) | 0.7118 ± 0.0044 | 0.7084 ± 0.0054 | 0.7066 ± 0.0034 | 0.8370 ± 0.0033 |
| Uniform | 0.7094 ± 0.0030 | 0.7060 ± 0.0028 | 0.7043 ± 0.0018 | 0.8348 ± 0.0027 |
| Topo v1 | 0.7088 ± 0.0036 | 0.7054 ± 0.0033 | 0.7036 ± 0.0024 | 0.8345 ± 0.0028 |
| FedProx | 0.7094 ± 0.0041 | 0.7063 ± 0.0036 | 0.7045 ± 0.0028 | 0.8358 ± 0.0029 |
| FedMedian | 0.7095 ± 0.0038 | 0.7065 ± 0.0031 | 0.7044 ± 0.0022 | 0.8349 ± 0.0037 |

The selected round is 8 for every federated arm and seed except FedMedian seed 43 (round 7).

## 2. Pre-registered verdict: v3 does NOT win

| Primary comparison (paired by seed, `phc_f1`) | Mean difference | 95% interval | p | Seeds > 0 | Rule |
|---|---|---|---|---|---|
| v3 − local-only | +0.0034 | [+0.0000, +0.0068] | 0.05 | 4/5 | interval just clears zero |
| v3 − FedAvg + fine-tune | −0.0037 | [−0.0061, −0.0014] | 0.01 | 0/5 | **fails (fine-tune is better)** |
| v3 − v3-shuffled | +0.0000 | [−0.0009, +0.0010] | 0.96 | 2/5 | **fails (fingerprint carries no signal)** |

All three intervals must exclude zero on the positive side; two do not. Worst-hospital F1 gives the same picture (v3 − local +0.0046, v3 − fine-tune
−0.0036, v3 − shuffled +0.0000).

## 3. Secondary comparisons (paired by seed, `phc_f1`)

| Comparison | Mean difference | 95% interval | p | Seeds > 0 |
|---|---|---|---|---|
| FedAvg + fine-tune − local-only | +0.0071 | [+0.0052, +0.0090] | <0.01 | 5/5 |
| v3 − FedAvg | +0.0020 | [+0.0005, +0.0035] | 0.02 | 5/5 |
| FedAvg − local-only | +0.0014 | [−0.0012, +0.0040] | 0.21 | 4/5 |
| Topo v1 − Uniform | −0.0006 | [−0.0013, +0.0002] | 0.11 | 0/5 |
| Uniform − FedAvg | −0.0005 | [−0.0027, +0.0017] | 0.55 | 3/5 |
| FedAvg − FedProx | +0.0005 | [−0.0009, +0.0019] | 0.39 | 4/5 |
| FedAvg − FedMedian | +0.0004 | [−0.0014, +0.0021] | 0.60 | 4/5 |

## 4. Per-hospital test F1 (mean over seeds, hospitals ranked by size)

| Hospital | Patients | Local-only | FedAvg | FedAvg + FT | v3 | v3-shuffled | Uniform | Topo v1 | FedMedian | FedProx |
|---|---|---|---|---|---|---|---|---|---|---|
| Largest | ~48,776 | 0.7021 | 0.7060 | 0.7103 | 0.7066 | 0.7066 | 0.7043 | 0.7036 | 0.7048 | 0.7057 |
| Middle | ~14,196 | 0.7229 | 0.7100 | 0.7245 | 0.7202 | 0.7201 | 0.7160 | 0.7157 | 0.7144 | 0.7087 |
| Smallest | ~4,036 | 0.7483 | 0.7444 | 0.7521 | 0.7486 | 0.7484 | 0.7465 | 0.7473 | 0.7454 | 0.7433 |

Hospitals beating local-only (of 3): FedAvg + fine-tune 3, v3 2, v3-shuffled 2, FedAvg 1, Uniform 1, Topo v1 1, FedMedian 1, FedProx 1. Plain FedAvg
is 0.013 below local-only on the middle hospital; fine-tuning removes that gap.

## 5. v3 weights

Self-weight (how much a hospital keeps its own model) is 0.73–0.83; the mean off-diagonal weight is 0.107–0.116. v3-shuffled has the same self-weights
and the same mean off-diagonal weight in a different order, which is why shuffling cannot change much on three hospitals.

## 6. What the numbers say

- **The fingerprint similarity carries no usable signal on this split.** v3 and its shuffled control are identical to four decimals. v3's small edge over
  FedAvg (+0.0020) comes from the personalization structure (each hospital keeps about 78% of its own model), not from which hospitals look similar.
- **FedAvg + fine-tune is the best arm** — above local-only by +0.0071 (5/5 seeds, all three hospitals) and above v3 by +0.0037 (0/5 seeds). This matches
  the literature on fine-tuning as a tough baseline.
- **Everything else ties.** FedAvg, FedProx, FedMedian, Uniform and Topo v1 are within 0.001 of each other; topology weighting (Topo v1) is not better than Uniform.
- Federated models again rank patients slightly better (AUROC +0.004 to +0.006 over local-only) while F1 differences stay small.

## 7. Caveats

- **Budget confound for fine-tuning.** Fine-tuning gets 8 extra local epochs (40 vs 32); everything is still improving at the training budget (selected round 8 almost
  always). Part of the fine-tune advantage over v3 and local-only may be extra training. A budget-matched control was not run
  (see `2026-10-28-low-data-and-new-site-results.md`, caveat 1).
- Three hospitals: the shuffled control has little room to differ (see section 5). The organism split (notebook 17) is the stronger test of the fingerprint signal.
- Three distinct partitions across five seeds; the intervals are narrow but cover training noise and these partitions only.
- The v3 − local-only interval reaches zero at its lower end (p = 0.05); treat it as borderline.
