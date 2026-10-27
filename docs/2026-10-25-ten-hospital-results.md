# Ten-hospital split (organism_community, n=10) — six-method results

**Notebooks:** `notebooks/12_ten_hospitals.ipynb` (Colab: FedAvg, FedProx, FedMedian, FedTrimmedAvg,
uniform control) and `notebooks/12_ten_hospitals_mac.ipynb` (local M2 CPU: topo, resumed from the
Colab JSON). Raw results: `results/amr_org10_full.json` (local; not in git).

## 1. Setup

- **Split:** `organism_community(df, n_clients=10)`. Each patient goes to the hospital holding their
  most common organism; organisms are greedily packed into 10 buckets. Deterministic: the seed is
  ignored, so all five seeds share the same partition. **Seeds (42–46) therefore vary only training
  randomness, not the split.**
- **Protocol:** rounds=8, local_epochs=4, local_only_epochs=32, hidden=128, macro-F1, five seeds.
- **Arms:** local-only, pooled, FedAvg (size-weighted), FedProx, FedMedian, FedTrimmedAvg,
  uniform-weight control (topology code path, T=∞), topo (distinctiveness, T=1.0).
- **Hospital sizes (patients):** 37,952 / 4,886 / 3,543 / 3,322 / 2,884 × 6. One hospital holds
  about 57% of patients.

## 2. Summary (mean ± std over 5 seeds)

| Method | Best-round F1 | Final-round F1 | Worst-hospital F1 | AUROC (best) | AUROC (final) |
|---|---|---|---|---|---|
| Local only | 0.6882 ± 0.0150 | — | 0.6527 ± 0.0240 | 0.8435 ± 0.0021 | — |
| Pooled | 0.6985 ± 0.0109 | — | 0.6653 ± 0.0184 | 0.8424 ± 0.0035 | — |
| FedAvg | 0.7151 ± 0.0028 | 0.6744 ± 0.0287 | 0.6945 ± 0.0041 | 0.8389 ± 0.0053 | 0.8477 ± 0.0021 |
| FedProx | 0.7161 ± 0.0041 | 0.6931 ± 0.0176 | 0.6949 ± 0.0042 | 0.8386 ± 0.0052 | 0.8469 ± 0.0025 |
| FedMedian | 0.7184 ± 0.0042 | 0.7146 ± 0.0084 | 0.6961 ± 0.0063 | 0.8432 ± 0.0038 | 0.8451 ± 0.0024 |
| FedTrimmedAvg | 0.7189 ± 0.0039 | 0.7140 ± 0.0100 | 0.6966 ± 0.0055 | 0.8440 ± 0.0032 | 0.8455 ± 0.0024 |
| Uniform control | 0.7188 ± 0.0037 | 0.7122 ± 0.0125 | 0.6962 ± 0.0053 | 0.8444 ± 0.0029 | 0.8457 ± 0.0024 |
| **Topo (T=1)** | 0.7148 ± 0.0047 | 0.6985 ± 0.0236 | 0.6893 ± 0.0049 | 0.8403 ± 0.0061 | 0.8449 ± 0.0030 |

Definitions: best-round = highest F1 over rounds (optimistic, since it selects on test data);
final-round = after round 8; AUROC (best) = AUROC at the best-F1 round, AUROC (final) = after round 8; worst-hospital = lowest per-hospital F1, per seed, then averaged.

## 3. Per-hospital F1 (mean over seeds, best round; hospital index = client id)

| Hospital | Patients | Local | Pooled | FedAvg | FedProx | FedMed | FedTrim | Uniform | Topo |
|---|---|---|---|---|---|---|---|---|---|
| H0 | 37,952 | 0.6527 | 0.6824 | 0.6945 | 0.6949 | 0.6963 | 0.6975 | 0.6971 | 0.6922 |
| H1 | 2,885 | 0.7550 | 0.7449 | 0.7673 | 0.7691 | 0.7738 | 0.7736 | 0.7738 | 0.7813 |
| H2 | 2,884 | 0.7343 | 0.7243 | 0.7376 | 0.7357 | 0.7408 | 0.7409 | 0.7404 | 0.7276 |
| H3 | 2,884 | 0.7261 | 0.6757 | 0.7358 | 0.7371 | 0.7431 | 0.7432 | 0.7436 | 0.7307 |
| H4 | 3,322 | 0.7373 | 0.7305 | 0.7565 | 0.7532 | 0.7589 | 0.7601 | 0.7607 | 0.7618 |
| H5 | 2,884 | 0.7244 | 0.6913 | 0.7140 | 0.7187 | 0.7160 | 0.7150 | 0.7155 | 0.7086 |
| H6 | 4,886 | 0.7702 | 0.7724 | 0.7724 | 0.7733 | 0.7771 | 0.7765 | 0.7764 | 0.7774 |
| H7 | 3,543 | 0.7310 | 0.6892 | 0.7374 | 0.7408 | 0.7468 | 0.7453 | 0.7467 | 0.7413 |
| H8 | 2,884 | 0.7636 | 0.7401 | 0.7560 | 0.7600 | 0.7598 | 0.7587 | 0.7585 | 0.7666 |
| H9 | 2,884 | 0.7742 | 0.7256 | 0.7399 | 0.7464 | 0.7518 | 0.7512 | 0.7514 | 0.7475 |

Hospitals where each method beats local-only: 7/10 for FedAvg, FedProx, FedMedian, FedTrimmedAvg,
uniform, and topo. Topo loses to local on H2, H5 and H9.

## 4. Topology weights (recomputed from fingerprints, T=1, distinctiveness)

| Hospital (by size) | Patients | Size share | Topo weight |
|---|---|---|---|
| Largest | 37,952 | 0.566 | 0.194 |
| One 3,322-patient hospital | 3,322 | 0.050 | **0.732** |
| Other eight hospitals | 2,884–4,886 | 0.043–0.073 each | 0.004–0.025 each |

Uniform would be 0.100 each. The max/min weight ratio is about 171. **On this split, topo does not
behave like the uniform control**: it concentrates the averaged model on one small, unusual hospital.
These weights are fixed for the whole run because the fingerprints come from the static partition.

## 5. What the numbers show

- **Federation helps the largest hospital.** Local-only is worst on H0 (0.653). Every federated method
  adds about +0.04 there, with the same direction in every seed.
- **Federation hurts several mid-sized hospitals.** H9 (local 0.774) drops 0.02–0.03 under every
  federated method. H5 and H8 also lose under most methods.
- **Topo's worst-hospital score is the lowest of the federated methods (0.6893).** On the largest
  hospital it scores 0.6922, below the uniform control (0.6971) and every other federated method.
  Topo is also below the uniform control on 6 of 10 hospitals (H0, H2, H3, H5, H7, H9).
- **Topo's best-round score (0.7148) is about 0.004 below the uniform control (0.7188).** The
  difference is about one standard deviation, so it is within noise.
- **Fixed-round view (no round selection).** At round 8, FedAvg is 0.014 below local-only (0.6744 vs
  0.6882); FedProx is +0.005, FedMedian/FedTrimmedAvg/uniform are +0.024 to +0.026, and topo is +0.010.
  AUROC (final) is 0.8449–0.8477 for every federated method, slightly above local-only (0.8435), so
  federation adds about +0.001 to +0.004 AUROC; the F1 gaps between methods are mostly threshold effects.
- **The robust methods are the strongest group.** FedMedian, FedTrimmedAvg and the uniform control
  are close together on best-round, final-round, and worst-hospital F1.
- **Final-round stability favors the robust methods.** FedAvg loses 0.04 from best to final round
  (0.7151 → 0.6744). FedMedian, FedTrimmedAvg and the uniform control each lose less than 0.01.

## 6. Caveats

- **The split is fixed across seeds**, so the spread shows training variability only. Results
  describe this partition, not ten-hospital splits in general.
- **Topo ran on a different machine.** Topo was run on a Mac (CPU) and the other arms on Colab. The
  two environments can differ slightly, so the topo-versus-control gap should be rerun on the same
  machine before it is treated as a result.
- **Best-round is optimistic.** It picks the peak round on the test data, so the final-round columns
  are the fairer comparison.
- **Pooled is worse than local-only on 7 of 10 hospitals** (H1, H2, H3, H5, H7, H8, H9). Its
  worst-hospital score (0.6653) is higher than local's (0.6527) only because the largest hospital
  gains from pooling.

## 7. Next steps

1. Rerun the uniform control and topo on the same machine, so the comparison is clean.
2. Check whether the topo weights' strong concentration on one hospital is a fingerprint artifact
   (which features drive the distinctiveness score), before reading topo as a method result.
3. The Louvain 3-hospital split is documented separately in
   `docs/2026-10-23-louvain-topology-results.md`; a full Louvain comparison (notebook 13) is in progress.
