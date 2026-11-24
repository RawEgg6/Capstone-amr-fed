# Validation-selected results — Louvain (3 hospitals) and organism split (10 hospitals)

**Notebooks:** `notebooks/14_louvain_val_selection.ipynb`, `notebooks/15_org10_val_controls.ipynb`.
**Raw results:** `results/amr_louvain_val.json`, `results/amr_org10_val_<TAG>.json` (local; not in git).
**Supersedes the headline numbers** in `2026-10-23-louvain-topology-results.md`,
`2026-10-25-ten-hospital-results.md` and `2026-10-25-team-report-federated-amr.md`. Those documents
used the legacy protocol (see section 1) and are kept for comparison only.

## 1. Protocol

Legacy protocol (all earlier docs): the federated score was the best round chosen **on test macro-F1**,
at a fixed 0.5 cutoff. Local-only was scored once, with no round or cutoff selection.

Validation-selected protocol (this document):

- Each hospital scores the global model on its **validation and test** rows every round, at 29 cutoffs
  (0.150–0.850 in steps of 0.025). Only per-hospital F1 values at those cutoffs reach the server.
- The server picks the (round, cutoff) with the best size-weighted **validation** macro-F1. Test is read
  once, at that choice.
- Local-only gets the same rule: a checkpoint every 4 epochs, and the (checkpoint, cutoff) with the best
  validation F1, **chosen per hospital**.
- F1 is unchanged: per-hospital macro-F1, size-weighted over hospitals. Rounds = 8, local epochs = 4,
  local-only epochs = 32, hidden = 128, seeds 42–46.
- The pooled baseline is untouched (0.71 is the Phase-1 constant; not revisited here).

## 2. Louvain split (3 hospitals, 5 seeds over 3 distinct partitions)

| Method | Val-selected F1 | @0.5 cutoff, same round | Worst hospital | AUROC |
|---|---|---|---|---|
| Local only | 0.7100 ± 0.0033 | — | 0.7043 ± 0.0027 | 0.8317 ± 0.0039 |
| FedAvg | 0.7081 ± 0.0034 | 0.6454 ± 0.0144 | 0.6896 ± 0.0060 | 0.8361 ± 0.0025 |
| FedProx | 0.7077 ± 0.0033 | 0.6489 ± 0.0151 | 0.6889 ± 0.0062 | 0.8358 ± 0.0025 |
| FedMedian | 0.7093 ± 0.0040 | 0.6549 ± 0.0104 | 0.6925 ± 0.0035 | 0.8365 ± 0.0025 |
| FedTrimmedAvg | 0.7079 ± 0.0035 | 0.6770 ± 0.0184 | 0.6899 ± 0.0077 | 0.8355 ± 0.0023 |
| Uniform control | 0.7081 ± 0.0035 | 0.6775 ± 0.0187 | 0.6918 ± 0.0075 | 0.8356 ± 0.0023 |
| Topo (T=1) | 0.7081 ± 0.0035 | 0.6889 ± 0.0160 | 0.6929 ± 0.0063 | 0.8356 ± 0.0024 |

The validation-selected round is 8 for every method and seed. "Final, val cutoff" equals the
val-selected column for the same reason.

**Federated minus local-only, paired by seed:** FedAvg −0.0018, FedProx −0.0023, FedMedian −0.0007,
FedTrimmedAvg −0.0021, Uniform −0.0018, Topo −0.0019. Almost every seed is below zero.

**Per-hospital test F1 (mean over seeds, hospitals ranked by size):**

| Hospital | Patients | Local | FedAvg | FedProx | FedMed | FedTrim | Uniform | Topo |
|---|---|---|---|---|---|---|---|---|
| Largest | 48,776 | 0.7043 | 0.7072 | 0.7068 | 0.7077 | 0.7062 | 0.7063 | 0.7060 |
| Middle | 14,196 | 0.7200 | 0.6896 | 0.6889 | 0.6925 | 0.6899 | 0.6918 | 0.6929 |
| Smallest | 4,036 | 0.7484 | 0.7411 | 0.7405 | 0.7455 | 0.7463 | 0.7459 | 0.7467 |

**Paired topo comparisons (val-selected F1):**

| Comparison | Mean difference | 95% CI | p |
|---|---|---|---|
| Topo − Uniform | −0.0001 | [−0.0014, +0.0012] | 0.90 |
| Topo − FedMedian | −0.0012 | [−0.0025, +0.0001] | 0.06 |
| Topo − FedAvg | −0.0001 | [−0.0015, +0.0014] | 0.91 |
| Worst hospital: Topo − Uniform | +0.0012 | [−0.0063, +0.0087] | 0.69 |
| Worst hospital: Topo − FedAvg | +0.0034 | [+0.0011, +0.0057] | 0.02 |

Uniform gets most of the worst-hospital gain over FedAvg (0.6918 vs 0.6896), so it comes from
equal weighting, not the topology signal.

## 3. Ten-hospital organism split (5 seeds, one fixed partition)

| Method | Val-selected F1 | @0.5 cutoff, same round | Worst hospital | AUROC | Val round per seed |
|---|---|---|---|---|---|
| Local only | 0.7233 ± 0.0019 | — | 0.6981 ± 0.0031 | 0.8430 ± 0.0024 | — |
| FedAvg | 0.7237 ± 0.0038 | 0.6736 ± 0.0299 | 0.7025 ± 0.0027 | 0.8477 ± 0.0021 | 8, 8, 8, 8, 8 |
| FedProx | 0.7227 ± 0.0038 | 0.6935 ± 0.0178 | 0.7017 ± 0.0024 | 0.8456 ± 0.0039 | 8, 8, 3, 8, 8 |
| FedMedian | 0.7202 ± 0.0032 | 0.7146 ± 0.0083 | 0.6984 ± 0.0030 | 0.8451 ± 0.0024 | 8, 8, 8, 8, 8 |
| FedTrimmedAvg | 0.7206 ± 0.0030 | 0.7138 ± 0.0103 | 0.6994 ± 0.0028 | 0.8455 ± 0.0024 | 8, 8, 8, 8, 8 |
| Uniform control | 0.7210 ± 0.0030 | 0.7123 ± 0.0125 | 0.6992 ± 0.0034 | 0.8457 ± 0.0024 | 8, 8, 8, 8, 8 |
| Topo (T=1) | 0.7224 ± 0.0034 | 0.7185 ± 0.0056 | 0.6991 ± 0.0037 | 0.8457 ± 0.0036 | 8, 8, 8, 8, 8 |
| Topo-shuffled (control) | 0.7177 ± 0.0048 | 0.6891 ± 0.0485 | 0.6945 ± 0.0036 | 0.8407 ± 0.0027 | 8, 8, 8, 8, 8 |
| Inverse-size (control) | 0.7214 ± 0.0029 | 0.7150 ± 0.0076 | 0.7000 ± 0.0034 | 0.8463 ± 0.0025 | 8, 8, 8, 8, 8 |

**Federated minus local-only, paired by seed:** FedAvg +0.0004, FedProx −0.0005, FedMedian −0.0031,
FedTrimmedAvg −0.0027, Uniform −0.0023, Topo −0.0009, Topo-shuffled −0.0055, Inverse-size −0.0019.

**Paired comparisons (val-selected F1):**

| Comparison | Mean difference | 95% CI | p | Seeds positive |
|---|---|---|---|---|
| Topo − Topo-shuffled | +0.0046 | [−0.0012, +0.0105] | 0.09 | 4 of 5 |
| Topo − Inverse-size | +0.0010 | [−0.0005, +0.0025] | 0.14 | 4 of 5 |
| Topo − Uniform | +0.0014 | [−0.0001, +0.0029] | 0.06 | 4 of 5 |
| Topo − FedMedian | +0.0022 | [+0.0003, +0.0040] | 0.03 | 4 of 5 |
| Uniform − FedAvg | −0.0027 | [−0.0047, −0.0007] | 0.02 | 0 of 5 |

Worst hospital: Topo − Shuffled +0.0046 (p 0.11), Topo − Inverse-size −0.0008 (p 0.17),
Topo − Uniform −0.0001 (p 0.83), Uniform − FedAvg −0.0032 (p 0.03).

**Per-hospital test F1 (mean over seeds, validation-selected; hospital index = client id):**

| Hospital | Patients | Local | FedAvg | FedProx | FedMed | FedTrim | Uniform | Topo | Shuffled | Inv-size |
|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 37,952 | 0.6981 | 0.7025 | 0.7017 | 0.6984 | 0.6994 | 0.6992 | 0.6991 | 0.6945 | 0.7000 |
| H1 | 2,885 | 0.7713 | 0.7749 | 0.7715 | 0.7702 | 0.7692 | 0.7725 | 0.7762 | 0.7701 | 0.7713 |
| H2 | 2,884 | 0.7482 | 0.7385 | 0.7390 | 0.7374 | 0.7381 | 0.7378 | 0.7352 | 0.7405 | 0.7374 |
| H3 | 2,884 | 0.7373 | 0.7468 | 0.7455 | 0.7466 | 0.7459 | 0.7475 | 0.7448 | 0.7468 | 0.7478 |
| H4 | 3,322 | 0.7674 | 0.7622 | 0.7616 | 0.7608 | 0.7614 | 0.7605 | 0.7643 | 0.7598 | 0.7621 |
| H5 | 2,884 | 0.7610 | 0.7238 | 0.7234 | 0.7211 | 0.7202 | 0.7214 | 0.7255 | 0.7239 | 0.7205 |
| H6 | 4,886 | 0.7812 | 0.7808 | 0.7812 | 0.7781 | 0.7774 | 0.7777 | 0.7843 | 0.7776 | 0.7782 |
| H7 | 3,543 | 0.7568 | 0.7504 | 0.7479 | 0.7432 | 0.7417 | 0.7463 | 0.7611 | 0.7422 | 0.7451 |
| H8 | 2,884 | 0.7846 | 0.7682 | 0.7665 | 0.7672 | 0.7664 | 0.7673 | 0.7642 | 0.7666 | 0.7658 |
| H9 | 2,884 | 0.7806 | 0.7586 | 0.7561 | 0.7577 | 0.7561 | 0.7575 | 0.7673 | 0.7583 | 0.7593 |

Hospitals where the method beats local-only (of 10): FedAvg 3, FedProx 4, FedMedian 2, FedTrimmedAvg 2,
Uniform 3, Topo 5, Shuffled 1, Inverse-size 2. Federation helps H0 and H3 and loses on H2, H5, H8, H9
(up to about −0.04 on H5 and −0.016 to −0.02 on H8 and H9). The legacy
protocol showed about +0.04 for every federated method on H0; against a tuned local-only it is +0.004.
Topo's per-hospital losses to local are smallest on H7 and H9, which is where its edge over the other
federated arms comes from.

**Weights actually used (seed-independent; fingerprints come from the static partition):**

| Hospital | Patients | Size share | Uniform | Real topo | Inverse-size |
|---|---|---|---|---|---|
| 0 | 37,952 | 0.566 | 0.100 | 0.183 | 0.009 |
| 1 | 4,886 | 0.073 | 0.100 | 0.007 | 0.071 |
| 2 | 3,543 | 0.053 | 0.100 | 0.006 | 0.098 |
| 3 | 3,322 | 0.050 | 0.100 | **0.751** | 0.104 |
| 4–9 (six hospitals) | 2,884–2,885 | 0.043 | 0.100 | 0.004–0.021 | 0.120 |

In the shuffled arm, hospital c gets the weight of hospital perm[c], so a different hospital holds
the 0.75 weight each seed (hospitals 4, 9, 9, 5, 7 for seeds 42–46).

## 4. What the numbers show

- **Federation does not beat local-only on F1 under the same selection rule.** Louvain: −0.0007 to
  −0.0023. Ten hospitals: FedAvg +0.0004 (a tie), everything else −0.0005 to −0.0055.
- **Federation raises AUROC slightly.** About +0.004 on Louvain and +0.002 to +0.005 on ten hospitals.
  The exception is the shuffled arm, which is 0.002 *below* local-only. F1 does not follow AUROC.
  Local-only tunes a cutoff per hospital; the federated arms share one cutoff. This is a confound that
  favors local, and per-hospital cutoffs for federated models are untested.
- **The legacy protocol inflated differences.** On Louvain the legacy spread across methods was 0.020
  (0.6785–0.6992); validation-selected it is 0.0016 (0.7077–0.7093). Local-only rose from 0.6825 to
  0.7100 (+0.028) on Louvain and, in the legacy-vs-new comparison on ten hospitals, from 0.6882 to 0.7233.
- **The fixed 0.5 cutoff is costly for federated models.** At the selected round, 0.5 scores 0.645–0.689
  on Louvain and 0.674–0.719 on ten hospitals, against 0.708–0.724 with a validation-tuned cutoff. The
  earlier "topo is stable in the final round" result was a calibration effect at 0.5 (Louvain topo 0.6889
  vs FedAvg 0.6454); with cutoffs tuned they tie at 0.7081.
- **"Robust methods beat FedAvg" does not survive.** On ten hospitals FedAvg is the best federated
  method; Uniform is 0.0027 below it (p = 0.02). The legacy advantage came from FedAvg's unstable
  final round at a fixed 0.5 cutoff.
- **Topology weighting adds nothing on Louvain** (topo − uniform −0.0001, CI ±0.0013). **On ten hospitals
  the evidence is suggestive, not conclusive:** topo is +0.0046 over shuffled (p = 0.09), +0.0010 over
  inverse-size (p = 0.14) and +0.0014 over uniform (p = 0.06), positive in 4 of 5 seeds each. None is
  significant, and these are at least eight comparisons. Seed 44 is the exception in each case.
  Shuffled puts 0.75 weight on an arbitrary small hospital and loses 0.005, so concentrating weight on
  the wrong hospital hurts; whether the fingerprint chooses a *good* hospital or just a harmless one is
  not answered by this design.
- **Topo does not beat plain FedAvg or local-only** on either split (point estimates −0.0001 and
  −0.0009 respectively on ten hospitals).

## 5. Caveats

- **Few independent splits.** Louvain gives 3 distinct partitions across 5 seeds (seeds 42/46 and 43/45
  share a partition). The ten-hospital split is one fixed partition, so seeds vary training only.
- **Eight rounds may be too few.** The validation-selected round is 8 for 69 of 70 method-seed runs
  (30 on Louvain, 40 on ten hospitals; the exception is FedProx seed 44 on ten hospitals, round 3). Training has not plateaued.
- **Cutoffs are asymmetric.** Local-only chooses a cutoff per hospital, federated arms one global cutoff.
- **Many comparisons, no multiplicity correction.** Treat p-values of 0.02–0.09 as hints.
- **Fingerprints use all rows, including test,** and may leak label information (flaw 6; unchanged).
- **FedTrimmedAvg equals the uniform mean on Louvain** (β = 0.1 trims 0 of 3 clients); the two rows are
  one result.
- **Pooled is not a ceiling** (undertrained; left unchanged by decision).

## 6. Next steps

1. Per-hospital cutoff tuning for the federated arms (the fair comparison with local-only).
2. 16–24 rounds, since round 8 is nearly always selected.
3. More partitions (additional Louvain resolutions, other organism or region splits), for fingerprint
   evidence that does not rest on one partition.
4. Fingerprints from train rows only (flaw 6), before any claim that the topology signal is real.
5. Framing: a benchmark/protocol paper (evaluation protocol changes conclusions by 0.02–0.04; federation
   ties local on F1, gains slightly on AUROC) is supported. A topology-weighting method paper is not.

## 7. Reproducing and file layout

- Louvain: `results/amr_louvain_val.json` (notebook 14).
- Ten hospitals: `results/amr_org10_val_all.json`, all eight arms and five seeds in one file. The
  paired statistics and per-hospital table above were recomputed from it and match the notebook output.
  Pooled is computed only in the `fedavg` arm.
- `results/amr_louvain_val.json` was checked: the summary rows and the per-hospital table (hospitals
  ranked by size within each seed) match the raw file.
- Raw files are not committed (`results/` is untracked).
