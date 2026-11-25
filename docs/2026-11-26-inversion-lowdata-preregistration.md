# Pre-registration — label-inversion shift (`shift25`) at 3% data, with the shuffled control

**Recorded 2026-11-26, BEFORE any training run of this test.** Changes after seeing results are *iterations*, appended in section 6 with their date.

## 1. Why
Three tests on the pair-rate split (flip labels, resample rows, rate inputs removed) all failed the headroom gate: even the oracle (equal weights inside the true region) does not beat FedAvg + fine-tune, and an offline
ceiling shows that knowing each hospital's own pair rates adds only about +0.004 AUROC over patient history (`2026-11-25-pair-rate-lowdata-preregistration.md`). Rate differences give weighting nothing to fix. The one shift with
measured headroom is label inversion (notebook 22, full data, 3 seeds): `shift25` oracle +0.0074 [+0.0040, +0.0109], v3_ab +0.0057 [+0.0019, +0.0094] over FedAvg + fine-tune, but no shuffled control and only 3 seeds.
This test repeats it at the low-data level (3% of each hospital's train/validation patients, same test patients) with 5 seeds and the shuffled control.

## 2. The shift (`federated/shift_runs.py`, unchanged)
10 hospitals = 2 groups of 5 by patient era (`era_cluster_split`). For LATER-era patients the label of a fixed random 25% of antibiotics is inverted, so the two groups follow conflicting rules for those drugs.
No input reveals which rule applies, unlike a rate difference.

## 3. Runs (`scripts/2026-11-26-inversion-lowdata.py`, Mac CPU)
3% data, seeds 42-46, 8 rounds x 4 local epochs, hidden 128, T = 0.5, every arm gets the same 8-epoch fine-tune. Arms: `fedavg`, `oracle`; then `v3_ab` (cross-antibiogram fingerprint) and `v3_ab_shuffled` only if the gate passes.
**Metric:** unweighted mean over hospitals of test macro-F1 after fine-tuning (own validation cutoff). Paired by seed, 95% t-interval, n = 5.

## 4. Decision rules (fixed in advance)
- **Headroom gate (first):** `oracle+ft - fedavg+ft` interval entirely above 0. If it fails, stop and report a null; the fingerprint arms are not run.
- **The fingerprint works** only if the gate passed and BOTH `v3_ab+ft - fedavg+ft` and `v3_ab+ft - v3_ab_shuffled+ft` have intervals entirely above 0.
- Also report the weights placed on own group / other group, and (if run) `oracle+ft - v3_ab+ft`.

## 5. Limits
- Label inversion is not a clinical process. A win shows the mechanism works when hospitals follow conflicting rules, not that ARMD needs it.
- One shift level (25%), one split, five seeds; the intervals cover training randomness (and the era partition, which changes with the seed) only.
- Inversion also moves hospital base rates; per-hospital cutoffs absorb part of that.
- The topology fingerprint is not tested here (the 2026-11-02 pre-registration tested it only on panel splits).

## 6. Iterations (append below, dated)

### 2026-11-26 — result (3%, `shift25`, seeds 42-46, unweighted mean F1 after the same 8-epoch fine-tune; `results/amr_inversion_f030.json`)
**Gate:** FedAvg + fine-tune 0.7536, oracle + fine-tune 0.7642; `oracle+ft - fedavg+ft` = +0.0106, 95% CI [+0.0049, +0.0162], p = 0.01, 5/5 seeds above zero. **Gate passed.**
**Fingerprint arms (run after the gate, as registered):** v3_ab + fine-tune 0.7627, v3_ab_shuffled + fine-tune 0.7531.
`v3_ab+ft - fedavg+ft` = +0.0090 [+0.0026, +0.0155], p = 0.02, 5/5 seeds; `v3_ab+ft - v3_ab_shuffled+ft` = +0.0096 [+0.0037, +0.0154], p = 0.01, 5/5 seeds; `oracle+ft - v3_ab+ft` = +0.0015 [-0.0001, +0.0032], p = 0.06, 4/5 seeds.
**Both pre-registered conditions hold: the fingerprint works at 3% on this shift.** Weights (mean over seeds): v3_ab self 0.200 | own group 0.771 | other group 0.029; shuffled 0.200 | 0.322 | 0.478; oracle 0.200 | 0.800 | 0.000.
**Reading:** the label fingerprint finds the true groups (77% of the weight on the own group against 80% for the oracle) and recovers about 85% of the oracle's gain; the shuffled control, which spreads weight over both groups, scores the same as FedAvg. Limits as in section 5: inversion is synthetic, one shift level, five seeds.
