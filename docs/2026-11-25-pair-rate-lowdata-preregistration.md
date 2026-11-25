# Pre-registration — notebook 29: pair-rate split at low data levels

**Recorded 2026-11-25, BEFORE any training run on this split.** Changes after seeing results are *iterations*, appended in section 6 with their date.

## 1. Why
Every natural split we ran is close to homogeneous in label behaviour (1.08-1.10x random), and per-hospital cutoffs absorb a hospital-wide rate
difference (`2026-10-29-label-heterogeneity-and-fingerprints.md`). This split makes hospitals differ the way real hospitals do: in the resistant
**rate on specific organism-antibiotic pairs**, by amounts taken from published hospital / regional comparisons. It is run at low data levels
because that is where federation helped most in earlier runs.

## 2. The split (`scripts/2026-11-24-pair-rate-split.py`)
- 10 equal hospitals (patients by a hash of the patient id), in 3 regions (hospital c is in region c % 3).
- Every organism-antibiotic pair with at least 200 tests (414 pairs, 96.5% of tests) gets a per-hospital target resistant rate: the pair's overall rate moved on
  the logit scale by a regional profile plus a smaller hospital term (within-region spread = 0.35 x between-region). Drugs of one class share their organism's
  shift. Each (organism, class) block's most-vs-least-resistant odds ratio is drawn from 2.0-3.6.
- Labels are changed by the minimum number of random flips (3.5% of tests). Features are not changed.
- Real references: Gaza, two hospitals (El Aila & Abu El Aish, BMC Microbiol 2025;25:559, Table 4): OR 2.6 (ceftriaxone), 3.5 (ciprofloxacin), 3.7 (amikacin);
  US VA census divisions (Livorsi et al. 2016): OR 1.9; Washington State regions (Fenelon et al. 2024): adjusted odds ratios 0.20-0.78 and 0.64-0.81;
  15 US emergency departments (2022): OR 3.6 (search snippet only, not verified).
- **Regions were added before any training**, after an offline check (no training) showed that with independent shifts every pair of hospitals is about
  equally far apart (true-rate distance 0.055-0.079, CV 0.08): a similarity method has nothing to find. With regions the true distance is 0.044 within a
  region against 0.080 between regions (ratio 0.55). The label fingerprint sees this (ratio 0.39); the topology fingerprint does not (1.10), as expected: it
  describes graph shape, not rates.

## 3. Runs (`scripts/2026-11-25-pair-rate-lowdata.py`, notebook 29, Mac CPU)
Levels 10%, 3%, 1%, 0.3% of each hospital's train/validation patients (same test patients). Seeds 42-46 (training randomness only; the split is fixed).
8 rounds x 4 local epochs, hidden 128, T = 0.5, every arm + the same 8-epoch fine-tune; local-only 40 epochs.
Arms: `fedavg`, `v3_topo`, `v3_topo_shuffled`, `v3_ab`, `v3_ab_shuffled`, `oracle` (equal weights inside the true region).
**Metric:** unweighted mean over hospitals of test macro-F1 after fine-tuning (own validation cutoff). Paired by seed, 95% t-interval, n = 5.

## 4. Decision rules (fixed in advance)
- **Headroom gate (run first, 3% level, fedavg + oracle only):** `oracle+ft - fedavg+ft` interval entirely above 0. If it fails, stop: knowing the true regions does
  not help, so no fingerprint can; report a null and run nothing else.
- **A fingerprint works at a level** only if the gate passed and BOTH `v3_X+ft - fedavg+ft` and `v3_X+ft - v3_X_shuffled+ft` have intervals entirely above 0
  (X = topo or ab). Report each level separately; no pooling over levels.
- Also report `fedavg+ft - local-only` and the weights placed on same-region hospitals.

## 5. Limits
- The shift is synthetic (random label flips), so a win shows the mechanism works, not that ARMD needs it. Features carry no sign of it, so the model cannot see it
  except through the labels.
- Regional structure is a design choice (3 regions, within-region 0.35); the real numbers only fix the size of the differences, not how hospitals cluster.
- The topology fingerprint is not expected to see rate differences; a null there is informative, not a surprise.
- Five seeds, one fixed split: the intervals cover training randomness only.

## 6. Iterations (append below, dated)

### 2026-11-25 — gate result and iteration 1 (diagnostic, not a rescue)
**Gate result (3%, 5 seeds):** `oracle+ft - fedavg+ft` = -0.0027, 95% CI [-0.0058, +0.0005], p = 0.08, 0/5 seeds above zero. **Gate failed**; per section 4
nothing else was run. Scores: local-only 0.6655, FedAvg 0.6705, FedAvg + fine-tune 0.6734, oracle + fine-tune 0.6707. FedAvg + fine-tune beat local-only by +0.0079.
**Iteration 1 (added after seeing the gate, written while the run started, to explain the gate and not to change the verdict):** `random_a/b/c` = the oracle with
random groups of the same sizes (4/3/3) as the true regions, each sharing at most 1/3 of its same-group pairs with them, at 3%. If these score like the true-region
oracle, its gap to FedAvg is just "fewer hospitals in the average", not the regions carrying (or lacking) information. The verdict above stands whatever this shows.
**Iteration 1 result (3%, 5 seeds, unweighted mean F1 after fine-tune):** local-only 0.6655, FedAvg + fine-tune 0.6734, oracle with the true regions 0.6707, oracle with random
groups 0.6727 / 0.6739 / 0.6741 (mean 0.6735). Random groups - FedAvg + fine-tune = +0.0001 [-0.0049, +0.0051], p = 0.94. True regions - random groups = -0.0028 [-0.0106, +0.0050], p = 0.37.
**Reading:** averaging with only 3-4 hospitals costs nothing against averaging with all 10, so "fewer hospitals in the average" does not explain the oracle's gap. The true regions score no better
than random groups, so on this split the regions carry no useful information. Intervals are wide (about +-0.005 to 0.01 with 5 seeds), so an effect under about 0.005 is not excluded.
**Offline check, same day (no training):** a predictor that uses each hospital's own pair resistance rate scores AUROC 0.8398 on the shifted data (0.8385 on the original), against 0.8344 for
the all-hospital pair rate. Each hospital's graph already holds its own training-label pair rates (`graph_build.py`, near line 371), so a rate shift is visible in each model's inputs and a
shared model has no conflict to resolve. The gate verdict stands: null on this split.
**Shift-cost diagnostic, same day (not pre-registered; `shift_cost()`):** FedAvg and local-only trained on the ORIGINAL labels (same hospitals, 3%, seeds 42-46), compared with the shifted run (paired by seed;
test labels are shifted too, so this is a rough bound). Unweighted mean F1: local-only 0.6703 -> 0.6655 (-0.0048, p = 0.23); FedAvg 0.6758 -> 0.6705 (-0.0053, p = 0.03); FedAvg + fine-tune 0.6828 -> 0.6734
(-0.0094, p < 0.01, 0/5 seeds above zero). FedAvg + fine-tune minus local-only: +0.0125 on original labels, +0.0079 on shifted. **Reading:** the whole shift costs a federated model about 0.005-0.009 F1, and the part
federation loses relative to local-only is about 0.005. That is the most any grouping method could win back, and it is the size of the noise at 5 seeds (+-0.005 to 0.01). The oracle recovered none of it
(-0.0027). This split is too mild to test similarity weighting; a test needs a shift that costs plain FedAvg clearly more.

### 2026-11-26 — iteration 2: resampling instead of label flips (written before the run)
Suggested by the team: reach each hospital's target rate without changing any label. `make_df(df, mode="resample")` keeps every label real and DROPS rows: to raise a rate only a share of the
susceptible rows is kept, to lower it only a share of the resistant rows. Offline check (no training): rates hit the targets (odds ratio median 2.94, middle half 2.49-3.59, same as the flip version),
the region structure is unchanged (true-rate distance ratio 0.53; label fingerprint 0.52; topology fingerprint 0.86), and **24.0% of rows are dropped** (1,601,980 -> 1,216,886; 116k-127k rows per hospital,
95 patients lost) against 3.5% of labels changed in the flip version. Overall resistant rate 0.195 -> 0.198.
**Run:** the same gate as section 4, at 3%, 5 seeds, fedavg + oracle only, on the resampled data (`resample_gate()`, `results/amr_pairrate_rs_f030.json`), plus FedAvg's cost against the original labels (same
diagnostic as above, reusing `amr_pairrate_orig_f030.json`). Same rule: the oracle must beat FedAvg + fine-tune with a 95% interval above zero, otherwise stop. This is a second attempt at the same gate and is
reported as such, not as a replacement for the first.
**Expectation (stated before the run):** resampling changes only the rate, which each hospital's model already sees in its inputs, so I expect the same null. A different result would mean the label noise in the flip version was masking a real effect.
**Iteration 2 result (resampled data, 3%, 5 seeds, unweighted mean F1 after fine-tune):** local-only 0.6535, FedAvg 0.6655, FedAvg + fine-tune 0.6724, oracle + fine-tune 0.6716.
`oracle+ft - fedavg+ft` = -0.0008, 95% CI [-0.0053, +0.0037], p = 0.65, 2/5 seeds above zero. **Gate failed again**, same rule as before: null on this split. As expected, the expectation stated above held.
**Cost against the original labels:** local-only 0.6703 -> 0.6535 (-0.0168, p = 0.01); FedAvg + fine-tune 0.6828 -> 0.6724 (-0.0104, p = 0.01). Most of this is the 24% of rows dropped, not the shift, which is why local-only
loses more than FedAvg: FedAvg + fine-tune minus local-only is +0.0189 here against +0.0125 on the original labels (federation helps more when each hospital has less data). So this version cannot be used to bound the
shift's own cost; the flip version's bound (about 0.005-0.009) stands.

### 2026-11-26 — rate-input ablation (diagnostic, not pre-registered; `scripts/2026-11-26-rate-input-ablation.py`)
**Correction first:** the graph's organism and antibiotic node features carry each hospital's own train-label resistance rate per ORGANISM and per ANTIBIOTIC (`graph_build.py`, `org_rate`, `abx_rate`), not per organism-antibiotic pair. The earlier
statement that each hospital's model sees its own pair rates was too strong, and the offline pair-rate predictor check (AUROC 0.8398) does not describe the model's inputs. The null verdicts above rest on the measured scores and are unchanged;
the explanation for them is open (the hospital's own training and the fine-tune could also learn pair rates from its labels).
**Test:** the same shifted data, 3%, seeds 42-46, FedAvg arm only, with the last column of the organism and antibiotic node features (the rate) set to 0. Unweighted mean F1:
local-only 0.6655 -> 0.6200 (-0.0455, CI [-0.0532, -0.0377]); FedAvg 0.6705 -> 0.6306 (-0.0399); FedAvg + fine-tune 0.6734 -> 0.6339 (-0.0395, CI [-0.0490, -0.0300]); 0/5 seeds above zero for every arm.
FedAvg + fine-tune minus local-only: +0.0079 with the rates, +0.0139 without (5/5 seeds).
**Reading:** the rate inputs are worth about 0.04 F1 at 3% data, far more than the whole pair-rate shift costs (0.005-0.009). A hospital's own rates are information the model uses heavily, and without them federation helps more because a small local
model cannot learn the base rates alone. This does not by itself say whether pair-level rate shifts are visible to the model; it only shows the rate inputs matter.

