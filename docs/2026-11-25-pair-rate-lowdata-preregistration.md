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
