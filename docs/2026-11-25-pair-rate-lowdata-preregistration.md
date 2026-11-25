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
