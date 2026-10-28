# Notebook 18 results — low-data curve and new-hospital test

**Notebook:** `notebooks/18_low_data_and_new_site.ipynb` (Colab, Tesla T4). **Raw:** `results/amr_regimes_all.json` (local; not in git).
**Pre-registration:** `2026-10-28-low-data-and-new-site-preregistration.md` (written before the run; no iterations).
Setting: 10 hospitals from `dirichlet_ward_mixture(alpha=0.5)`, seeds 42–46 = five different partitions, rounds 8, local epochs 4,
local-only epochs 32, hidden 128. Every hospital picks its own cutoff (and the round / epoch) on its own validation rows; test is read once.

## 1. Part A — low-data curve

Each hospital keeps a fraction of its train and validation patients; its test patients never change. Primary metric: the unweighted
mean over the 10 hospitals of test macro-F1 (mean ± std over 5 seeds).

| Data kept | Train rows / hospital | Local-only | FedAvg | FedAvg + fine-tune |
|---|---|---|---|---|
| 100% | 111,839 | 0.6996 ± 0.0018 | 0.6980 ± 0.0040 | 0.7074 ± 0.0044 |
| 25% | 28,138 | 0.6895 ± 0.0065 | 0.6947 ± 0.0045 | 0.7019 ± 0.0050 |
| 10% | 11,169 | 0.6762 ± 0.0059 | 0.6894 ± 0.0064 | 0.6934 ± 0.0065 |
| 3% | 3,343 | 0.6569 ± 0.0062 | 0.6732 ± 0.0058 | 0.6781 ± 0.0072 |

Paired by seed (mean difference, 95% interval, p, seeds above zero):

| Data kept | FedAvg − local-only | FedAvg + fine-tune − local-only | FedAvg + fine-tune − FedAvg |
|---|---|---|---|
| 100% | −0.0016 [−0.0055, +0.0023] p=0.32, 1/5 | +0.0078 [+0.0034, +0.0122] p=0.01, 5/5 | +0.0094 [+0.0073, +0.0116] p<0.01, 5/5 |
| 25% | +0.0052 [−0.0014, +0.0119] p=0.09, 4/5 | +0.0124 [+0.0057, +0.0191] p=0.01, 5/5 | +0.0072 [+0.0064, +0.0080] p<0.01, 5/5 |
| 10% | +0.0132 [+0.0016, +0.0247] p=0.03, 5/5 | +0.0172 [+0.0047, +0.0297] p=0.02, 5/5 | +0.0040 [+0.0009, +0.0072] p=0.02, 4/5 |
| 3% | +0.0162 [+0.0113, +0.0211] p<0.01, 5/5 | +0.0212 [+0.0171, +0.0253] p<0.01, 5/5 | +0.0050 [−0.0007, +0.0107] p=0.07, 4/5 |

Size-weighted version (FedAvg − local / fine-tune − local): 100%: −0.0018 / +0.0070; 25%: +0.0018 / +0.0085; 10%: +0.0084 / +0.0132;
3%: +0.0125 / +0.0167.

**Reading.**
- Federation's advantage over local-only **grows steadily as local data shrink**: FedAvg − local-only goes −0.0016 → +0.0052 → +0.0132 →
  +0.0162, with the interval clearly above zero at 10% and 3%. At full data FedAvg ties local-only (consistent with notebooks 14–17).
- The pre-registered "hump" (very small sites breaking FedAvg) did **not** appear down to 3%, where hospitals still have about
  3,300 training rows each. Smaller levels were not tested.
- FedAvg + fine-tune is above local-only at **every** level, including full data (see caveat 1).

## 2. Part B — a new hospital

A hospital is held out; FedAvg trains on the other nine; the held-out hospital gets k labelled training patients. Test = its full test set
(mean ± std over 5 seeds; each seed is a different partition).

**Held-out = 5th-largest hospital (6,400–7,400 patients)**

| Labelled patients k | Zero-shot | Fed + fine-tune | From scratch |
|---|---|---|---|
| 25 | 0.6385 ± 0.0192 | 0.6461 ± 0.0087 | 0.6097 ± 0.0172 |
| 100 | 0.6718 ± 0.0073 | 0.6808 ± 0.0096 | 0.6622 ± 0.0153 |
| 400 | 0.6913 ± 0.0063 | 0.7022 ± 0.0088 | 0.6861 ± 0.0138 |
| all (~4,860) | 0.6960 ± 0.0093 | 0.7043 ± 0.0085 | 0.6994 ± 0.0052 |

**Held-out = smallest hospital (930–1,850 patients)**

| Labelled patients k | Zero-shot | Fed + fine-tune | From scratch |
|---|---|---|---|
| 25 | 0.6518 ± 0.0294 | 0.6651 ± 0.0224 | 0.6293 ± 0.0312 |
| 100 | 0.6762 ± 0.0083 | 0.6806 ± 0.0119 | 0.6528 ± 0.0265 |
| 400 | 0.6953 ± 0.0084 | 0.7025 ± 0.0113 | 0.6885 ± 0.0098 |
| all (~1,010) | 0.6944 ± 0.0100 | 0.7039 ± 0.0116 | 0.6962 ± 0.0065 |

Paired by seed — fine-tuned − scratch (zero-shot − scratch in brackets):

| k | Median hospital | Smallest hospital |
|---|---|---|
| 25 | +0.0364 [+0.0193, +0.0534] p<0.01, 5/5 (zero-shot +0.0288, p=0.01) | +0.0358 [+0.0188, +0.0528] p<0.01, 5/5 (zero-shot +0.0225, p=0.01) |
| 100 | +0.0186 [+0.0028, +0.0344] p=0.03, 5/5 (zero-shot +0.0095, p=0.35) | +0.0278 [−0.0078, +0.0633] p=0.10, 4/5 (zero-shot +0.0234, p=0.15) |
| 400 | +0.0161 [+0.0043, +0.0279] p=0.02, 5/5 (zero-shot +0.0052, p=0.26) | +0.0140 [+0.0034, +0.0246] p=0.02, 5/5 (zero-shot +0.0068, p=0.10) |
| all | +0.0049 [−0.0008, +0.0107] p=0.08, 4/5 (zero-shot −0.0034, p=0.23) | +0.0077 [−0.0004, +0.0159] p=0.06, 5/5 (zero-shot −0.0017, p=0.42) |

**Reading.**
- With **few labelled patients (k ≤ 400)** the federated model, fine-tuned, beats a model trained from scratch by +0.014 to +0.036 F1, with
  intervals above zero at k = 25 and 400 for both hospitals (k = 100 is above zero for the median hospital only).
- The advantage **shrinks as the new hospital gets more data** and is no longer significant at k = all (p = 0.06–0.08).
- At k = 25 the federated model **with no local training at all** (zero-shot) already beats a model trained on those 25 patients
  (+0.029 / +0.023, p = 0.01) and at k = all it is statistically indistinguishable from a model trained on all of the hospital's own data.

## 3. Caveats (state these in the paper)

1. **Everything is still improving at the training budget.** Local-only's best epoch is the last one (epoch 32) for all 50 hospital-seed
   pairs at 100% data and 40/50 at 25%, falling to 17/50 at 3%; FedAvg's selected round is 8 (the last) in every run; fine-tuning picks its
   last epoch (8) in 50/50 at 100%. So at higher data levels the local baseline is budget-limited, and **fine-tuning gets 8 more local epochs
   than local-only or FedAvg get (40 vs 32)**. The FedAvg − local-only comparison is budget-matched; the fine-tune comparisons are not, and part
   of the full-data fine-tune gain (+0.0078) may be extra training rather than federation. At 3% local-only often stops early (best epoch < 32 in 33/50),
   so the low-data FedAvg − local-only result is not budget-limited. A budget-matched control (local-only and FedAvg for 40 epochs) was not run.
2. Hospitals are subsets of one dataset sharing organisms and drugs; the held-out hospital is a best case for transfer, not another health system.
3. Five seeds / five partitions; the unweighted mean gives small hospitals equal say.
4. Validation shrinks with training at low data levels, making cutoff / epoch selection noisier for every arm.
5. The lowest level tested (3%) still leaves about 3,300 training rows per hospital; behaviour below that is unknown.
6. The 10 federated runs per rule use nine hospitals, so the "new hospital" model saw slightly less data than the Part A FedAvg models.
