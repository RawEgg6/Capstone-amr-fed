# Pre-registration — notebook 22: hospitals made to differ on purpose (injected concept shift, different test panels)

**Recorded 2026-11-02, BEFORE any run of notebook 22.** Changes after seeing results are *iterations*, appended in section 6 with their date.

## 1. Why
Notebook 21 (results read 2026-11-02): at equal training, FedAvg + fine-tune is not better than local-only or plain FedAvg on Louvain / org10
(Part A), and on the era split even the **oracle** (average exactly within the true era group) does not beat FedAvg + fine-tune
(+0.0008, CI includes 0; Part B). ARMD's hospitals are subsets of one lab, so their label rules barely differ. The question left
is whether similarity-weighted aggregation (v3 with a label fingerprint) can win **when the hospitals do differ**. If it cannot even there,
the method is not worth tuning; if it can, we learn how large the difference must be.

## 2. Experiments (10 hospitals = 2 groups of 5; `federated/shift_runs.py`)
- **Idea 1, concept shift** `shift10/25/50/100`: later-era patients (group 1 of `era_cluster_split`) have the labels of 10/25/50/100 % of
  antibiotics (fixed random set) inverted. The two groups follow conflicting rules for those antibiotics.
- **Idea 2, test panels** `panel50`, `panel20`: patients split at random (hash of patient id) into 2 groups of 5 hospitals; each group only has the
  rows of its panel = 50 % / 20 % of antibiotics shared by both groups + half of the rest exclusive to it. Labels untouched.

## 3. Arms (identical budget: 8 rounds x 4 local epochs, then the same 8-epoch fine-tune for every arm)
`fedavg` (+ fine-tune; no local-only), `v3_ab` (v3 + cross-antibiogram fingerprint, T = 0.5), `oracle` (equal weights within the true group),
`v3_topo` (panel experiments only), `v3_ab_shuffled` (control; run only after v3_ab beats FedAvg + fine-tune). Seeds 42-44.
**Metric:** unweighted mean over hospitals of test macro-F1 (own validation cutoff) after fine-tuning (`budget.finetune_score`).

## 4. Decision rules (fixed in advance; paired by seed, 95 % t-interval, n = 3)
- **Headroom gate, per experiment:** `oracle+ft - fedavg+ft` CI entirely above 0. If it fails at `shift100` (the strongest), report "no headroom
  even under full label conflict", run nothing else, and conclude similarity weighting cannot beat FedAvg + fine-tune on this pipeline.
- **The method works** at an experiment only if the gate passed and BOTH `v3_ab+ft - fedavg+ft` and `v3_ab+ft - v3_ab_shuffled+ft` are above 0.
- **Report the threshold:** the weakest shift at which the gate passes is the smallest heterogeneity at which similarity could matter;
  compare with ARMD's measured heterogeneity (`2026-10-29-label-heterogeneity-and-fingerprints.md`: natural splits 1.08-1.10x random).
- `panel*`: also report `v3_topo+ft - fedavg+ft` (can the structure-based fingerprint see panel differences?). Weights placed on own / other group are reported per arm.

## 5. Limits
- The shift is synthetic (label inversion is not a clinical process); a positive result shows the mechanism works, not that ARMD needs it.
- Three seeds (n = 3): intervals are wide; only large effects can pass the gate, which is the intent (we are looking for a large one).
- Inversion also moves hospital base rates; per-hospital cutoffs absorb part of that, which slightly helps FedAvg.
- No convergence run: the effect is expected to dwarf the ~0.025 training deficit; if the gate passes only barely, treat it as budget-limited.

## 6. Iterations (append below, dated)
