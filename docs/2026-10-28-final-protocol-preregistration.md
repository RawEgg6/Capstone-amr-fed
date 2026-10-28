# Pre-registration — final protocol runs (Louvain 3 hospitals, organism 10 hospitals)

**Recorded 2026-10-28, BEFORE any run of notebooks 16 and 17.** These are meant to be the last
training runs on these two splits, so everything the analysis needs is saved (section 5) and every
decision that could be tuned on results is fixed here. A change after seeing results is an
*iteration*: append it below with its date and report both versions.

**How this design came about (disclosure).** It follows the exploratory results in
`2026-10-28-validation-selected-results.md` (notebooks 14 and 15), where federated learning tied
local-only on F1 and topology weighting tied its controls. v3 and the per-hospital cutoffs were designed
*after* seeing those results, on the **same splits, seeds (42-46) and test patients**. So this round is
not an independent confirmation: the win rule below, the controls and the validation-only choice of T are
what protect it, and a tie or a loss is reported as a tie or a loss.

## 1. Questions

1. **Cutoffs.** With each hospital choosing its own cutoff (as local-only always did), how do federated
   models compare with local-only on F1?
2. **Personalization (topo v3).** Does giving every hospital its own similarity-weighted average of all
   hospitals' models beat local-only and a FedAvg + fine-tune baseline, and does the fingerprint
   similarity matter (v3 vs v3-shuffled)?
3. **Re-check of earlier arms** (FedAvg, FedProx, FedMedian, Uniform, Topo v1) under per-hospital cutoffs.

## 2. Splits (no others will be run in this round)

| Notebook | Split | Hospitals | Seeds |
|---|---|---|---|
| 16 | `louvain_split(n_clients=3)` | 3 (about 73% / 21% / 5%; 3 distinct partitions across 5 seeds) | 42–46 |
| 17 | `organism_community(n_clients=10)` | 10 (one fixed partition; seeds vary training only) | 42–46 |

The antibiogram-conflict split (`2026-10-18-antibiogram-conflict-split.md`) is **not** part of this round.

## 3. Arms

Common budget: rounds = 8, local epochs = 4, local-only epochs = 32, hidden = 128. Federated runs per seed:

| Arm | What it is | Runs |
|---|---|---|
| FedAvg | size-weighted; also trains **local-only, pooled and the fine-tune baseline** once per seed | 1 |
| v3, T = 0.25 / 0.5 / 1.0 | personalized aggregation (section 4) | 3 |
| v3-shuffled | v3 at the validation-chosen T, fingerprints handed to the wrong hospitals | 1 |
| Uniform | topology code path, T = inf | 1 |
| Topo v1 | original distinctiveness weights, T = 1, fingerprints from all rows | 1 |
| FedProx / FedMedian | standard baselines | 2 |
| **Federated runs per seed** | **on both splits** | **9** |

Local-only is trained **once per seed** (in the FedAvg arm) and every paired comparison uses that copy.
Before, every arm retrained its own local-only model. Total: 9 x 5 seeds = 45 federated runs per split.

**Deliberately not run (pruned after a literature check, 2026-10-28):**

- *Topo-shuffled and Inverse-size.* Notebook 15 already answered them (Topo v1 vs shuffled +0.0046, p = 0.09;
  vs inverse-size +0.0010, p = 0.14, shared cutoff). Per-hospital cutoffs apply to every arm alike, so
  rerunning them would not change that reading, and v3-shuffled is now the fingerprint-signal test.
- *FedTrimmedAvg.* Equals Uniform on 3 hospitals and was within 0.0005 of Uniform on 10 (notebook 15).
  `final_protocol.stage1_specs(include_trimmed=True)` re-adds it.
- *More rounds, model-similarity v3 (FedAMP / pFedGraph style), temperature sweeps beyond three values,
  extra partitions.* Not part of this round. Reasons: nothing in the results so far points to them as the
  bottleneck, and the literature below says the baseline we already include is the hard one.

**Literature check behind the arm list (searched 2026-10-28).** FedAvg followed by local fine-tuning is
repeatedly reported as a tough baseline that dedicated personalization methods often fail to beat
(Cheng, Chadha, Duchi, arXiv 2108.07313; Matsuda et al., arXiv 2206.13190). That is why FedAvg + fine-tune is
a required comparison. FedAMP and pFedGraph (arXiv 2007.03797; ICML 2023) weight clients by **model**
similarity; v3 uses static fingerprint similarity, which has not been shown to work, so the realistic
expectation is that v3 does not beat FedAvg + fine-tune. Per-client cutoffs: a federated dental-imaging study
reports a pooled cutoff can penalize sites whose score distributions differ, but that per-site cutoffs are
noisy for small sites (arXiv 2603.11850); we use per-hospital cutoffs as the primary metric because
local-only always did, and report the shared-cutoff number beside it.

## 4. Definitions fixed in advance

- **Primary metric (`phc_f1`).** Test macro-F1, size-weighted over hospitals. Every hospital picks its **own
  cutoff** (29 cutoffs, 0.150–0.850) on its own validation rows. The round is the one with the best
  validation macro-F1 (weighted by validation size, each hospital at its best cutoff). Test is read once, at
  that round and those cutoffs. Local-only uses the same rule per hospital (checkpoint every 4 epochs).
  Secondary metrics: worst-hospital F1 (same rule), shared-cutoff F1 (one cutoff for all hospitals,
  `val_fed`), AUROC.
- **FedAvg + fine-tune.** Start every hospital from the **final-round** global FedAvg model, train 8 local
  epochs, evaluate at epochs 0, 2, 4, 6, 8; pick (epoch, cutoff) on that hospital's validation rows.
- **v3 (topo v3).** Hospital i's own model is the weighted average of every hospital's locally trained
  model: `W[i, j] = softmax_j(-D[i, j] / T)`, where D is the pairwise Euclidean distance between
  z-scored fingerprints divided by its mean off-diagonal value. Fingerprints are computed from **training
  rows only** and exclude `resistance_rate`, `log_patients` and `log_tested_edges`. The diagonal is
  always the largest entry; T -> 0 is local training, T = inf is a uniform average. Each hospital trains
  from its own personalized model every round and is evaluated on it.
- **Temperature.** T in {0.25, 0.5, 1.0}; the chosen T maximizes the mean **validation** score (primary
  protocol, mean over seeds). Test is not used. The shuffled control then runs at that T only.
- **v3-shuffled.** Same T; fingerprints permuted by a seeded derangement (a different one per seed), so the
  diagonal stays dominant but neighbour similarities are wrong.

## 5. What is saved (so nothing has to be rerun)

Per run: the confusion counts (TP, FP, FN, TN) at all 29 cutoffs and 20-bin probability histograms (by true
class), validation and test, per hospital per round; test and validation AUROC; sample sizes and class
counts (derivable from the counts); local-only and fine-tune counts at every checkpoint; the aggregation
weights (vector for topo, matrix for v3); fingerprints from all rows and from training rows; shuffle
permutations; the temperature grid with validation scores; the patient-to-hospital assignment (FedAvg arm);
wall-clock, GPU, library versions, git commit. Everything is in `results/amr_<split>_final_<TAG>.json`
(not committed).

## 6. Comparisons and the win rule

**Primary (pre-registered), on `phc_f1`, paired by seed:** (a) v3 − local-only, (b) v3 − FedAvg + fine-tune,
(c) v3 − v3-shuffled.

**v3 "wins" on a split only if all three paired 95% intervals exclude zero on the positive side.** Anything
else is reported as no win. A win on one split and not the other is reported as exactly that.

**Secondary (reported, not part of the win rule):** the same differences on worst-hospital F1; v3 − FedAvg;
FedAvg − local-only; FedAvg + fine-tune − local-only; Topo v1 − Uniform;
Uniform − FedAvg; FedAvg − FedProx; FedAvg − FedMedian. No multiplicity correction is applied to secondary comparisons;
p-values between 0.01 and 0.10 are hints, not findings.

## 7. Limits we state up front

- Louvain has 3 distinct partitions across 5 seeds; the organism split is one partition, so the spread is
  training noise only.
- Hospital sizes are very unequal; the size-weighted score is dominated by the largest hospital, so
  worst-hospital and per-hospital tables are reported next to it.
- Per-hospital cutoffs use small validation sets for the smallest hospitals; local-only has the same noise.
- Eight rounds; the earlier runs picked round 8 in 69 of 70 cases, so more rounds may change absolute
  numbers. Not tested in this round.
- v3 is close to published personalized-FL methods (FedAMP, FedFomo, pFedGraph); related work must be
  cited and, if v3 wins, compared against one of them in a later round.

## 8. Iterations (append below, dated)

_None yet._
