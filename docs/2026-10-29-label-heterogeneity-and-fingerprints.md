# Why similarity weighting cannot win on our splits, and where it could (offline analysis, 2026-10-29)

No training. Everything here is computed on a Mac CPU from the ARMD frame and the saved result files of notebooks 16, 17 and 19.
Code: `src/amr_fed/split_diagnostics.py`, `src/amr_fed/label_fingerprints.py`, `src/amr_fed/federated/budget.py`,
`scripts/2026-10-29-fingerprint-offline-test.py`. Tests: `tests/test_label_fingerprints.py`.

## 1. The fine-tune lead is the size of 8 extra epochs

`budget.fed_upto` / `budget.local_upto` re-select the round / epoch on validation when only the first R rounds / E epochs are allowed
(they reproduce every saved `phc_f1` and `val_local` exactly). Size-weighted test F1, mean over 5 seeds:

| | Louvain | org10 |
|---|---|---|
| local-only, epoch <= 24 -> <= 32 | 0.6981 -> 0.7085 (+0.0104) | 0.7133 -> 0.7220 (+0.0087) |
| FedAvg, round <= 6 -> <= 8 (24 -> 32 epochs) | 0.7031 -> 0.7098 (+0.0068) | 0.7180 -> 0.7257 (+0.0078) |
| FedAvg@8 -> + fine-tune 8 epochs (32 -> 40) | 0.7098 -> 0.7156 (+0.0057) | 0.7257 -> 0.7308 (+0.0051) |

The last 8 epochs of plain training were worth more than fine-tuning's 8 epochs. The budget-matched rerun is notebook 21 Part A.

## 2. How different are the hospitals in what the model must learn?

`split_diagnostics.heterogeneity_vs_null`: nested logistic models on per-(organism, antibiotic, hospital) counts. *prior* = deviance
explained by a hospital-wide base-rate shift (what a per-hospital cutoff fixes); *concept* = what is left, pair-level differences.
Nats per 1000 tests. The null is the same hospital sizes with patients shuffled (3 shuffles). "+ history" refines the pair key by the
patient's prior-resistance quartile (what the GNN's history features carry). Ratio = observed / null concept (1 = no real difference).

| Split | Hosp. | prior | concept / null | ratio | + history: concept / null | ratio |
|---|---|---|---|---|---|---|
| random (alpha = 1e6 Dirichlet; check) | 10 | 0.11 | 6.58 / 6.70 | 0.98 | 12.06 / 12.00 | 1.00 |
| ward-Dirichlet alpha 0.5 (nb18, SimFT ward10) | 10 | 1.64 | 7.52 / 6.29 | 1.19 | 12.23 / 11.16 | 1.10 |
| organism_community (org10) | 10 | 0.87 | 6.44 / 6.19 | 1.04 | 11.50 / 10.66 | 1.08 |
| louvain_split (3) | 3 | 1.02 | 1.94 / 1.51 | 1.28 | 4.14 / 2.70 | 1.53 |
| adi_baseline (4) | 5 | 1.09 | 4.70 / 3.22 | 1.46 | 6.82 / 5.81 | 1.17 |
| specimen_baseline | 3 | 1.97 | 4.10 / 1.51 | 2.71 | 4.76 / 2.76 | 1.73 |
| ward_baseline (home ward) | 5 | 5.09 | 7.54 / 2.81 | 2.68 | 8.42 / 5.17 | 1.63 |
| ordering mode (in/outpatient) | 3 | 1.88 | 4.28 / 1.42 | 3.01 | 4.49 / 2.60 | 1.73 |
| antibiogram_conflict (5) | 5 | 123.8 | 73.2 / 2.96 | 24.7 | 12.58 / 5.28 | 2.39 |
| time era, 4 equal bins (first culture) | 4 | 5.57 | 12.12 / 2.43 | 5.00 | 13.43 / 4.44 | 3.02 |
| time era, 8 bins | 8 | 6.04 | 16.32 / 5.28 | 3.09 | 19.60 / 9.54 | 2.05 |
| **era_cluster_split, 2 eras x 5 hospitals** | 10 | 0.01 | — | 1.49* | between groups excess 5.77; **inside a group 0.99x** | |
| era_cluster_split, 5 eras x 2 | 10 | 0.03 | — | 1.79* | inside a group 1.01x | |

\* with history strata. `era_mix` dials it down: 2 eras with era_mix 0.25 -> 1.13x, 0.5 -> 1.01x.

Reading:
- **The splits our method results rest on (org10, ward-Dirichlet, Louvain) are the least label-heterogeneous of all**: 1.08-1.10x random
  for the ten-hospital splits. They were designed to differ in graph topology, and they do, but hospitals still share the label mechanism.
- antibiogram_conflict looks extreme on raw pairs (24.7x), but most of it disappears once patient history is conditioned on (2.4x): it
  assigned patients by their own resistance tendency, which the history features explain. That is why notebook 08 found FedAvg helped.
- **Time is the strongest natural concept shift, and it has no base-rate shift left after history (prior 0.01).** A per-hospital cutoff
  cannot absorb it. The drivers are clinically recognisable: E. coli ampicillin/sulbactam 0.88 -> 0.35 resistant across eras, S. aureus
  levofloxacin 0.57 -> 0.11, P. aeruginosa amikacin 0.50 -> 0.02 (early selective testing / panel changes), and E. coli levofloxacin
  0.21-0.23 until 2019, 0.32-0.33 from 2020 (consistent with labs adopting the 2019 CLSI fluoroquinolone breakpoint revision).
- Specimen, home ward and ordering mode carry real but smaller shifts (1.6-1.7x after history). The model does not see specimen or ward.

## 3. Do label fingerprints predict transfer on the existing splits?

Ground truth: SimFT's cross-evaluation (notebook 19), G[i, j] = loss of hospital j's model on hospital i's validation half A minus i's
own model's loss. Fingerprints from TRAIN rows. Spearman over off-diagonal pairs, mean over 5 seeds (sd), round-1 G (models trained 4
epochs from a common start, the cleanest "data similarity" signal). Seeds with permutation p < 0.05 in brackets.

| Fingerprint | org10 | ward10 @10% | Louvain (3 hosp., uninformative) |
|---|---|---|---|
| topology v3 (current) | +0.10 (0.05) [0/5] | +0.08 (0.16) [1/5] | −0.07 |
| case mix JS (label-free) | +0.24 (0.04) [5/5] | +0.02 (0.17) [0/5] | +0.14 |
| per-antibiotic rates | +0.07 (0.07) [1/5] | +0.02 (0.13) [0/5] | +0.14 |
| antibiogram (pair rates) | +0.02 (0.11) [0/5] | −0.02 (0.18) [0/5] | 0.00 |
| antibiogram, centred | +0.03 (0.12) [1/5] | −0.01 (0.17) [0/5] | 0.00 |
| **cross-antibiogram loss** | **+0.34 (0.09) [3/5]** | **+0.21 (0.19) [2/5]** | +0.17 |

Noise ceiling (org10, one fixed partition): one seed's G predicts another seed's G at Spearman 0.74. So the best label fingerprint
captures about half of what is predictable, the topology fingerprint about a seventh. **But** on these splits the transfer gaps themselves
are tiny (a neighbour's model is ~1% worse than one's own), and SimFT, which used G directly, did not beat its flat ablation. A better
fingerprint cannot help where there is nothing to exploit.

## 4. On the era split, every fingerprint recovers the groups

AUC of "different-era pairs look farther apart than same-era pairs" (seed 42, train rows):

| Fingerprint | 2 eras x 5 | 2 eras, era_mix 0.25 | 5 eras x 2 |
|---|---|---|---|
| topology v3 | 1.00 | 0.96 | 0.85 |
| case mix JS | 1.00 | 1.00 | 1.00 |
| antibiogram / centred | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |
| cross-antibiogram loss | 1.00 | 1.00 | 0.96 |

So on the era split the question is no longer "can the fingerprint see the structure" but "does using it beat FedAvg + fine-tune".
That is notebook 21 Part B, with an oracle arm that averages exactly within the true group as the upper bound.

## 5. How single-dataset FL papers make non-IID clients (literature check)

| Family | How | Used by | What it varies |
|---|---|---|---|
| Label / quantity skew | Dirichlet over labels or sizes | NIID-Bench (Li et al., ICDE 2022); MIMIC/eICU FL papers | P(y), sizes |
| Feature skew | sort on an attribute, contiguous chunks; noise per client | NIID-Bench; FS-G "instance-space splitter" (scaffold sort on molhiv) | P(x) |
| Graph community | Louvain / Metis communities -> clients | FS-G, OpenFGL, FedGTA, FED-PUB | topology, P(x) |
| Meta / natural attribute | an interpretable attribute defines clients | FS-G meta splitter (FedDBLP); MIMIC-IV by care unit; eICU by hospital id; FLamby (natural partitions) | whatever the attribute drives |
| **Concept shift (clustered FL)** | groups of clients with a transformed input or relabelled targets (rotations 0/90/180/270, label swaps) | IFCA (Ghosh et al.), CFL surveys | P(y given x): the case similarity methods are built for |
| **Temporal** | time windows as clients / drift over rounds | FedDrift (Jothimurugesan et al., AISTATS 2023); Wild-Time (MIMIC by year) | P(y given x) over time |

Our earlier splits are in the label/quantity, community and natural-attribute families. Similarity-weighted and clustered methods are
evaluated in the literature almost always under an explicit concept shift (rotations, label swaps or drift). The era split is the
natural-data version of that family; a controlled label perturbation (a breakpoint-style relabelling of chosen pairs in one group) is
the synthetic version, kept as a fallback if the era split's shift is too weak.

Sources: NIID-Bench https://arxiv.org/abs/2102.02079 · FederatedScope-GNN https://arxiv.org/abs/2204.05562 · OpenFGL
https://arxiv.org/abs/2408.16288 · FedGTA https://arxiv.org/abs/2401.11755 · FED-PUB https://proceedings.mlr.press/v202/baek23a.html ·
FLamby https://arxiv.org/abs/2210.04620 · FedDrift https://arxiv.org/abs/2206.00799 · Wild-Time https://arxiv.org/abs/2211.14238 ·
clustered-FL survey https://arxiv.org/abs/2501.17512 · CLSI 2019 fluoroquinolone revision https://pubmed.ncbi.nlm.nih.gov/30971460/

## 6. Caveats

- The heterogeneity measure is pair-level (plus a history stratum); the GNN sees more (patient features, graph context), so it is an
  upper-bound-style screen, not a proof that a model must differ.
- Era "hospitals" are time windows of one lab. The early era has selective-testing artefacts (some pairs tested mainly when resistance
  was suspected) as well as genuine trends; both are real reasons why two labs' labels differ, but they are not two institutions.
- Fingerprint scores rest on 5 seeds and 10 hospitals (90 ordered pairs); Louvain (3 hospitals, 6 pairs) cannot discriminate.
