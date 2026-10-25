# Team report: federated learning for antibiotic resistance prediction

**Date:** 2026-10-25
**Scope:** Two ways of splitting the same patient data across hospitals, six ways of combining what
the hospitals learn, five random seeds each. Numbers come from `notebooks/12_ten_hospitals.ipynb`
(organism split), `notebooks/12_ten_hospitals_mac.ipynb` (topo arm of the organism split), and
`notebooks/13_louvain_full_mac.ipynb` (Louvain split).

---

## 1. The big question

Can several hospitals train one shared model without sharing patient records, and do better than
each hospital training alone?

Our test: take one dataset, split it into pretend hospitals, and compare:

- **Local only:** each hospital trains alone. No sharing.
- **Federated:** hospitals train locally, send only model weights to a server, and the server
  combines them into one shared model each round.
- **Pooled:** all data in one place. This is not allowed in real life, so it is a reference point.

---

## 2. The two ways we split the data

Both splits are fake hospitals built from one real dataset (ARMD, about 1.6 million antibiotic-test rows).

| | **Louvain split** | **Organism split** |
|---|---|---|
| How patients are assigned | By clustering organisms that share antibiotic-testing patterns (a graph algorithm) | Each patient goes to the hospital holding their most common organism |
| Number of hospitals | 3 (set by the graph, always 3) | 10 (set by us) |
| Sizes | About 73% / 21% / 5% of patients | One hospital has 57%; the other nine each have 4–7% |
| Changes with the seed? | Yes, slightly (seed 44 differs) | No, the same 10 hospitals every time |

The Louvain split is the more realistic "messy" split. The organism split is the more controlled test,
with more hospitals.

---

## 3. The six methods, in plain words

| Method | What it does |
|---|---|
| **Local only** | No sharing. Each hospital learns from its own data. |
| **Pooled** | Everyone's data in one place. A reference point, not a realistic option. |
| **FedAvg** | Each hospital trains, then the server averages the models, weighting bigger hospitals more. |
| **FedProx** | Like FedAvg, but each hospital is nudged to stay close to the shared model. |
| **FedMedian** | The server takes the median value of each weight, so one unusual hospital can't pull the result. |
| **FedTrimmedAvg** | Like FedMedian, but drops the most extreme values before averaging. |
| **Uniform control** | Like FedAvg, but every hospital gets equal weight. Our baseline for topology. |
| **Topology (topo)** | Our method. The server gives more weight to hospitals whose data looks unusual, measured by a "fingerprint" of each hospital. |

**Why the uniform control matters:** topo and the uniform control use the same code. The only
difference is the weights. If topo beats the uniform control, the fingerprint weighting helps.
If not, the gain comes from equal weighting, not from topology.

---

## 4. What the score means

- **F1 (macro):** a number from 0 to 1 for how well the model predicts resistant vs. susceptible.
  It averages the score for both classes, so a model that ignores one class scores badly. Higher is better.
- **Best round:** the highest score across training rounds. Optimistic, because it picks the peak
  on the test data.
- **Final round:** the score after the last round. The fairer number, because it doesn't cherry-pick.
- **Worst hospital:** the lowest score among the hospitals. Shows whether the method leaves anyone behind.
- **AUROC:** how well the model ranks resistant cases above susceptible ones. 0.5 is chance, 1.0 is perfect.
- **± number:** how much the score varies across the five seeds. Smaller means more consistent.

A rule of thumb: differences smaller than about 0.01 are within the noise of five seeds.

---

## 5. Results: organism split (10 hospitals)

### 5a. Summary

| Method | Best-round F1 | Final-round F1 | Worst-hospital F1 | AUROC |
|---|---|---|---|---|
| Local only | 0.6882 ± 0.0150 | — | 0.6527 ± 0.0240 | 0.8435 ± 0.0021 |
| Pooled | 0.6985 ± 0.0109 | — | 0.6653 ± 0.0184 | 0.8424 ± 0.0035 |
| FedAvg | 0.7151 ± 0.0028 | 0.6744 ± 0.0287 | 0.6945 ± 0.0041 | 0.8389 ± 0.0053 |
| FedProx | 0.7161 ± 0.0041 | 0.6931 ± 0.0176 | 0.6949 ± 0.0042 | 0.8386 ± 0.0052 |
| FedMedian | 0.7184 ± 0.0042 | 0.7146 ± 0.0084 | 0.6961 ± 0.0063 | 0.8432 ± 0.0038 |
| FedTrimmedAvg | 0.7189 ± 0.0039 | 0.7140 ± 0.0100 | 0.6966 ± 0.0055 | 0.8440 ± 0.0032 |
| Uniform control | 0.7188 ± 0.0037 | 0.7122 ± 0.0125 | 0.6962 ± 0.0053 | 0.8444 ± 0.0029 |
| **Topo (T=1)** | 0.7148 ± 0.0047 | 0.6985 ± 0.0236 | 0.6893 ± 0.0049 | 0.8403 ± 0.0061 |

### 5b. Per hospital (mean F1 over seeds, best round)

| Hospital | Patients | Local | FedAvg | FedMedian | FedTrim | Uniform | Topo |
|---|---|---|---|---|---|---|---|
| H0 (largest) | 37,952 | 0.6527 | 0.6945 | 0.6963 | 0.6975 | 0.6971 | 0.6922 |
| H1 | 2,885 | 0.7550 | 0.7673 | 0.7738 | 0.7736 | 0.7738 | 0.7813 |
| H2 | 2,884 | 0.7343 | 0.7376 | 0.7408 | 0.7409 | 0.7404 | 0.7276 |
| H3 | 2,884 | 0.7261 | 0.7358 | 0.7431 | 0.7432 | 0.7436 | 0.7307 |
| H4 | 3,322 | 0.7373 | 0.7565 | 0.7589 | 0.7601 | 0.7607 | 0.7618 |
| H5 | 2,884 | 0.7244 | 0.7140 | 0.7160 | 0.7150 | 0.7155 | 0.7086 |
| H6 | 4,886 | 0.7702 | 0.7724 | 0.7771 | 0.7765 | 0.7764 | 0.7774 |
| H7 | 3,543 | 0.7310 | 0.7374 | 0.7468 | 0.7453 | 0.7467 | 0.7413 |
| H8 | 2,884 | 0.7636 | 0.7560 | 0.7598 | 0.7587 | 0.7585 | 0.7666 |
| H9 | 2,884 | 0.7742 | 0.7399 | 0.7518 | 0.7512 | 0.7514 | 0.7475 |

**Reading the table:** federation clearly helps the largest hospital (local is weakest there, at 0.653,
and every federated method adds about +0.04). It hurts several smaller hospitals, especially H9, which
does well alone (0.774) and drops under every federated method.

### 5c. Topology weights on this split

| Hospital | Patients | Share of data | Topo weight |
|---|---|---|---|
| Largest | 37,952 | 57% | 0.19 |
| One hospital | 3,322 | 5% | **0.73** |
| Other eight | 2,884–4,886 | 4–7% each | 0.004–0.025 each |

Equal weighting would give each hospital 0.10. Topo puts almost three-quarters of its weight on one
small, unusual hospital. This is very different from the Louvain split.

---

## 6. Results: Louvain split (3 hospitals)

Hospitals ranked by size within each seed, so "largest" always means the same group.

### 6a. Summary

| Method | Best-round F1 | Final-round F1 | Worst-hospital F1 | AUROC |
|---|---|---|---|---|
| Local only | 0.6838 ± 0.0167 | — | 0.6680 ± 0.0133 | 0.8308 ± 0.0038 |
| Pooled | 0.6851 ± 0.0091 | — | 0.6406 ± 0.0096 | 0.8297 ± 0.0014 |
| FedAvg | 0.6935 ± 0.0032 | 0.6671 ± 0.0151 | 0.6719 ± 0.0119 | 0.8195 ± 0.0040 |
| FedProx | 0.6930 ± 0.0041 | 0.6681 ± 0.0175 | 0.6753 ± 0.0136 | 0.8176 ± 0.0083 |
| FedMedian | 0.7020 ± 0.0037 | 0.6875 ± 0.0270 | 0.6783 ± 0.0147 | 0.8316 ± 0.0038 |
| FedTrimmedAvg | 0.7010 ± 0.0069 | 0.6874 ± 0.0233 | 0.6791 ± 0.0130 | 0.8329 ± 0.0045 |
| Uniform control | 0.7009 ± 0.0068 | 0.6877 ± 0.0229 | 0.6792 ± 0.0130 | 0.8329 ± 0.0045 |
| **Topo (T=1)** | **0.7027 ± 0.0038** | **0.6990 ± 0.0093** | **0.6829 ± 0.0097** | 0.8336 ± 0.0031 |

### 6b. Per hospital (mean F1 over seeds, best round)

| Hospital | Patients | Local | Pooled | FedAvg | FedMedian | FedTrim | Uniform | Topo |
|---|---|---|---|---|---|---|---|---|
| Largest | ~48,800 | 0.6785 | 0.6886 | 0.6923 | 0.7010 | 0.6997 | 0.6997 | **0.7011** |
| Middle | ~14,200 | **0.6818** | 0.6406 | 0.6719 | 0.6783 | 0.6791 | 0.6792 | **0.6829** |
| Smallest | ~4,000 | **0.7357** | 0.7111 | 0.7328 | 0.7416 | 0.7401 | 0.7400 | **0.7435** |

### 6c. Topology weights on this split

Weights stay close to equal (about 0.22–0.50 per hospital, against 0.33 for equal weighting). That is
why topo and the uniform control perform almost the same here.

---

## 7. What the results say

**Federation helps, but not evenly.**
- On the organism split, federation helps the largest hospital by about +0.04 and hurts several smaller hospitals.
- On the Louvain split, federation helps the largest hospital, but FedAvg hurts the middle and smallest ones.

**Topology is better than plain FedAvg, mostly for a simple reason.**
- Best-round: topo beats FedAvg on Louvain (+0.009), but the uniform control gets most of that gain.
  On the organism split, topo and FedAvg are tied.
- Final round: topo beats FedAvg on both splits (+0.032 on Louvain, +0.024 on the organism split).
  FedAvg's score drops during training, and topo's doesn't.
- Worst hospital on the organism split: topo is slightly worse than FedAvg (0.6893 against 0.6945).

**Topology does not beat the robust methods.**
- FedMedian and FedTrimmedAvg match or beat topo on almost every measure on both splits.
- The uniform control (equal weights) matches or beats topo on the organism split.
- So the topology signal itself is not clearly helping yet. Equal weighting and robust averaging do as well.

**Topo's advantage over local-only is small.**
- On Louvain, topo beats local on all three hospital sizes, but the middle-hospital gain is only +0.001.
- On the organism split, topo loses to local on three hospitals (H2, H5, H9).

---

## 8. Caveats

- **Pooled is undertrained.** It gets 32 full-batch optimizer steps, which is probably too few. Its low
  score (about 0.69) should not be read as the real ceiling. The 0.71 figure in the code is an old constant.
- **Best-round is optimistic.** Use the final-round columns for the fairest comparison.
- **Few seeds, few partitions.** Five seeds, and only one partition for the organism split. Differences
  under about 0.01 are not reliable.
- **Rank matching is approximate** on Louvain, because the partition changes with the seed.
- **Topo ran on a different machine.** The topo arm of the organism split ran on a Mac (CPU) and the
  other arms ran on Colab. The gap to the uniform control should be rerun on one machine before we
  treat it as a result.
- **FedAvg on Louvain is not yet reconciled.** An older doc reported 0.6785; this run gives 0.6935. Local-only
  matches the older doc, so the difference is specific to FedAvg, and we haven't found the cause.

---

## 9. Bottom line

1. Federation can help a hospital with a lot of data and can hurt hospitals that do well alone.
2. Our topology method beats plain FedAvg at the end of training, but simple equal weighting does
   almost as well.
3. Robust methods (FedMedian, FedTrimmedAvg) are the strongest group on both splits.
4. We are not yet at the point where topology clearly helps.

## 10. Next steps

1. Fix pooled's training budget so it can serve as a real ceiling.
2. Match hospitals by patient membership, not size rank, on the Louvain split.
3. Run topo and the uniform control on the same machine.
4. Add more seeds, especially for the topo vs. uniform comparison.
5. Find the cause of the FedAvg difference on Louvain.
