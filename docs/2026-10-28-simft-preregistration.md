# Pre-registration — SimFT (personalized aggregation by function-space similarity + final fine-tune), notebook 19

**Recorded 2026-10-28, BEFORE any run of notebook 19.** Changes after seeing results are *iterations*, appended in section 9 with their date;
both versions are reported.

## 1. Why this method, and what the evidence says (research done 2026-10-28)

Our own results: static-fingerprint similarity (v3) does not beat its shuffled control (Louvain: +0.0000, p = 0.96), and FedAvg + fine-tune is the
best arm so far. A model-similarity method therefore has to (i) measure similarity better than a static fingerprint and (ii) compete with
fine-tuning. What the literature says, and what we took from it:

| Finding | Source | Design consequence |
|---|---|---|
| Parameter-space distances do not reliably track similarity in *behaviour*; models with close parameters can behave differently | NTK-guided aggregation, https://arxiv.org/pdf/2608.12108 | Do **not** use cosine of weights/updates as the similarity |
| Cosine over dense network weights suffers the curse of dimensionality | FedPref, https://arxiv.org/pdf/2501.13604 | same |
| Parameter similarity rises as training continues, so similarity thresholds depend on training length | Irida Labs tinyML note (cosine clustering) | use a scale-free similarity recomputed every round, not a fixed threshold |
| Score each received model on the receiving client's own validation data and weight by the result; reported to beat FedAvg and local training on non-IID CIFAR-100 (authors' own experiments) | FedFomo, Zhang et al., ICLR 2021, https://arxiv.org/pdf/2012.08565 | **Function-space similarity**: loss of hospital j's model on hospital i's validation rows |
| Server-side collaboration graph from model similarity *and* dataset size, personalized aggregation per client; authors report beating 14 baselines | pFedGraph, Ye et al., ICML 2023, https://proceedings.mlr.press/v202/ye23b.html | personalized mixing matrix; keep a self-weight |
| Similarity-driven weight updates can make the graph nearly reducible and induce self-clustering | Controlled Collaboration Geometry, ICML 2026 (listing: https://icml.cc/virtual/2026/poster/61399) | keep a **self-floor**, softmax with temperature, no hard graph |
| FedAvg + local fine-tuning is a tough baseline that dedicated personalization methods often fail to beat | Cheng, Chadha, Duchi, https://arxiv.org/pdf/2108.07313; Matsuda et al., https://arxiv.org/pdf/2206.13190 | SimFT **ends with the same fine-tune** as the baseline; FedAvg + fine-tune is the primary comparison |

FedFomo's head-to-head against fine-tuning was not found in the search results; that comparison is what this experiment provides.

## 2. The method

Hospitals 1..n; model of hospital j after local training in a round: θ_j.

1. **Cross-evaluation.** Every hospital i scores every θ_j on its own validation **half A** (half of its validation *patients*, by parity of the
   patient node id): L[i, j] = class-weighted BCE. (In a deployment each hospital scores the received models itself and returns only these numbers;
   models are therefore visible to the other hospitals, a privacy cost FedAvg does not have.)
2. **Weights.** d_ij = L[i, j] − L[i, i]; s_i = mean over j ≠ i of |d_ij|; z_ij = d_ij / (τ · s_i);
   W[i] = β · e_i + (1 − β) · softmax_j(−z_ij), with **self-floor β = 0.25**. (`simft.simft_weights`.)
3. **Personalized models.** θ_i ← Σ_j W[i, j] θ_j; hospital i starts the next round from θ_i and is evaluated on θ_i.
4. **After the last round** each hospital fine-tunes its own θ_i for 8 local epochs (checkpoints at 0, 2, 4, 6, 8; same optimiser as the FedAvg + fine-tune baseline).
5. **Selection.** Validation **half B** (never used for the weights) chooses the round (size-weighted), each hospital's cutoff (29 cutoffs) and fine-tune epoch;
   test is read once at the choice. The temperature τ ∈ {0.5, 1.0} is chosen by the mean validation-B score over seeds before the primary comparisons are read.
6. **Ablation `SimFT-flat`:** identical, but W[i] = β · e_i + (1 − β) · uniform. Any gain of SimFT over SimFT-flat is attributable to the similarity signal.

Round 1 starts from a common initialisation, so the first weights are close to flat; this is expected.

## 3. Settings (one partition per seed; seeds 42–46)

| Setting | Split | Hospitals | Why |
|---|---|---|---|
| **conflict** | `antibiogram_conflict(n_clients=5)` (deterministic, taper shares 0.55^i, floor 3%; design and validity checks in `2026-10-18-antibiogram-conflict-split.md`) | 5, about 47 / 26 / 14 / 8 / 4% of patients | hospitals disagree about the same bug-drug pairs (concept shift): the case where a single shared model suffers and picking *similar* hospitals has a mechanism |
| **ward10** | `dirichlet_ward_mixture(10, alpha=0.5)` keeping **10%** of every hospital's train/validation patients (test unchanged) | 10 (partition changes with the seed) | the low-data regime where notebook 18 showed federation helps most |

Arms run in notebook 19: **conflict** — FedAvg (+ local-only, + fine-tune), SimFT τ = 0.5, SimFT τ = 1.0, SimFT-flat (4 arms × 5 seeds);
**ward10** — SimFT τ = 0.5, τ = 1.0, SimFT-flat only (3 arms × 5 seeds); its local-only, FedAvg and FedAvg + fine-tune come from notebook 18's `scarcity_f0.1`
(same seeds, partitions and subsample; run on a Tesla T4).

## 4. Metrics

**Primary metric:** the **unweighted mean over hospitals** of per-hospital test macro-F1, each hospital at its own validation-chosen cutoff (every hospital
counts once; the method is about each hospital's model). Secondary: size-weighted mean and worst hospital. SimFT is scored **after** its final fine-tune; the pre-fine-tune
scores are also reported.

## 5. Comparisons and the win rule (paired by seed)

**Primary (per setting):** (a) SimFT − FedAvg + fine-tune, (b) SimFT − local-only, (c) SimFT − SimFT-flat.
**SimFT "wins" in a setting only if all three paired 95% intervals are above zero.** A win in one setting and not the other is reported exactly so.
**Secondary (no multiplicity correction; p between 0.01 and 0.10 are hints):** SimFT − FedAvg (no fine-tune), SimFT before fine-tune − FedAvg,
SimFT-flat + fine-tune − FedAvg + fine-tune, FedAvg + fine-tune − local-only, FedAvg − local-only; the same on size-weighted and worst-hospital F1.

## 6. What is saved

Everything the final protocol saves (confusion counts per cutoff, histograms, per-hospital values, provenance), plus for every SimFT run the mixing matrix **W** and loss
matrix **L** of every round (`simft_rounds`) and the per-hospital fine-tune details.

## 7. Compute

About as long as notebook 16: conflict 4 arms (FedAvg about 3 units per seed, SimFT arms about 1.5–2 units because of the n² cross-evaluation per round),
ward10 3 small arms on 10%-size data. Resume is seed by seed.

## 8. Limits and honest expectations

- **Odds.** I expect a small effect at best (0.001–0.005 F1), at or below what five seeds can reliably detect (about 0.003–0.005). The most plausible place for a real effect is the
  conflict split; a tie is a likely outcome, and FedAvg + fine-tune may stay on top.
- **Protocol cost.** Cross-evaluation shows every hospital's model to every other hospital (n× communication) — unlike FedAvg. The write-up must say so.
- **Validation is split in halves** for SimFT (A for weights, B for selection), so SimFT selects on half as much validation as the baselines (conservative for SimFT, noisier selection).
- **Hardware pairing.** ward10 baselines were produced on a Tesla T4 (notebook 18); run notebook 19 on the same kind of machine if possible, otherwise small
  non-determinism differences enter the paired comparisons.
- **Training budget.** Everything is still improving at 8 rounds; SimFT and the baseline fine-tune get the same 8 extra epochs, so that comparison is budget-matched; the comparison with local-only is not
  (see `2026-10-28-low-data-and-new-site-results.md`, caveat 1).
- Five seeds; the conflict split is one partition (seeds vary training only); ward10 is five partitions.
- SimFT resembles FedFomo (validation-based weighting) and pFedGraph/FedAMP (personalized mixing); the novelty claim, if any, is the combination with a final fine-tune and the function-space
  similarity inside a graph-structured clinical setting. Related work must be cited, and a head-to-head against a FedFomo or pFedGraph implementation was not run here.

## 9. Iterations (append below, dated)

_None yet._
