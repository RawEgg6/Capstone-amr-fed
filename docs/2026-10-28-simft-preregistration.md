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

## 3. Settings (one partition per seed; seeds 42–46) — only the SimFT arms are trained

In every setting the baselines (local-only, FedAvg, FedAvg + fine-tune) are **read from the notebook that already ran them on the same split and seeds**;
notebook 19 trains only SimFT at τ = 0.5, SimFT at τ = 1.0 and SimFT-flat (3 arms × 5 seeds per setting).

| Setting | Split | Hospitals | Baselines from | Why |
|---|---|---|---|---|
| **louvain** | `louvain_split(n_clients=3)` | 3 (about 73 / 21 / 5%) | notebook 16, `amr_louvain_final_all.json` (Mac CPU) | main-paper split; cheap |
| **org10** | `organism_community(n_clients=10)` | 10 (one partition; sizes 2.9k–38k) | notebook 17, `amr_org10_final_all.json` (FedAvg arm only) | main-paper split; hospitals hold different organisms, so some are genuinely more alike than others |
| **ward10** | `dirichlet_ward_mixture(10, alpha=0.5)`, **10%** of every hospital's train/validation patients kept (test unchanged) | 10 (partition changes with the seed) | notebook 18, `amr_regimes_all.json`, `scarcity_f0.1` (Tesla T4) | the low-data regime where federation helped most |

Louvain and org10 are the primary settings (the headline splits of the paper, not chosen because the method is expected to win there); ward10 is an additional low-data setting.
Each setting is judged separately by the win rule in section 5.

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

3 arms × 5 seeds per setting, no FedAvg rerun. SimFT arms cost about 1.5–2 units per seed (a unit is one plain federated run) because of the n² cross-evaluation per round:
louvain about 45 minutes on a Mac CPU; ward10 small (10%-size data); org10 the heaviest (10 × 10 scoring passes per round on a 38k-patient graph), best on Colab. Resume is seed by seed.

## 8. Limits and honest expectations

- **Odds.** I expect a small effect at best (0.001–0.005 F1), at or below what five seeds can reliably detect (about 0.003–0.005). On louvain and org10 federation barely beats local-only on F1 (only fine-tuning helps), so there is little room; org10 (different organisms per hospital) and ward10 at
  10% data are the likelier places for an effect. A tie is a likely outcome, and FedAvg + fine-tune may stay on top.
- **Protocol cost.** Cross-evaluation shows every hospital's model to every other hospital (n× communication) — unlike FedAvg. The write-up must say so.
- **Validation is split in halves** for SimFT (A for weights, B for selection), so SimFT selects on half as much validation as the baselines (conservative for SimFT, noisier selection).
- **Hardware pairing.** louvain baselines were produced on a Mac CPU (notebook 16) and ward10 baselines on a Tesla T4 (notebook 18); run each setting on the same kind of machine if possible, otherwise small
  non-determinism differences enter the paired comparisons (org10: notebook 17 ran on a Mac CPU).
- **Training budget.** Everything is still improving at 8 rounds; SimFT and the baseline fine-tune get the same 8 extra epochs, so that comparison is budget-matched; the comparison with local-only is not
  (see `2026-10-28-low-data-and-new-site-results.md`, caveat 1).
- Five seeds; org10 is one partition (seeds vary training only), louvain has three distinct partitions, ward10 five.
- SimFT resembles FedFomo (validation-based weighting) and pFedGraph/FedAMP (personalized mixing); the novelty claim, if any, is the combination with a final fine-tune and the function-space
  similarity inside a graph-structured clinical setting. Related work must be cited, and a head-to-head against a FedFomo or pFedGraph implementation was not run here.

## 9. Iterations (append below, dated)

- **2026-10-28 (before any run of notebook 19):** the antibiogram-conflict setting was dropped. Reasons: (i) its FedAvg baseline would have had to be rerun under the final protocol, while the other splits already have
  local-only / FedAvg / FedAvg + fine-tune; (ii) the earlier pre-registered experiment on that split (2026-10-18, notebook 08) found that FedAvg *helps* the minority hospital (+0.028) — the concept-shift condition did not reproduce, so the
  rationale I gave for the split (a mechanism for similarity-based weighting) was too optimistic. Louvain and org10 (existing baselines, headline splits) replace it as the primary settings; ward10 at 10% data stays as the low-data setting.
