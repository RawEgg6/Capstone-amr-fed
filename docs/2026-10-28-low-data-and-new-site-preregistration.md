# Pre-registration and paper notes — where does federation help? (notebook 18)

**Recorded 2026-10-28, BEFORE any run of notebook 18.** Companion to
`2026-10-28-final-protocol-preregistration.md` (notebooks 16/17). Same rule: a change after seeing results is an
*iteration*, appended in section 8 with its date, and both versions are reported.

## 1. Why this experiment exists (and how to write it up honestly)

On our well-sized hospitals (smallest 2,900 patients in the organism split; 3,500 in Louvain) federated learning
**ties local-only on F1** and gives a small AUROC gain (`2026-10-28-validation-selected-results.md`). The literature
says that is expected: federation helps most when local data are scarce, and gives little or nothing — sometimes a
loss — to data-rich hospitals (Leave-Silo-Out prostate-biopsy study, *Int. J. Med. Inform.* 2025,
DOI 10.1016/j.ijmedinf.2025.106046). It also says very small sites can break plain FedAvg (Kamp et al., ICLR 2023,
https://iclr.cc/virtual/2023/poster/11012), so the benefit may be a **hump, not a monotone curve**. And a federated
model's main selling point is service to a hospital that has little data or was never part of training
(Leave-Center-Out, https://arxiv.org/pdf/2107.03901).

**Hypothesis (stated neutrally):** federation's advantage over local-only grows as local data shrink, at least over
some range, and a federated model (with a little fine-tuning) beats a from-scratch local model at a new hospital that
has few labelled patients. **We report every data level and both held-out hospitals, whatever they show.** If federation
does not help at some level, that level stays in the paper. The framing for the paper is "*when* does federation help",
with the high-data regime (notebooks 16/17) as one end of the curve and this experiment as the other — not "federation
helps" as a headline chosen after the fact.

## 2. Setting (both parts)

- **Split:** `dirichlet_ward_mixture(n_clients=10, alpha=0.5, seed)` — all hospitals see the same organisms and drugs and
  differ in ward mix, so there is something to share. The partition changes with the seed, so seeds 42–46 are **five
  different partitions** (hospital sizes about 900–15,000 patients), unlike the organism split (one partition).
  Rationale for not reusing the organism split: its hospitals hold disjoint organisms, which makes "borrowing from other
  hospitals" a harsh test and a held-out hospital's organisms unseen by everyone else.
- **Budget:** rounds = 8, local epochs = 4, local-only epochs = 32, hidden = 128. Seeds 42–46.
- **Protocol:** as in notebooks 16/17 — every hospital picks its own cutoff (29 cutoffs) and, for federated/fine-tuned
  models, the round/epoch on its own validation rows; test is read once at that choice.

## 3. Part A — low-data curve

- **Levels:** each hospital keeps a fraction f ∈ {1.0, 0.25, 0.10, 0.03} of its **train and validation** patients (at least
  2 of each); its **test patients never change** (`regimes.subsample_frame`), so test sets are identical at every level.
  Validation shrinks with training because a hospital with 60 labelled patients would not have a large validation set;
  this makes selection noisier for every arm alike.
- **Arms per level and seed (one run):** local-only (trained from scratch on that hospital's reduced data), FedAvg, and
  FedAvg + fine-tune (8 local epochs from the final global model, epoch and cutoff chosen on the reduced validation set).
  20 FedAvg runs in total (4 levels × 5 seeds).
- **Primary metric:** the **unweighted mean over the 10 hospitals** of per-hospital test macro-F1 (each hospital counts
  once, because the question is about the small hospital). Secondary: the size-weighted mean.
- **Comparisons (paired by seed, at each level):** FedAvg − local-only; FedAvg + fine-tune − local-only; FedAvg +
  fine-tune − FedAvg.
- **What counts as support for the hypothesis:** the FedAvg + fine-tune − local-only gap is positive with a 95% interval
  above zero at one or more reduced levels (f ≤ 0.25) and is larger there than at f = 1.0. If the gap is positive only at
  one intermediate level, we report a hump. If it is not positive anywhere, we report that federation did not help
  in this setting.

## 4. Part B — a new hospital

- **Hold-out rule (fixed now, not by outcome):** for each seed, hold out (i) the 5th-largest of the 10 hospitals
  (`median`) and (ii) the smallest (`smallest`) (`regimes.holdout_hospital`). FedAvg trains on the other nine.
  10 federated runs (2 rules × 5 seeds) on 9 hospitals, no local-only or fine-tune baselines inside them.
- **Labelled data at the new hospital:** k ∈ {25, 100, 400, all} training patients, with `max(30, k/2)` validation
  patients (`regimes.slice_frame`); the hospital's **full test set** is scored. If a hospital has fewer than k training
  patients, k means all of them (the smallest hospital does at k = 100/400; reported as is).
- **Arms:** (1) **zero-shot** — the federated model unchanged, cutoff chosen on the validation slice; (2) **fine-tuned** —
  the federated model plus up to 8 local epochs (epoch 0 is the zero-shot model, epoch and cutoff chosen on the
  validation slice, so fine-tuning can only match or beat zero-shot on validation); (3) **scratch** — a fresh model
  trained for 32 epochs on the same slice (checkpoint every 4, same selection).
- **Comparisons (paired by seed, per k and per rule):** fine-tuned − scratch (primary), zero-shot − scratch.
- **Support for the hypothesis:** fine-tuned − scratch positive with a 95% interval above zero at small k, shrinking as k grows.

## 5. What is saved

Per run, as in the final protocol (confusion counts per cutoff, probability histograms, AUROC, per-hospital values,
weights, provenance); for Part A also the reduced sizes (`train_frac`, `n_triples`); for Part B per k the zero-shot /
fine-tuned / scratch scores, chosen epochs and cutoffs, patient counts and per-epoch counts. File
`results/amr_regimes_<TAG>.json` (not committed).

## 6. Compute

About 20 FedAvg runs (Part A, each with local-only and fine-tune) and 10 nine-hospital FedAvg runs plus 8 small adaptation
sets per seed (Part B). The largest hospital has 7–15k patients, so each run is lighter than a notebook-17 run; expect
less total time than either notebook 16 or 17. Resume is seed by seed.

## 7. Limits we state up front

- One data-subsampling scheme (uniform fraction per hospital) and one split family; other kinds of scarcity (one tiny site
  among large ones, rare organisms) are not tested.
- Five seeds, ten hospitals: interval widths of about 0.003–0.01 F1 at the smallest levels are likely.
- The unweighted mean over hospitals gives small hospitals equal say; the weighted version is reported beside it.
- A new hospital here comes from the same dataset and shares the organism and drug vocabulary, so this is a best case
  for transfer, not a test on another health system.
- Per-hospital cutoffs on tiny validation sets are noisy for every arm alike.
- No model-similarity or topology-weighted aggregation is part of this experiment.

## 8. Iterations (append below, dated)

_None yet._

## 9. Notes for the paper

- Lead with the protocol finding (selection on test and a fixed cutoff change federated-vs-local conclusions by 0.02–0.04),
  then the high-data result (federation ties local-only on F1, small AUROC gain), then this section: *when* does federation
  help — the data-scarcity curve and the new-hospital test.
- Cite for the framing: FedAvg + fine-tuning as a tough baseline (Cheng, Chadha, Duchi, arXiv 2108.07313; Matsuda et al.,
  arXiv 2206.13190); the data-richness dependence (the Leave-Silo-Out study above); the small-site failure mode of FedAvg
  (Kamp et al., ICLR 2023); out-of-site evaluation (arXiv 2107.03901).
- Report in the paper exactly what is pre-registered here, including levels where federation does not help.
