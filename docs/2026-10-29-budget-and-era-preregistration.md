# Pre-registration — notebook 21: budget-matched fine-tune test (Part A) and the era split (Part B)

**Recorded 2026-10-29, BEFORE any run of notebook 21.** Changes after seeing results are *iterations*, appended in section 7 with
their date; both versions are reported. Offline evidence that motivated this design: `2026-10-29-label-heterogeneity-and-fingerprints.md`.

## 1. Why

1. **Budget confound.** In notebooks 16-19, FedAvg + fine-tune got 40 local epochs per hospital while local-only and FedAvg got 32, and
   every arm was still improving at its budget. Reading the saved runs at smaller budgets (`federated/budget.py`, exact) shows that
   the last 8 epochs were worth +0.009 / +0.010 to local-only and +0.007 / +0.008 to FedAvg (org10 / Louvain, size-weighted F1),
   while fine-tuning's 8 epochs were worth +0.005 over FedAvg. **The whole fine-tune lead could be extra training.**
2. **No concept shift in the headline splits.** Patient subsets of one institution share P(resistant | organism, antibiotic,
   history). Measured against a random split, the pair-level label differences are 1.08x random on org10 and 1.10x on ward-Dirichlet
   (after conditioning on patient history). Time is the strongest natural axis: two eras give 1.49x between groups and 1.0x inside a group.
   No similarity method can beat FedAvg + fine-tune where there is nothing to be similar about; the era split is where it can.

## 2. Part A — budget-matched comparison (Louvain 3 hospitals, organism-10; seeds 42-46)

Runs per seed: `long` = FedAvg 24 rounds x 4 local epochs + local-only 104 epochs + fine-tune 8 epochs; `ft8` = FedAvg 8 rounds + fine-tune 8.
Smaller budgets come from `long` (training is identical up to round R / epoch E: constant LR, deterministic seeds). The selection rule is the
final protocol (round and per-hospital cutoffs on validation, test read once). Numbers in brackets = local epochs per hospital.

**Primary metric:** size-weighted test macro-F1, per-hospital cutoffs (`phc_f1`, as in notebooks 16/17).

**Primary comparisons (paired by seed, 95% t-interval):**
- A1 at 40 epochs: (a) FedAvg@8 + fine-tune (40) − local-only@40; (b) FedAvg@8 + fine-tune (40) − FedAvg@10 (40).
- A2 near convergence: (c) FedAvg@24 + fine-tune (104) − local-only@104; (d) FedAvg@24 + fine-tune (104) − FedAvg@24 (96).

**Reading, fixed in advance:**
- "Fine-tuning helps beyond its extra epochs" at a budget only if both of that budget's intervals are above zero.
- If (b) or (d) includes zero, the earlier fine-tune advantage is reported as **explained by training budget** at that budget.
- If (a) or (c) includes zero, "federation + fine-tune beats local-only" is withdrawn at that budget.
- Plateau: if local-only's best epoch is 104 in most hospital-seed pairs or FedAvg's selected round is 24 in most seeds, the 104-epoch
  comparison is reported as still budget-limited.

**Secondary:** FedAvg@10 − local@40, FedAvg@24 − local@104, local@104 − local@32, FedAvg@24 − FedAvg@8, on size-weighted, unweighted and
worst-hospital F1. Determinism check: FedAvg@8 from `long` must equal the `ft8` run's FedAvg (|diff| < 0.002; otherwise reported).

## 3. Part B — era split (`partition.era_cluster_split(n_clients=10, n_eras=2)`; seeds 42-46 = five partitions)

10 hospitals in 2 groups of 5; group 0 = patients whose first culture is in the earlier half of the timeline, group 1 = the later half;
patients are random within a group. Rounds 8 x 4 local epochs; **every arm ends with the same 8-epoch fine-tune** (40 local epochs);
local-only gets 40 epochs. Temperature fixed at **T = 0.5** for every similarity arm (no tuning; 0.5 was chosen on validation in
notebooks 16 and 19).

| Arm | What it is |
|---|---|
| `fedavg` | FedAvg + fine-tune (also trains local-only 40) |
| `v3_topo` | v3 personalized aggregation, current graph fingerprint, + fine-tune |
| `v3_ab` | v3 with the cross-antibiogram fingerprint (`label_fingerprints.cross_antibiogram_loss`, train rows), + fine-tune |
| `v3_ab_shuffled` | `v3_ab` with fingerprints handed to the wrong hospitals (seeded derangement) |
| `oracle` | each hospital averages equally with the 5 hospitals of its true era group, + fine-tune: an upper bound for clustering |
| `simft` | SimFT tau = 0.5, self-floor 0.25, + fine-tune (also saves the cross-loss matrices for the offline fingerprint test) |

**Primary metric:** unweighted mean over hospitals of per-hospital test macro-F1 (own validation cutoff) after fine-tuning
(hospitals are equal-sized here, so it nearly equals the size-weighted mean).

**Primary comparisons (paired by seed):**
- B0 headroom: oracle − FedAvg + fine-tune.
- B1: v3_ab − FedAvg + fine-tune; B2: v3_ab − v3_ab_shuffled.

**Reading, fixed in advance:**
- If B0's interval does not exclude zero on the positive side: **even perfect grouping does not beat FedAvg + fine-tune on this split**;
  similarity-weighted aggregation is reported as unhelpful on ARMD-derived splits, and B1/B2 are reported but not claimed.
- The label fingerprint "works" only if B1 and B2 are both above zero.
- Secondary: v3_topo − FedAvg + fine-tune, simft − FedAvg + fine-tune, v3_ab − v3_topo, oracle − v3_ab, FedAvg + fine-tune − local-only,
  FedAvg − local-only; size-weighted and worst-hospital versions; weight placed on same-era vs other-era hospitals per arm.
- Offline (no training): Spearman of each fingerprint's dissimilarity with SimFT's measured transfer gap on this split (`budget_runs.offline_fingerprint_test`).

## 4. What is saved

Everything the final protocol saves (confusion counts per round and checkpoint, histograms, per-hospital values, weights, fingerprints,
provenance), plus SimFT's per-round W and L. Files: `results/amr_budget_<setting>_<TAG>.json`, `results/amr_era10_<TAG>.json`.

## 5. Compute

Part A: 2 runs per seed per setting; `long` costs about 3x a plain run. Part B: 6 runs per seed; SimFT about 1.5-2x. Estimated 2-3 h and 1.5-2 h on a T4.

## 6. Limits

- Five seeds; Louvain has three distinct partitions; org10 one; the era split five.
- The era split is semi-natural: "hospitals" are time windows of one lab, not different labs. It stands in for labs on different
  testing panels / breakpoint editions; it is not evidence about real multi-site data.
- Part B runs at 8 rounds; if Part A shows the fine-tune lead is budget, Part B's comparisons are still budget-matched (every arm fine-tunes).
- Many comparisons, no multiplicity correction; p between 0.01 and 0.10 are hints.

## 7. Iterations (append below, dated)
