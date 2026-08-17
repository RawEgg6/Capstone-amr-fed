# Split Experiments & Results — Full Record (context for teammates + their AI)

**Updated 2026-08-11** · AMR Federated Knowledge-Graph Capstone · Team: Vikram, Saffiya, Sanjana, Arshia · Guide: Dr. Swati Jagdale

> **How to use this doc:** it is written to be self-contained. A teammate (or their AI) should be able to read this file alone and understand *what splits were tried, under what protocol, what happened, and what it means for the project*. All numbers are from real Colab runs logged in `res.txt` and our docs in this folder. Where a number is a mean over seeds it is written `mean ± std`.

---

## 1. Project context (30 seconds)

We simulate several "hospitals" as **different subsets of one real dataset** (ARMD — Stanford antimicrobial-resistance data; 10 of 16 CSV tables). Each hospital trains a local **GNN** (heterogeneous GraphSAGE, "AMR-SAGE") to predict **whether a patient's organism is resistant to a given antibiotic** (binary: Resistant+Intermediate vs Susceptible) on the `(organism, tested, antibiotic)` edge.

Three baselines are always compared:

| Baseline | What it is | What it represents |
|---|---|---|
| **Pooled** | ONE model trained jointly on **every hospital's data** | The theoretical ceiling: "everyone shares all data" (privacy-violating, but the reference point) |
| **Local-only** | Each hospital trains **alone on its own data**, no federation | "Every hospital does its own thing, no collaboration" |
| **FedAvg** | Hospitals train locally; a server **averages their model weights** each round (Flower) | The standard federated baseline — collaboration WITHOUT sharing patient data |

**The whole point of the novelty:** a **topology-aware aggregator** (Phase 5) should beat plain FedAvg by weighting hospitals based on *graph structure*. But it only matters if there's **headroom** — i.e. a setting where FedAvg visibly fails. The `headroom_gate` is our acceptance test for "is a split hard enough to show headroom."

### Metrics glossary (so the numbers mean the same thing to everyone)

- **Macro-F1** — average of the F1 of the *resistant* and *susceptible* classes equally (the model can't coast on the common class). Our headline metric. **Pooled ceiling ≈ 0.71.**
- **AUROC** — probability the model ranks a random resistant case above a random susceptible one; the metric the AMR literature reports. **Pooled ≈ 0.84.**
- **Worst-hospital F1** — the F1 of the *worst-performing hospital* (fairness metric). FedAvg's real value is lifting these underserved sites.
- **Matched budget** — FedAvg's total local training (rounds × local epochs) is set equal to what local-only / pooled get, so the comparison is fair. v2 protocol: **rounds=8, local_epochs=4 → 32 epochs**, model width `hidden=128`.
- **Patient-history features** — a patient's own prior resistance results (temporally strict) are fed to the decoder. This is what broke the old 0.66 ceiling → 0.71.

---

## 2. What each "split" is (the patients→hospitals assignment)

A **split** decides which patients end up in which simulated hospital. We tried every family of non-IID in the FL literature to find ones that make hospitals *genuinely different*:

| # | Split | How hospitals differ | Family |
|---|---|---|---|
| 1 | **Ward-Dirichlet (α)** | Each hospital = a random *blend* of wards (α controls how different) | Quantity/feature skew (baseline) |
| 2 | **Label-Dirichlet (β)** | Hospitals have different **resistant/susceptible ratios** | Label skew |
| 3 | **Specimen** | One hospital per culture source (**urine / blood / respiratory**) | Real / natural split |
| 4 | **Organism-community** | Each hospital sees a **disjoint set of organisms** (bugs) | Topology (structural) |
| 5 | **Homophily spectrum** | Hospitals span heterophilic ↔ homophilic (is resistance *clustered* by bug or scattered?) | Topology (single-axis) |
| 6 | **Degree skew** | Hospitals span sparse-rare-bugs ↔ hub-common-bugs | Topology (single-axis) |
| 7 | **Topology corners** | **2-D crossed**: homophily × hubness quadrants → 8 hospitals (4 corners × 2 buckets) | Topology (2-D crossed) |
| 8 | **Louvain communities** | Hospitals = **community detection** on the organism–antibiotic graph (FedGTA-style) | Topology (community) |

---

## 3. The results, split by split

### Phase-1 model ceiling (no split — all data)

| Model | macro-F1 | AUROC |
|---|---|---|
| Local GNN, core features only | 0.660 | 0.771 |
| **+ patient-history features** | **0.707–0.712** | **0.836** |

**Takeaway:** patient-history is the literature's #1 predictor and broke our ceiling. **0.71 / 0.84 is the reference the whole federation story targets.**

---

### Generation 1 — the baseline splits (2026-08-06/07, strong model, 3 seeds)

#### 1. Ward-Dirichlet α-sweep (5 hospitals)

| α | local-only | FedAvg-best | gain (best−local) | worst-hosp local |
|---|---|---|---|---|
| 0.1 | 0.6974 ± 0.0025 | 0.7036 ± 0.0029 | +0.006 ± 0.005 | ~0.685 |
| 0.5 | 0.6883 ± 0.007 | 0.7007 ± 0.0023 | +0.012 ± 0.009 | ~0.671 |
| 1.0 | 0.6913 ± 0.0058 | 0.7000 ± 0.0084 | +0.009 ± 0.011 | ~0.670 |

**Result:** FedAvg beat local-only in **all 9 runs** (sign test p≈0.002 — federation reliably helps). FedAvg lands ~0.70 vs the 0.71 pooled ceiling → recovers ~99% of centralized accuracy **without sharing patient data**. That's the clean Phase-3 result.

**The honest problem:** every gain's error bar overlaps zero, and the whole spread is tiny (local 0.69 → FedAvg 0.70 → pooled 0.71, ~2 points). A ward blend leaves hospitals near-IID in the dimension FedAvg actually struggles with → **no headroom for a smarter aggregator.**

#### 2. Label-Dirichlet on resistance rate (5 hospitals) — ❌ DEAD END

| β | local-only | FedAvg-best | gain | worst-hosp local → FedAvg | FedAvg>local |
|---|---|---|---|---|---|
| 0.1 | 0.678 ± 0.024 | 0.601 ± 0.112 | **−0.077 ± 0.128** | 0.493 → 0.516 | 1/3 |
| 0.5 | 0.687 ± 0.013 | 0.690 ± 0.006 | +0.003 ± 0.006 | 0.627 → 0.625 | 1/3 |

**Result:** informative *negative*. β=0.5 did nothing; β=0.1 went degenerate (one seed had a **1-patient hospital** — no size guard) and FedAvg collapsed toward the majority baseline (0.44). **Why:** our decoder is driven by strong per-test features (patient-history, organism/antibiotic identity), so it learns feature→resistance regardless of a hospital's class balance. **Label-prior skew is the wrong lever for a feature-driven model.** Worth stating explicitly in the thesis.

#### 3. Specimen split (urine/respiratory/blood) — ✅ CLEAN WIN

| local-only | FedAvg-best | gain | worst-hosp local → FedAvg |
|---|---|---|---|
| 0.696 ± 0.002 | 0.715 ± 0.001 | **+0.019 ± 0.004** | 0.679 → 0.702 (**+0.023 ± 0.007**) |

FedAvg beats local in 3/3 seeds, gain doesn't overlap zero (unlike every ward split), worst hospital (urine, n≈52k, the hardest) gains +0.023. **Works because specimen source induces topological + feature heterogeneity** (urine vs blood vs respiratory involve different organisms/antibiotics → structurally different subgraphs) — exactly what a topology-aware aggregator keys on. Most clinically defensible split.

#### 4. Organism-community split (disjoint bugs) — ✅ STRONGEST

| local-only | FedAvg-best | gain | worst-hosp local → FedAvg |
|---|---|---|---|
| 0.696 ± 0.008 | 0.719 ± 0.006 | **+0.023 ± 0.006** | 0.665 → 0.703 (**+0.037 ± 0.010**) |

Largest clean gain AND the largest worst-hospital lift of any split. The giant hospital (one dominant organism, n≈38k) is the worst alone (~0.66) and gains most from federation (+0.02…+0.05). **This was the planned Phase-5 setting.** (Note: the first run had OOM errors on the giant hospital — it skipped ~half of fit rounds — but evaluation never failed, and since it *under*-trained yet FedAvg still won, the gain is a *conservative floor*. Fixed via `task.free_gpu()`.)

---

### Generation 2 — hard topology splits + headroom gate (2026-08-09 to now)

We built deterministic topology splits (in `partition.py`): **homophily**, **degree-skew**, **topology corners** (2-D crossed), **louvain** (community detection). The **`headroom_gate`** then checks: *does FedAvg measurably trail pooled?* (so a topology-aware aggregator has something to chase).

**Gate conditions (all must hold, mean over 3 seeds):**
1. `FedAvg-best > local-only` (federation still helps vs training alone)
2. `worst_fed ≤ pooled_worst − 0.02` **OR** `FedAvg-best ≤ pooled − 0.01` (FedAvg trails pooled)
3. `pooled_worst ≥ 0.60` (pooled stays strong — split is hard but not hopeless)

#### v1 protocol (hidden=64, 6 rounds × 3 epochs = 18 budget) — ALL ❌

| Split | n | Pooled | FedAvg-best | Local-only | worst local→Fed | FedAvg vs pooled | PASS? |
|---|---|---|---|---|---|---|---|
| homophily | 8 | 0.6732 ± .0065 | 0.7014 ± .0025 | 0.6918 ± .0017 | 0.6695 → 0.6930 | **+0.028** | ❌ |
| degree-skew | 8 | 0.6785 ± .0056 | 0.6928 ± .0085 | 0.6862 ± .0014 | 0.6371 → 0.6790 | **+0.014** | ❌ |
| topology corners | 8 | 0.6881 ± .0016 | 0.7036 ± .0014 | 0.6900 ± .0076 | 0.6044 → 0.6751 | **+0.015** | ❌ |
| louvain | 3* | 0.6596 ± .0096 | 0.6976 ± .0009 | 0.6877 ± .0062 | 0.6722 → 0.6614 | **+0.038** | ❌ |

\* Louvain: requested 5, got 3 communities (~73%/21%/5%). **Worst hospital got WORSE with FedAvg** (0.6722→0.6614) — the *right shape*, but pooled was too weak.

**v1 diagnosis:** FedAvg beat pooled on every split. Suspected **pooled was under-trained** (0.66–0.69 vs its 0.71 potential) — narrow model + short budget starved the joint model. → Designed **v2**: wider model (128), more budget (8r×4e = 32).

#### v2 protocol (hidden=128, 8 rounds × 4 epochs = 32 budget) — STILL ❌

| Split | Pooled | FedAvg-best | Local-only | FedAvg vs pooled | worst local→Fed |
|---|---|---|---|---|---|
| topology corners | 0.6616 ± .0051 (worst 0.6269) | 0.7037 ± .0024 | 0.6777 ± .0064 | **+0.042** | 0.6425 → 0.6678 (+0.025) |
| louvain | 0.6876 ± .0141 (worst 0.6277) | 0.7012 ± .0023 | 0.6803 ± .0179 | **+0.014** | 0.6657 → 0.6716 (+0.006) |

**Gate (topology corners, v2):** `fed>local` ✓ · `pooled_worst≥0.60` ✓ · **gap ✗ (FedAvg ahead +0.035)** → **FAIL**.

**The v2 diagnosis — this is the important finding.** Two things:

1. **Pooled is NOT the ceiling on these splits; FedAvg is.** Smoking gun (louvain seed 42, the tiny 5% community): local-only **0.7423**, pooled **0.6735** (−0.069 collapse), FedAvg **0.7308**. Root cause: `run_pooled` **size-weights each hospital's loss by its patient count** (`loss × n_tr/total_tr`), so a 5% hospital contributes ~5% of pooled's gradient → pooled *under-trains rare/small regimes*. FedAvg trains each hospital **independently**, so it rescues exactly what pooled starves. This is a mechanism, not a bug — it's why FedAvg wins on unbalanced topology splits.

2. **FedAvg results are non-deterministic run-to-run.** Cells 9 and 11 ran the *identical* config (same split, same seeds, same v2 protocol) yet got FedAvg 0.7037 vs 0.6967; seed 42 climbed monotonically in one run and peaked-then-decayed in the other. Cause: `torch.manual_seed(seed)` is set in the parent process, but the Flower/Ray client processes build their models with **unseeded** lazy init — every actor's init is random each run. Pooled and local-only ARE deterministic; FedAvg is not. **We cannot trust ±0.007 verdicts around a 0.01 threshold until client seeding is fixed.**

---

## 4. Summary table — every split at a glance (macro-F1, mean over 3 seeds)

| Split | Protocol | local-only | FedAvg-best | Pooled | FedAvg vs pooled | Worst-hosp gain | FedAvg>local | Verdict |
|---|---|---|---|---|---|---|---|---|
| Ward-Dirichlet α=0.1 | 10r×6e | 0.697 | 0.704 | 0.71* | — | ~+0.02 | 3/3 | Baseline OK, no headroom |
| Ward-Dirichlet α=0.5 | 10r×6e | 0.688 | 0.701 | 0.71* | — | ~+0.03 | 3/3 | Baseline OK, no headroom |
| Ward-Dirichlet α=1.0 | 10r×6e | 0.691 | 0.700 | 0.71* | — | ~+0.03 | 3/3 | Baseline OK, no headroom |
| Label-Dirichlet β=0.5 | 10r×6e | 0.687 | 0.690 | — | — | 0.627→0.625 | 1/3 | ❌ dead end |
| Label-Dirichlet β=0.1 | 10r×6e | 0.678 | 0.601 | — | — | 0.493→0.516 | 1/3 | ❌ degenerate |
| Specimen | 10r×6e | 0.696 | 0.715 | 0.71 | ~0 | 0.679→0.702 (+0.023) | 3/3 | ✅ clean win |
| Organism-community | 10r×6e | 0.696 | 0.719 | 0.71 | +0.009 | 0.665→0.703 (+0.037) | 3/3 | ✅ strongest |
| Homophily | v1 (64, 6r×3e) | 0.692 | 0.701 | 0.673 | **+0.028** | 0.670→0.693 | — | ❌ gate |
| Degree-skew | v1 (64, 6r×3e) | 0.686 | 0.693 | 0.679 | **+0.014** | 0.637→0.679 | — | ❌ gate |
| Topology corners | v1 (64, 6r×3e) | 0.690 | 0.704 | 0.688 | **+0.015** | 0.604→0.675 | — | ❌ gate |
| Topology corners | v2 (128, 8r×4e) | 0.678 | 0.704 | 0.662 | **+0.042** | 0.643→0.668 | — | ❌ gate |
| Louvain | v1 (64, 6r×3e) | 0.688 | 0.698 | 0.660 | **+0.038** | 0.672→0.661 (worse) | — | ❌ gate (right shape) |
| Louvain | v2 (128, 8r×4e) | 0.680 | 0.701 | 0.688 | **+0.014** | 0.666→0.672 | — | ❌ gate (closest) |

\* Ward splits used the hardcoded 0.71 pooled reference (pre apples-to-apples fix); later runs compute a matched pooled.

---

## 5. What this means for Phase 5 (the current decision)

**The consistent finding across every calibration attempt:** on topology-divergent splits, **FedAvg beats pooled** — the gate fails in the *wrong direction*. "Close the gap to pooled" would mean *getting worse*, because pooled is the one that under-serves rare/small communities (size-weighted loss).

Two honest framings for the topology-aware aggregator (we're deciding between these):

- **Path A — "FedAvg's averaging compromises a hospital."** Hunt splits where FedAvg *degrades* a hospital below its local-only baseline (louvain seed 44 showed one: −0.020). Phase 5 then = *recover the hospital FedAvg's averaging dragged down*, without losing the others. Gate becomes: `worst_fed < worst_local − δ` while other hospitals stay strong.
- **Path B — make pooled a fair ceiling first.** De-weight pooled's loss (train each hospital's share equally, not by patient count) so rare communities get learned. If pooled then overtakes FedAvg, the original "close the gap to centralized" story holds.

**Known open issue (fix first, regardless of path):** seed the Flower client models so FedAvg is reproducible — currently non-deterministic run-to-run.

---

## 6. Where the code / evidence lives

- **Splits:** `src/amr_fed/partition.py` (`dirichlet_ward_mixture`, `label_dirichlet`, `specimen_baseline`, `organism_community`, `homophily_split`, `degree_skew_split`, `topology_split`, `louvain_split`)
- **Runner / gate:** `src/amr_fed/federated/run.py` (`run_fedavg`, `run_multiseed`, `run_pooled`, `headroom_gate`)
- **Federated client/task code:** `src/amr_fed/federated/{client_app,server_app,task}.py`
- **Experiment notebook:** `notebooks/04_federated.ipynb` (cells 4–11; cells 9–11 are the v2 calibration cells)
- **Raw run log:** `res.txt` (the committed Colab log)
- **Earlier writeups:** `docs/2026-08-06-federation-results.md`, `docs/2026-08-09-headroom-calibration.md`
- **Feature detail:** `docs/2026-08-05-phase1-results.md`
