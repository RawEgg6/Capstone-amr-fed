# Federation Results & Status (Phases 2–3)

**AMR Federated Knowledge-Graph Capstone** · updated 2026-08-06
**Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale

---

## TL;DR (updated — we broke the earlier ceiling)

Adding **patient-history predictors** (the personalized-antibiogram signal — a patient's
own prior resistance results) lifted the local GNN from **macro-F1 0.66 / AUROC 0.77** to
**macro-F1 0.71 / AUROC 0.84**, with resistant-case recall up from 0.58 to **0.69**. That
AUROC is **at/above the published Stanford work on this same data lineage** (0.74–0.81).
The earlier "0.66 is a hard ceiling" was only a ceiling for the *features we'd tried* — the
literature's #1 predictor, which we'd skipped, broke it. Federation now has real headroom
again; re-running Phase 3 on the stronger model is the next step.

---

## Model results

### Local GNN (all data, "pooled"), core vs + patient-history

| | Core (org/abx rates, demographics, ADI) | **+ Patient-history** |
|---|---|---|
| macro-F1 @0.5 | 0.660 | **0.707** |
| macro-F1 @tuned threshold | 0.667 | **0.712** |
| **AUROC** | 0.771 | **0.836** |
| Resistant-class recall | 0.58 | **0.69** |
| Resistant-class F1 | ~0.48 | **~0.56** |
| Majority baseline | 0.446 | 0.446 |

The +0.045 macro-F1 / +0.065 AUROC jump is far above run-to-run noise (±0.01). It's real
signal, not leakage: the history features are **temporally strict** (each test sees only the
patient's cultures *before* it — unit-tested), and AUROC landing at 0.84 (not ~0.95) is the
tell-tale of genuine signal, exactly matching where the literature lands.

### What "macro-F1 0.71 / AUROC 0.84" means
- **Macro-F1** averages the F1 of the *resistant* and *susceptible* classes equally, so the
  model can't coast on the common class. 0.71 = a solid, balanced result (was "moderate" at
  0.66; now genuinely good).
- **AUROC 0.84** = probability the model ranks a random resistant case above a random
  susceptible one. This is the metric the AMR literature reports, so it's our apples-to-apples
  number — and 0.84 is **competitive with or better than** the personalized-antibiogram papers.

---

## The ceiling — revised

- Old view: ~0.663 (pooled) was a hard ceiling; exhaustive feature/architecture sweeps
  couldn't beat it. **True — but only for the feature families we had tried.**
- New view: the patient's **own prior-resistance history** (untested until now) is the
  literature's top predictor and lifted the pooled model to **~0.71 / AUROC 0.84.**
- So the practical ceiling any federated method targets is now **~0.71**, not 0.66 — and
  because prior-resistance is a strongly *per-patient, hospital-varying* signal, there is now
  more for a topology-aware aggregator to exploit than the earlier analysis suggested.

---

## Federated results (Phase 3) — on the strong patient-history model

Multi-seed α sweep, 5 hospitals (Dirichlet **ward**-mixture), 3 seeds each, patient-history
features on all clients (`phase3-federated` branch, pooled ceiling = 0.71):

| α | local-only | FedAvg best | gain (best − local) | worst-hospital (local) |
|---|---|---|---|---|
| 0.1 | 0.6974 ± 0.0025 | 0.7036 ± 0.0029 | **+0.006 ± 0.005** | ~0.685 |
| 0.5 | 0.6883 ± 0.007  | 0.7007 ± 0.0023 | **+0.012 ± 0.009** | ~0.671 |
| 1.0 | 0.6913 ± 0.0058 | 0.7000 ± 0.0084 | **+0.009 ± 0.011** | ~0.670 |

**What's solid:** FedAvg beat local-only in **all 9 runs** (every α × seed) — a sign test at
p ≈ 0.002, so *federation reliably helps*. FedAvg lands at ~0.70 vs the 0.71 pooled ceiling, so
it recovers ~99% of centralized accuracy **without hospitals sharing patient data**. That is the
clean Phase-3 baseline result.

**The honest problem for the novelty:** the *magnitude* is small and noisy — every gain's error
bar overlaps zero (e.g. α=1.0: one seed +0.025, two seeds ~0.001). More fundamentally, **the whole
spread is tiny: local 0.69 → FedAvg 0.70 → pooled 0.71 — about two points total**, and FedAvg
already eats most of it. That leaves a topology-aware aggregator only ~1 point of headroom above
FedAvg (up to the pooled ceiling). The textbook "smarter aggregation helps most when clients differ
most" is **not** visible here: the gain does not grow as α shrinks, because a **ward**-mixture
split barely changes each hospital's *label* distribution (resistant/susceptible rate) — so the
clients are near-IID in the dimension FedAvg actually struggles with.

**Two responses (both now wired into the run output):**
1. **Report worst-hospital F1, not just the weighted mean.** The mean (~0.69) hides the struggling
   sites (worst hospital ~0.65–0.68). FedAvg's real value is lifting the *underserved* clients —
   the axis a topology-aware method should beat plain FedAvg on. This is likely the **headline
   metric**, not mean F1. (Per-hospital `local → FedAvg` deltas + worst-client now print every run;
   FedAvg-per-client requires the next re-run since the earlier run only logged the aggregate.)
2. **Change the split** so clients are genuinely non-IID in *label* and *graph topology*, not just
   ward mixture — see the next section.

---

## Features: what we use, what we're missing (with sources)

The EHR-based AMR literature (esp. the Stanford/ARMD-lineage
[Corbin et al. 2022](https://www.nature.com/articles/s43856-022-00094-8) and the
[multitask antibiogram](https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciag027/8428387))
consistently finds resistance is driven by **microbial identity + antibiotic-exposure history**,
not demographics ([review](https://www.sciencedirect.com/science/article/pii/S2666991924000198),
[ML-AMR feature study](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0319460&type=printable)).

| Feature | In literature? | Do we use it? |
|---|---|---|
| Organism / species identity | **Top signal** | ✅ (node) |
| Prior resistance history | **#1 predictor (Corbin)** | ✅ (just added) |
| Prior antibiotic prescriptions | **#2 predictor (Corbin)** | ⚠️ tried as an edge (flat); **not yet as temporal count/recency features** |
| Specimen / culture source (urine/blood/resp) | Yes — resistance varies by source | ❌ **we have `culture_description`, not using it** (our EDA: urine 0.18 vs resp 0.29) |
| Ordering mode (inpatient/OP/ER) | Yes (care setting) | ❌ have it, not used |
| Demographics (age, gender) | Minimal influence | ✅ (patient node) — consistent with "didn't help" |
| Comorbidities | Modest | ✅ tried (flat) |
| Labs / vitals | Mixed | ✅ tried (flat) |
| Healthcare exposure (prior hospitalization, nursing home, LOS, device days) | Known strong risk factors | ❌ **not readily in our 10 ARMD tables** |

**Cheap wins we're not yet using (from data we already have):**
1. **Specimen type** — strongest untapped signal; our own EDA shows big resistance differences by source.
2. **Prior antibiotic prescriptions** as temporal features (count + recency) — the literature's #2 predictor.
3. **Ordering mode** and **season/temporal** — minor, cheap.

**Not available to us:** nursing-home status, prior-hospitalization/length-of-stay, device days —
strong risk factors in the literature, but not exposed in the ARMD tables we use. Worth naming as a limitation.

---

## Partition strategies — how to widen the FedAvg-vs-local gap (with sources)

The gap is small because **how we split hospitals doesn't create the kind of non-IID that
federation actually struggles with.** The FL benchmarking literature is explicit that the
accuracy gap is driven mostly by **label-distribution skew**, less by feature/quantity skew.
Our current split (Dirichlet over *wards*) is closest to a feature/quantity split, so it leaves
each hospital's resistant/susceptible balance roughly IID — hence the flat, tiny gain.

**The canonical taxonomy** — [Li, Diao, Chen & He, "Federated Learning on Non-IID Data Silos:
An Experimental Study" (ICDE 2022) / NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)
([paper](https://arxiv.org/pdf/2102.02079)) defines six partition schemes in three families:

| Family | Scheme | What it skews | Fit for us |
|---|---|---|---|
| **Label skew** | `noniid-#label-k` (each client sees only k classes) | class balance — extreme | too extreme for binary R/S |
| **Label skew** | `noniid-labeldir` (Dirichlet over the *label*, param β) | class balance — tunable | **best lever: split on resistant-rate** |
| **Quantity skew** | `iid-diff-quantity` (Dirichlet over sample *counts*) | data volume per client | the "more/smaller hospitals" idea |
| **Feature skew** | noise-based / real-feature | covariate shift | ≈ what our ward-mixture does now |
| **Homogeneous** | `homo` (IID) | nothing | control baseline |
| **Real** | natural attribute in the data | all of the above, realistically | **most defensible for a clinical story** |

**Concretely, the options we have (ranked for this project):**

1. **Label-Dirichlet split (`noniid-labeldir`, β).** Partition so hospitals differ in their
   **resistant-vs-susceptible rate** (or in the organism–antibiotic pairs that carry the label),
   not in ward mix. Small β → strong skew. This is the single highest-leverage change: the
   literature shows FedAvg accuracy can fall tens of points under label-Dir where it barely moves
   under feature skew ([Hsu, Qi & Brown 2019](https://arxiv.org/abs/1909.06335), the origin of the
   α/β-Dirichlet protocol; [Li 2022](https://arxiv.org/pdf/2102.02079)). Widening local↔FedAvg
   widens the room for a topology-aware method.
2. **More, smaller hospitals (quantity skew).** Go from 5 to 10–20 clients so each has less data;
   local-only degrades and aggregation matters more. Cheap — just bump `n_clients`. Best combined
   with (1), not alone (alone it mostly adds variance).
3. **Natural / real split.** Partition by a real attribute already in ARMD — **specimen source**
   (urine/blood/resp; our EDA: urine 0.18 vs resp 0.29 resistant → real label skew *and* different
   subgraph shapes), **ordering mode** (inpatient/ER/OP), or **ADI** (already have the baseline).
   This is the most clinically defensible framing and the way healthcare-FL benchmarks argue splits
   should be done ([FLamby, du Terrail et al. 2022](https://arxiv.org/abs/2210.04620), which shows
   synthetic Dirichlet splits are unrealistic and ships *natural* hospital splits).
4. **Graph-topology split (most aligned with our novelty).** Federated *graph* learning simulates
   clients with **community/structure partitioning (Metis / Louvain)** so subgraphs are
   structurally divergent — the two heterogeneity axes named are *statistical* (label) and
   *topological* (structure) ([OpenFGL benchmark 2024](https://arxiv.org/html/2408.16288v1);
   [FedGraphNN](https://arxiv.org/abs/2104.07145)). Splitting by **organism family** (each hospital
   sees a different bug mix → different tested/grew edges) creates exactly the topological
   heterogeneity a topology-aware aggregator is designed to exploit, and it moves label rates too.

**Recommendation:** keep the ward-mixture α-sweep as the reported *baseline* (it's a legitimate
feature/quantity split and shows FedAvg ≈ pooled), but **add a label-Dirichlet split on
resistant-rate (option 1) and a specimen-source natural split (option 3)** as the settings where
the topology-aware method is evaluated. Report **worst-hospital F1** as the headline. Option 4
(organism-community split) is the strongest tie to the "topology-aware" thesis if we want one
partition that is heterogeneous in structure *and* label.

---

## Where this leaves the project

- **Local model: strong and literature-competitive** (0.71 / 0.84). Phase 1 is comfortably done.
- **Federation headroom reopened** — Phase 3 should be re-run with patient-history features to
  see whether the topology-aware novelty now has room to beat FedAvg.
- **A couple of cheap, literature-backed features remain** (specimen type, prior prescriptions)
  that may lift the number further.

**Open questions for the advisor** are now much more favorable: the system works end-to-end, the
local model matches published performance, and the contribution (topology-aware federated
aggregation) has renewed headroom to demonstrate.

---

*All numbers from the full ARMD dataset (Stanford), 2026-08-06. Local model on branch
`phase1-core-pipeline` (merged to `main`); federated Phase-3 code + strong-model sweep on branch
`phase3-federated`. Drivers `notebooks/03_train_local.ipynb` (local) and `04_federated.ipynb`
(federated). Raw federated sweep log in `res.txt`.*
