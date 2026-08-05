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

## Federated results (Phase 3) — on the EARLIER 0.66 model, to be re-run

Multi-seed α sweep, 5 hospitals, 3 seeds each (this predates the patient-history fix):

| α | local-only | FedAvg best | gain (best − local) |
|---|---|---|---|
| 0.1 | 0.624 ± 0.018 | 0.657 ± 0.005 | +0.033 ± 0.023 |
| 0.5 | 0.624 ± 0.003 | 0.657 ± 0.008 | +0.032 ± 0.006 |
| 1.0 | 0.631 ± 0.008 | 0.661 ± 0.009 | +0.030 ± 0.015 |

Findings **on that weaker model**: FedAvg beat local-only by a steady ~+0.03 (federation
works), flat across heterogeneity, with a ~0.02–0.03 end-of-training drift. **These need
re-running with patient-history features**, since the pooled ceiling has moved up and the
per-patient signal may change the heterogeneity story.

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

*All numbers from the full ARMD dataset (Stanford), 2026-08-06. Code: branch
`phase1-core-pipeline`; drivers `notebooks/03_train_local.ipynb` (local) and
`04_federated.ipynb` (federated). Raw federated sweep log in `res.txt`.*
