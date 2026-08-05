# Federation Results & Status (Phases 2–3)

**AMR Federated Knowledge-Graph Capstone** · 2026-08-06
**Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale

---

## TL;DR

We built and validated the full federated pipeline — partitioning into simulated
hospitals, a FedAvg baseline, and a multi-seed heterogeneity sweep. **Federation
works** (hospitals collaborating beat each going it alone by a steady ~+0.03 macro-F1,
without sharing patient data). But the gain is **flat across heterogeneity** and
**FedAvg already sits at the pooled ceiling (~0.663)** — so the planned topology-aware
aggregation has **limited headroom** to show a clear numerical win. This document
records the numbers and the decision that follows.

---

## What "macro-F1 = 0.663" means, and is it good?

**F1** balances *precision* (of what the model flagged as resistant, how much really
was) and *recall* (of the truly resistant cases, how many it caught). **Macro-F1**
averages the F1 of both classes *equally*, so a model can't score well just by
predicting the common "susceptible" class — it's the honest metric for imbalanced data.

Our 0.663 is the average of:
- **Susceptible** class F1 ≈ 0.84 (does well)
- **Resistant** class F1 ≈ 0.49 (recall ≈ 0.59 — misses ~40% of resistant cases)

**Verdict — moderate, not strong.** Against the 0.446 majority-class baseline it's a
clear, meaningful improvement (+0.21) that proves the graph approach learns real signal.
In absolute terms it's a respectable-but-not-stellar number: useful and clinically
*suggestive*, but not a deployable classifier on its own. Antibiotic-resistance
prediction is genuinely hard, and our Phase-1 analysis showed the signal is driven by
*which organism and which antibiotic* — individual-patient context adds nothing — so
~0.66 is likely close to the intrinsic ceiling of this task and data.

---

## The ceiling: why ~0.663 is the practical maximum

- **Pooled** (one model on all data) = 0.663 is the natural upper reference for
  federated learning: federation tries to *approach* it without sharing data.
- Federation almost never *exceeds* pooling — pooling has strictly more information in
  one place. (Model averaging can occasionally act like an ensemble and edge slightly
  above pooled, but it's rare and small.)
- So any aggregation method — including the topology-aware one — realistically tops out
  at **≈ 0.663**, which also appears to be near the task's intrinsic ceiling.

This is the crux: there is very little room between **"each hospital alone" (~0.63)** and
**"best possible" (~0.66)** for a smarter aggregator to fill.

---

## Results

### Phase 1 — local model (all data, no federation)
| | macro-F1 |
|---|---|
| Majority baseline | 0.446 |
| **Local model (pooled)** | **0.663** |

Exhaustively confirmed as a ceiling: 48-model grid (all feature combos × architectures)
never beat ~0.66.

### Phase 3 — FedAvg vs local-only, multi-seed α sweep (5 hospitals, 3 seeds each)

| α (heterogeneity) | local-only | FedAvg **best** | FedAvg final | **gain** (best − local) |
|---|---|---|---|---|
| 0.1 (most different) | 0.624 ± 0.018 | 0.657 ± 0.005 | 0.638 ± 0.007 | **+0.033 ± 0.023** |
| 0.5 | 0.624 ± 0.003 | 0.657 ± 0.008 | 0.631 ± 0.010 | **+0.032 ± 0.006** |
| 1.0 (most alike) | 0.631 ± 0.008 | 0.661 ± 0.009 | 0.627 ± 0.006 | **+0.030 ± 0.015** |

**Reading it:**
1. FedAvg-best beats local-only by a steady **~+0.03** at every heterogeneity level.
2. FedAvg-best (~0.66) lands **just under pooled (0.663)** — federation recovers most of
   the go-it-alone gap without sharing data.
3. The gain is **flat across α** — it does *not* grow when hospitals differ more. (An
   earlier single-run result suggesting it did was noise; multi-seed averaging removed it.)
4. A consistent **~0.02–0.03 end-of-training drift** (best → final) appears at all α.

---

## The decision point

The topology-aware aggregation (our planned novelty) weights hospitals by how *different*
their graphs are. But the sweep shows the amount of difference (α) barely changes how much
federation helps — so weighting by difference may not help much either, and FedAvg is
already at the ceiling. **Headroom for a clear win is small.**

**Questions for the advisor:**
1. Is a working KG + per-patient federated system on real AMR data, with an *honest
   characterization* of when topology-aware aggregation helps, sufficient — or does the
   thesis need a clear numerical win over FedAvg?
2. Is it acceptable to **engineer a harder regime** (more & smaller hospitals, or a
   specimen-based split) to give the topology-aware method room to matter?
3. Would a **stability/robustness** framing (curbing FedAvg's end-of-training drift) count
   as the contribution, rather than peak accuracy?
4. Is a **nuanced/negative result** ("topology-aware gave marginal gains here, and here's
   why") an acceptable capstone outcome?

**Cheap experiment that could open headroom (not yet run):** rerun with 15–20 smaller
hospitals. Data-starving each hospital should lower local-only and widen the federation
gain — giving the novelty room to matter. One-argument change (~15 min).

---

*All numbers from the full ARMD dataset (Stanford), 2026-08-06. Code: branch
`phase1-core-pipeline`; federated driver `notebooks/04_federated.ipynb`. Raw sweep log
in `res.txt`.*
