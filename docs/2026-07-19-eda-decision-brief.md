# AMR Federated KG — EDA Decision Brief

**Date:** 2026-07-19 · **Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale

Full-data EDA is done. Below are the decisions to lock before we write graph code — each with a recommendation and a short why.

## What we're building

Predict whether an infection will **resist a given antibiotic**. To keep patient data private, we split it into several simulated **"hospitals"** that each train a model locally; a central server blends those models with a **topology-aware** rule (it weights each hospital by how *different* its data is, instead of averaging them equally). Each hospital first turns its data into a **graph**.

## Already locked

- Dataset = **ARMD** (Stanford, one site, 2.24M rows)
- We predict **antibiotic resistance**
- "Hospitals" are **simulated** (one real site) — *how* is Decision C
- Method = **federated + graph + topology-aware blending**

## Decisions at a glance

| | Decision | Recommendation |
|---|---|---|
| A | Which rows to keep | Positive cultures, S/I/R labels only → **1.60M rows** |
| B | What we predict | **Binary:** Resistant+Intermediate vs Susceptible |
| C | How to split "hospitals" | **Dirichlet ward-mixture** (α-swept); plain ward + ADI as baselines |
| D | Nodes | **5:** organism, antibiotic, patient, procedure, comorbidity |
| E | Edges | 3 core + 3 enrichment (procedure, comorbidity, prior-exposure) |
| F | Features | Real signals only — **never one-hot IDs** |

---

### A · Which rows to keep

**Keep** `was_positive = 1` and susceptibility ∈ {S, I, R} → **1.60M rows, 315 organisms, 55 antibiotics**.

**Why:** 632k negative cultures have no bug/drug to test (they appear as fake "Null" values); another ~8k are inconclusive. Pure data hygiene.

### B · What we predict

**Binary: Resistant-or-Intermediate vs Susceptible.**

**Why:** Intermediate is only 3% — too small for its own class, and clinically it's treated as "not reliably effective," so it groups with Resistant. Balance is ~4:1 → we use **class weights** and report **macro-F1**, not accuracy.

### C · How to split into "hospitals" — Dirichlet ward-mixture *(decided)*

The method pays off only if hospitals are **different from each other** (non-IID) yet still **overlap** enough to learn from one another. So instead of making each hospital a single ward, we make each one a **different blend of all wards** — like a real tertiary center (ICU-heavy) vs a community clinic (outpatient-heavy).

**How it works — the Dirichlet(α) dial.** A Dirichlet draw hands each hospital a ward-mix (proportions that add to 1, e.g. ICU 35% / ER 20% / IP 35% / OP 10%). **α controls how different the hospitals are:** small α → very lopsided, very different hospitals (strong non-IID); large α → near-identical. We build 4–6 hospitals, **sweep α** (e.g. 0.1 / 0.5 / 1.0), and assign each **patient to exactly one hospital** (so there's no leakage).

**Why not just one natural axis?** We measured five real splits on full data:

| Split | Hospitals | Difference (non-IID) | Patient overlap |
|---|---|---|---|
| Specimen | 3 | 0.040 | 4.6% |
| Ward | 4 | 0.027 | 13.7% |
| ADI | 4 | 0.015 | 0% |
| Ordering mode | 3 | 0.014 | ~11% |
| Age | 9 | 0.006 | 0% |

None is both strongly non-IID *and* clean. The Dirichlet mixture gives us a **tunable** amount of heterogeneity instead of being stuck with whatever one axis happens to offer.

**Baselines we also report:** plain **ward** (a real, un-engineered partition) and **ADI** (a real but near-IID one). Showing the method wins on the tunable mixture *and* holds up on the real splits answers the "did you engineer the heterogeneity?" question honestly.

*Standard method — Dirichlet / NIID-Bench partitioning ([Hsu et al. 2019](https://arxiv.org/pdf/1909.06335); [Li et al. 2022](https://arxiv.org/abs/2102.02079)).*

### D · What's a node vs a feature

Think of the graph as a map: **nodes** = places many patients pass through; **features** = details written onto them.

| Entity | Call | Why |
|---|---|---|
| Antibiotic (55) | **Node** | one endpoint of the prediction |
| Organism (315) | **Node** (bucket rare tail) | other endpoint; E. coli dominates |
| Patient (283k) | **Node** | holds a person's details; where a new patient plugs in |
| Procedure (6) | **Node** (enrichment) | few + clinically meaningful |
| Comorbidity (515) | **Node** (enrichment) | well-shared (~1,300 patients each) — promoted to a node |
| Antibiotic class (18) | **Via prior-exposure edge** | enters as a patient→antibiotic exposure edge, not its own node |
| Specimen (3) | **Edge feature** | a property of the culture |

**5 node types:** patient, organism, antibiotic, procedure, comorbidity.

*A node and its edge come as a pair — a node does nothing until something connects to it. **Core nodes:** organism, antibiotic, patient. **Enrichment nodes** (each added with its edge, one at a time): procedure, then comorbidity.*

### E · What edges to build

**Core (first):**

1. `organism —tested— antibiotic` — **holds the label**
2. `patient —grew— organism` — carries patient context (ward, labs, vitals, timing)
3. `organism —known_resistant— antibiotic` — historical prior (not the label)

**Enrichment (add one at a time, keep what lifts macro-F1):**

4. `patient —underwent— procedure` — brings the **Procedure** nodes in
5. `patient —has— comorbidity` — brings the **Comorbidity** nodes in
6. `patient —prior_exposure— antibiotic` — recent drug exposure; **reuses the existing antibiotic node** (no new node type)

**Why lean first:** the patient side is sparse (46% of patients have one culture), so extra edges can add noise. Build core → strong score → add each and measure.

### F · What features to use

**Real signals only — no one-hot IDs** (identity features cap a GNN at ~0.49 F1).

- **Resistance ratios** per organism/antibiotic (historical resistance rate)
- **Degrees / counts** (drugs per organism = spectrum; cultures per patient = illness burden)
- **Demographics** (age, gender) + **ADI** (socioeconomic)
- **Labs & vitals** (WBC, lactate, heart rate…) — clean but only ~66–77% coverage, so add a "was it measured?" flag + impute
- **Timing** (how recent a prior exposure/procedure was)

---

## How a real patient gets a prediction — Patient P

**P: 68, diabetic, ventilated, had a fluoroquinolone last month, grew E. coli. Will cipro work?**

- P becomes a **new node**, linked *by name* to existing nodes: E. coli (grew), Mechanical Ventilation (underwent), fluoroquinolone class (prior exposure), with P's age/labs attached. (Matching is a simple ID lookup, not fuzzy matching.)
- The model (GraphSAGE — *inductive*, so brand-new patients work without retraining) blends P's neighbourhood and predicts resistance for **each antibiotic**, personalised to P.
- Susceptibility is recorded **per culture**, so two E. coli patients can get different answers — the prediction is really about **(patient, organism, antibiotic)**, and P's node is what makes it *P's* answer.

*This is the **organism-known** setup (a culture identified the bug). Walk-in "no culture yet" prediction is out of scope.*

---

## Still open / next steps

1. Team confirms Decisions A–F (especially **C**, the split).
2. Encode choices into `config.py` / `CLAUDE.md`.
3. Two build-time checks: patient→one-hospital assignment (Dirichlet, no leakage); labs/vitals imputation.
4. Write `data_loader.py` → `graph_build.py` (core 3 edges first).
5. Get one hospital to a strong macro-F1 **before** going federated.

**Data facts worth remembering:** target core is small & dense (3,848 real organism–antibiotic pairs of 17,325); comorbidity table is 206M rows (must aggregate first); coverage — comorbidity 98%, vitals 77%, labs 66%, procedures 33%.

*All numbers from the full ARMD dataset, 2026-07-19.*
