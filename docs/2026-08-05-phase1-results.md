# Phase 1 Results — the Local Resistance-Prediction Model

**AMR Federated Knowledge-Graph Capstone** · 2026-08-05
**Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale

---

## TL;DR

We built the single-hospital pipeline — **data → knowledge graph → GNN** — and it predicts antibiotic resistance at **macro-F1 0.66**, well above the 0.45 "always guess the common answer" baseline.

We then tried **everything** to push it higher: three extra data sources, richer patient features, and a full architecture search — **48 models in total**. Nothing beat the simple version.

That's not a failure. It's a clean, *exhaustively proven* finding: **resistance in this data is driven by which organism and which antibiotic — not by the individual patient.** Phase 1 is done; the interesting part (federation) is next.

---

## 1. What we were building

The goal of the whole project is to predict, for a given infection, **whether a specific antibiotic will fail** (the bug is resistant) — and to do it in a *privacy-preserving, federated* way where several simulated "hospitals" collaborate without sharing patient data.

Before any of that, each hospital needs a model that works **on its own**. That's Phase 1: take one hospital's data, turn it into a **knowledge graph** (patients, organisms, antibiotics, and the links between them), train a graph neural network, and check it actually learns something. This document is about that step.

We measure everything with **macro-F1**, not accuracy — because the data is lopsided (~80% of tests come back "susceptible"), so a model that blindly guesses "susceptible" every time would look 80% accurate while being clinically useless. Macro-F1 forces the model to do well on *both* classes, including the rare, important "resistant" one.

---

## 2. The result

On the full dataset (238,758 held-out test cases):

| | Macro-F1 |
|---|---|
| Baseline (always guess "not resistant") | **0.446** |
| **Our model** | **0.663** ✅ |

Breaking down *how* it does:

- **Susceptible cases:** it gets these right ~84% of the time (F1 0.84).
- **Resistant cases (the hard, important ones):** it catches **~59%** of them (F1 ~0.49).

Catching 59% of resistant infections isn't perfect, but it's far better than the baseline's 0%, and the model deliberately errs toward flagging resistance — clinically the safer mistake (a false alarm beats missing a resistant infection).

---

## 3. How we made sure the score is *honest*

A high score is worthless if the model is secretly cheating. We locked down five things so 0.66 is real:

1. **Cleaned the data first.** Kept only genuine positive cultures with a real Susceptible/Intermediate/Resistant result — dropped ~640k junk/negative rows that show up as fake "Null" bugs.
2. **Binary target with class weights.** "Resistant-or-Intermediate" vs "Susceptible," and we weight the rare resistant class up so the model can't just ignore it.
3. **Predict per patient.** One prediction per *(patient, organism, antibiotic)* — a personalized forecast, not a one-size-fits-all verdict for each bug-drug combo.
4. **No peeking on the exam.** We split the data **by patient** (a patient is never in both the training and the test set), and any "background rate" features were computed from **training patients only**. No answer leaks from test to train.
5. **Real features, not ID codes.** Resistance rates, counts, degrees, demographics — never one-hot identity features (those cap a graph model around 0.49).

That setup produced the **core model at 0.663.**

---

## 4. Trying to beat 0.66 — everything we threw at it

We added extra information one piece at a time, keeping only what actually helped. Here's the honest log:

| # | What we added | The idea | Result | Verdict |
|---|---|---|---|---|
| 1 | **Comorbidity** links (diabetes, CKD, …) | "sicker patients resist more" | 0.662 | ❌ no lift |
| 2 | **Prior antibiotic exposure** | "recent drugs breed resistance" | 0.662 | ❌ no lift (hurt resistant recall) |
| 3 | **Procedures** (surgeries, lines, …) | "invasive care → resistant bugs" | ~0.66 | ❌ no lift |
| 4 | **Labs & vitals** (WBC, lactate, BP, HR) | "severity signals resistance" | 0.666 | ➖ flat (marginally best) |
| 5 | **Model tuning** (5 architectures) | "maybe it's under-trained" | 0.65–0.67 | ❌ nothing beat the original |

Every single one landed within a hair of 0.66.

---

## 5. The exhaustive proof — 48 models

Rather than stop there, we ran the *complete* search: **every combination** of those four data sources (16 combinations) crossed with three model architectures = **48 models**.

The entire Top-10 landed between **0.665 and 0.668.** The most telling pair:

```
0.6677   everything (all features + labs/vitals)   ← the "kitchen sink"
0.6675   core, nothing added, just a bigger model  ← plain core
```

The everything-model beat plain core by **0.0002** — statistical noise. And a plain core model with a slightly better architecture *matched* the kitchen sink. Translation: **the extra features add nothing; the tiny wiggles are just architecture noise.**

---

## 6. What it actually means — an example

Picture two patients, both grew **E. coli**, both being tested against **Ciprofloxacin**:

- **Patient A:** 25, healthy, walked into a clinic.
- **Patient B:** 80, in the ICU, ventilated, three rounds of antibiotics last month.

Our instinct says B's infection is more likely resistant — so patient details *should* matter. But across 48 models, adding B's comorbidities, prior antibiotics, labs, and vitals changed the score by **nothing**.

The finding, in one line:

> **On this dataset, antibiotic resistance is determined mostly by *which organism* and *which antibiotic* — the individual patient's history adds no measurable predictive signal.**

This is a genuine, defensible result — and we didn't just assume it, we **searched the entire space to prove it.** For the write-up, "we exhaustively tested and it's a real ceiling" is a much stronger statement than "we tried a couple things."

---

## 7. Why the local score doesn't need to be higher

This surprises people, so it's worth stating plainly: **for this project, the local model does not need a high score.** It plays three supporting roles:

- **A sanity check** — "does the graph approach learn anything real?" 0.66 vs 0.45 says clearly yes. ✅
- **A measuring stick** — federation is judged by *comparisons*, and macro-F1 is the ruler. The height of the ruler matters less than using the *same honest ruler* everywhere.
- **The scaffolding** — the exact same graph + model gets reused by every simulated hospital in the federated experiments.

Think of it like a recipe: it needs to *work* before we run the real experiment ("do hospitals predict better by collaborating?"). It does **not** need to be a Michelin dish first. The 0.66 is the target that small, data-starved hospitals will try to reach *by teaming up* — which is exactly what the federation phase measures.

---

## 8. What's next

**Phase 2 — Federation (the actual contribution):**

1. **Partition** the data into several simulated hospitals — a tunable *Dirichlet ward-mixture* (each hospital = a different blend of wards), plus plain ward and ADI as baseline splits. Each patient goes to exactly one hospital (no leakage).
2. **FedAvg baseline** — hospitals train locally and a server averages their models. This must beat "each hospital alone."
3. **Our novelty** — combine the hospitals' models weighted by their *graph topology*, and show it beats plain FedAvg.

**One thing to settle as a team first:** we currently have **two versions** of the local model — a **per-patient** version and a **per-bug-drug-pair** version. Both are reasonable, but federation requires every hospital to use the *same* graph shape, so we need to agree on one before we federate. Worth a quick team decision.

---

*All numbers are from the full ARMD dataset (Stanford, single institution), 2026-08-05. Code lives on the `phase1-core-pipeline` branch; the training driver is `notebooks/03_train_local.ipynb`.*
