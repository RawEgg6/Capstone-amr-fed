# Next Steps — Build the Single-Hospital Pipeline

**Date:** 2026-07-19 · **Team:** Vikram, Saffiya, Sanjana, Arshia · **Guide:** Dr. Swati Jagdale

The EDA and all design decisions are done. This is the build phase. **Goal of this round: each of you builds the data → graph → local-model pipeline yourselves and gets one hospital training with a decent macro-F1.** Federation comes *after* that works — don't jump ahead.

---

## Read these first (in order)

1. **[Decision brief](2026-07-19-eda-decision-brief.md)** — *the* reference. Every choice (rows, target, split, nodes, edges, features) with the reasoning. If you read one thing, read this.
2. **[CLAUDE.md](../CLAUDE.md)** — the locked schema + build order + repo conventions, in short form.
3. **[Dataset.md](../Dataset.md)** — what each ARMD table and column means.
4. **[README.md](../README.md)** — environment setup and how we share code (GitHub) vs data (Drive).
5. **[notebooks/02_graph_eda.ipynb](../notebooks/02_graph_eda.ipynb)** — the EDA all these numbers come from; re-run it if a decision feels unclear.

Everything below is a summary — the brief is the source of truth if anything conflicts.

---

## The spec, in one place (so you don't have to hunt)

**Keep only these rows:** `was_positive == 1` **and** `susceptibility ∈ {Susceptible, Intermediate, Resistant}` → ~1.60M rows, 315 organisms, 55 antibiotics. Everything else (negative cultures = fake "Null" organism/antibiotic; Inconclusive/Synergism) is dropped.

**Target (label):** **binary** — `Resistant OR Intermediate = 1`, `Susceptible = 0`. It sits on the `(organism, tested, antibiotic)` edge. Use **class weights** (~4:1 imbalance) and report **macro-F1**, never accuracy.

**5 node types:** patient, organism, antibiotic, procedure, comorbidity.

**6 edges:**

- Core (build first): `(organism, tested, antibiotic)` [label], `(patient, grew, organism)`, `(organism, known_resistant, antibiotic)`
- Enrichment (add one at a time, keep what lifts macro-F1): `(patient, underwent, procedure)`, `(patient, has, comorbidity)`, `(patient, prior_exposure, antibiotic)`

**Features:** real signals only — resistance ratios, degrees/counts, demographics (age, gender), ADI, labs/vitals summary stats (+ a "was it measured?" flag), timing. **Never one-hot identity features** (they cap the model at ~0.49 F1).

**Reproducibility:** seed everything from `config.SEED`; keep the model architecture identical across clients.

---

## Your tasks this round

Build these as modules under `src/amr_fed/` (notebooks only import + call — no logic in notebooks). `config.py` already holds the schema constants; import from it, don't hardcode.

### 1. `data_loader.py`
Load and join the 10 tables on the 4 keys (`anon_id`, `pat_enc_csn_id_coded`, `order_proc_id_coded`, `order_time_jittered_utc`); apply the row filter and binary label above; select only the columns you need (labs/vitals are wide — don't load all 59/28).

**Done when:** it returns a clean per-culture frame with the binary label and joined patient context, row counts match the brief (~1.60M), and no "Null" organism/antibiotic rows survive.

### 2. `graph_build.py`
Turn one hospital's frame into a PyG `HeteroData` with the **core 3 edges** first and **real (non-identity) features**. Add the enrichment edges only after the core works.

**Done when:** the graph loads, node/edge counts are sane (target core ≈ 3,848 organism–antibiotic pairs), and features are ratios/counts/clinical values — not one-hot IDs.

### 3. `model.py` + `train_local.py`
Heterogeneous GraphSAGE ("AMR-SAGE"), class-weighted loss, train on **one** hospital, evaluate with macro-F1.

**Done when:** a single local model trains end-to-end and reports a macro-F1 clearly above a majority-class baseline (~0.45). Beating that consistently is the target for this round.

*(Suggested division of labour: pair up on `data_loader` + `graph_build` since they're tightly coupled, and the other pair on `model` + `train_local`. Compare results — everyone should be able to reproduce the same macro-F1 from the same seed.)*

---

## Gotchas the EDA already found (save yourself the pain)

- **Drop negatives first.** ~632k rows are negative cultures with a fake "Null" organism/antibiotic. Filter `was_positive==1` before *any* node/edge counting.
- **Patients are sparse.** 46% of patients have exactly one culture — patient nodes are low-degree, so lean on patient *features/edges*, not patient-to-patient structure.
- **Labs/vitals are clean but partial.** 0% null where present, but only ~66% (labs) / ~77% (vitals) of cultures have them → add a "measured?" indicator + impute; don't drop those rows.
- **Comorbidity table is huge** (206M rows, ~282 per culture) — aggregate to a per-culture summary before building edges, or it'll blow up memory.
- **Coverage varies:** comorbidity ~98%, vitals ~77%, labs ~66%, procedures ~33%. Missing context is normal — the model must tolerate it.

---

## Definition of done for this round

One hospital's data → heterogeneous graph → local AMR-SAGE → **macro-F1 clearly beating the majority baseline**, reproducible from `config.SEED`, with all logic in `src/amr_fed/` modules. **Do not start partitioning or federation until this holds** — that's the golden rule from CLAUDE.md.

## What comes after (not now)

Dirichlet ward-mixture partitioning → FedAvg baseline (must beat local-only) → topology profiling → the topology-aware aggregation (our novelty) → evaluation vs FedAvg/FedProx/FedGTA. See the build order in [CLAUDE.md](../CLAUDE.md).

---

## Appendix — How to prompt an AI assistant (no Claude Code needed)

A normal chatbot (ChatGPT, Claude.ai, Gemini, etc.) **can't see this repo**, so you have to give it the context, and **you** run the code it writes. Workflow: ask → paste the code into your editor → run it → paste back any error → repeat.

**Step 1 — give it context.** If your tool allows file uploads, attach this doc + the decision brief + `Dataset.md` + `src/amr_fed/config.py` and say "read these." If it doesn't, paste this block first:

```
PROJECT: Federated, graph-based prediction of antibiotic resistance on the ARMD dataset (Stanford EHR). We build a heterogeneous knowledge graph per hospital and train a GNN. I'm building the SINGLE-hospital pipeline now (data → graph → local model). No federation yet.

DATA: 10 ARMD CSVs. Join keys: anon_id (patient), pat_enc_csn_id_coded (encounter), order_proc_id_coded (culture), order_time_jittered_utc (time). Main table cohort has: organism, antibiotic, susceptibility, was_positive, culture_description, ordering_mode. Others: demographics (age, gender), adi (adi_score), comorbidity (comorbidity_component), priorprocedures (procedure_description), ward (hosp_ward_ICU/ER/IP/OP), labs + vitals (wide, prefixes median_/Q25_/Q75_/first_/last_), antibiotic_class_exposure (antibiotic_class), microbial_resistance (organism, antibiotic).

ROW FILTER: keep was_positive==1 AND susceptibility in {Susceptible, Intermediate, Resistant} → ~1.60M rows, 315 organisms, 55 antibiotics. Drop everything else (negatives show up as a fake "Null" organism/antibiotic).

TARGET: binary label on the (organism, tested, antibiotic) edge — Resistant OR Intermediate = 1, Susceptible = 0. ~4:1 imbalance → use class weights, report MACRO-F1, not accuracy.

GRAPH — 5 node types: patient, organism, antibiotic, procedure, comorbidity.
6 edges: CORE = (organism,tested,antibiotic)[label], (patient,grew,organism), (organism,known_resistant,antibiotic); ENRICHMENT (add later) = (patient,underwent,procedure), (patient,has,comorbidity), (patient,prior_exposure,antibiotic).

FEATURES: real signals only — resistance ratios, node degrees/counts, demographics, ADI, labs/vitals summary stats + a "was it measured?" flag, timing recency. NEVER one-hot identity features (they cap the model ~0.49 F1).

STACK: Python 3.11, pandas, PyTorch (CPU is fine), torch_geometric (PyG). Model = heterogeneous GraphSAGE via HeteroConv over SAGEConv per edge type.

GOTCHAS: filter the "Null" negative-culture rows first. 46% of patients have only 1 culture (patient nodes are low-degree). Labs/vitals only ~66–77% coverage (impute + add a measured-flag). Comorbidity table is 206M rows — aggregate per culture before building edges.
```

**Step 2 — then paste one task prompt at a time (in order):**

*Data loader:*
```
Using the context above, write data_loader.py: load the 10 tables, join on the 4 keys, apply the row filter, add the binary label, select only needed columns (labs/vitals are wide). Return one clean per-culture dataframe. Print row count, #organisms, #antibiotics as a self-check (expect ~1.60M / 315 / 55). Give me the full file to paste in and run.
```

*Graph builder (after the loader runs):*
```
Using the context above, write graph_build.py: turn the loader's dataframe into a PyG HeteroData with the 5 node types and the 3 CORE edges only. Features must be real signals, never one-hot IDs. Aggregate comorbidity per culture. Print node/edge counts (target core ≈ 3,848 organism-antibiotic pairs). Full file, ready to run.
```

*Model + training (after the graph builds):*
```
Using the context above, write model.py (heterogeneous GraphSAGE, "AMR-SAGE") and train_local.py: binary edge classification on (organism,tested,antibiotic) with class weights; train on one hospital; report MACRO-F1 and a confusion matrix, plus the majority-class baseline for comparison. Full files, ready to run.
```

**Step 3 — iterate.** Run each file locally; if it errors, paste the full traceback back to the AI. If the AI drifts (one-hot features, 3-class target, reports accuracy), correct it: "target is binary, features must be real signals, report macro-F1."

**Before any of this:** set up the environment once — create the venv, `pip install -r requirements.txt`, `pip install -e .`, and set `ARMD_DIR` to your ARMD folder (see [README.md](../README.md)). The AI can't do this for you.

---

*Stuck? The [decision brief](2026-07-19-eda-decision-brief.md) explains the "why" behind every choice here, and [02_graph_eda.ipynb](../notebooks/02_graph_eda.ipynb) has the numbers.*
