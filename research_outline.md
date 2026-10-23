# IEEE Conference Paper Outline — Topology-Aware Federated Learning for Antimicrobial-Resistance Prediction

**Status:** outline only — no paper text written yet. All project-specific statements below
are grounded in the repository (`src/amr_fed/`, `tests/`, `docs/`, `notebooks/`); anything
not evidenced is explicitly labeled **UNKNOWN** or **UNVERIFIED**.

---

## A. Project understanding

### A.1 Problem and motivation
Antimicrobial resistance (AMR) is a major global health threat, and predicting whether a
patient's infection will resist a given antibiotic can guide empiric therapy. Patient data
cannot leave hospitals, so the project studies **privacy-preserving federated learning**:
several simulated "hospitals" each train a local graph neural network on their own
patients, and a central server aggregates only model weights (plus a small topology
fingerprint) — never patient rows.

### A.2 System architecture (as built)
```
ARMD CSVs (Drive, never committed)
  → config.py            locked schema: 5 node types, 6 edge types, binary label, data path
  → data_loader.py       10/16 tables joined on 4 keys → 1.60M-row per-test frame, label {0,1}
  → graph_build.py       frame → PyG HeteroData (core 3 node types in federation runs)
  → model.py             AMR-SAGE: HeteroConv of SAGEConv (hidden 64/128, 2 layers)
                         + MLP decoder over (patient, organism, antibiotic) + per-test features
  → train_local.py       class-weighted BCE; reports macro-F1, AUROC, threshold tuning
  → partition.py         one dataset → simulated hospitals (8+ split functions)
  → federated/           Flower simulation: client_app / server_app (FedAvg) / task.py
                         (graphs via disk, seeded RNGs) / strategy.py (topology-aware)
  → topology.py          14-feature hospital fingerprint + similarity + mixing weights
  → tests/               60+ offline checks (37 pipeline + 17 topology + strategy/seeding)
```

### A.3 Data (measured, not assumed)
- **Source:** ARMD, Stanford Healthcare EHR, de-identified (repo `Dataset.md` cites
  Nateghi Haredasht et al., arXiv:2503.07664; data via shared Drive, 10 of 16 CSVs used).
- **Scale after filtering** (`was_positive=1`, S/I/R only): **1,601,980 rows, 67,008
  patients, label rate 0.1952** (measured in notebook 06 run).
- **Target:** binary — Resistant+Intermediate (1) vs Susceptible (0) — on the
  `(organism, tested, antibiotic)` edge; supervision is per-test triples
  `(patient, organism, antibiotic)` (~1.6M) over only ~3,848 unique bug–drug pairs.
- **Splits:** patient-level train/val/test (70/15/15, no leakage); hospitals simulated by
  partitioning patients (Dirichlet ward-mixture α-sweep + ward/ADI baselines +
  label-Dirichlet + specimen + organism-community + 4 hard topology splits:
  homophily, degree-skew, topology-corners, Louvain).
- **Key features:** resistance/susceptibility ratios, counts, degrees, demographics, ADI,
  and patient-history prior-resistance features (temporally strict, leakage-tested). No
  one-hot identity features (caps a GNN at ~0.49 F1 per repo docs).

### A.4 Training, evaluation, baselines
- **Local model:** full-batch heterogeneous GraphSAGE, Adam (lr 1e-3, wd 1e-4),
  60 epochs (Phase 1) / matched budgets in federation; class-weighted loss.
- **Federation:** Flower `run_simulation` (Ray), FedAvg baseline; matched-budget
  comparisons (local-only epochs = rounds × local_epochs); seeds 42/43/44; deterministic
  client/server seeding (commit `0af90d0`).
- **Three-way comparison per split:** pooled (centralized, same per-hospital protocol) vs
  local-only vs federated (FedAvg or topology-aware).
- **Metrics:** macro-F1 (headline), AUROC, per-hospital F1, worst-hospital F1 (fairness
  headline), best-round + final-round reporting.
- **Topology-aware arm:** `TopologyAwareStrategy(FedAvg)` — fingerprint-similarity mixing
  (distinctiveness mode, T=1.0 tested); Path-B gate: beat FedAvg-best by ≥ 0.005.

### A.5 Implementation facts
- Stack: Python 3.11, torch + torch-geometric, `flwr[simulation]`, pandas/numpy/scipy,
  scikit-learn, networkx, matplotlib (per `requirements.txt`).
- Hardware: Colab GPU (T4/P100, high-RAM) for training/simulation; topology module is
  pure pandas/numpy (CPU-only).
- Evidence artifacts: `res.txt` (committed Colab logs), `docs/` (9 writeups with real
  tables), notebooks 00–07 (01/02 EDA, 03 local, 04 federated, 05 demo, 06 fingerprints,
  07 comparison), `tests/` (6 files).
- Team (per docs): Vikram, Saffiya, Sanjana, Arshia; Guide: Dr. Swati Jagdale.

---

## B. Potential research contribution and novelty assessment

### B.1 What the project demonstrably accomplishes
1. An end-to-end, tested, reproducible pipeline for federated AMR prediction on real EHR
   data (60+ unit tests; deterministic reruns verified to ~1e-4).
2. A strong local model (macro-F1 0.71 / AUROC 0.84, at/above published Stanford-lineage
   numbers 0.74–0.81) via leakage-safe patient-history features.
3. A clean FedAvg baseline: recovers ~99% of pooled accuracy without sharing data;
   biggest gains on the worst hospitals (organism-community worst +0.037).
4. A working topology-aware aggregator that **passed its pre-registered gate on 2 splits**
   (worst-hospital +0.0054 organism-community, +0.0050 specimen; mean +0.001) with a
   secondary stability finding (tighter seed-variance, less round-decay).

### B.2 Candidate novel elements (require prior-work verification — see §I)
1. **Fingerprint-similarity aggregation with distinctiveness weighting for worst-hospital
   fairness** in cross-silo FL: up-weighting the *odd* hospital rather than the average
   one, motivated by an observed failure mode rather than mean accuracy.
2. **Empirical reframing of "pooled as ceiling":** on topology-divergent splits, plain
   FedAvg beats size-weighted centralized training because pooled under-trains rare
   communities (mechanism documented with per-hospital numbers, e.g. Louvain tiny
   community: local 0.7423 → pooled 0.6735 → FedAvg 0.7308).
3. **The "shared-backbone" argument:** in subgraph-FL where clients share one
   entity backbone, classic structural statistics are near-identical across clients, so
   fingerprints must emphasize feature/label/topology-distribution signals — a design
   principle with evidence (density 0.19–0.24 everywhere; discriminative power from
   history/rate/homophily/hubness).
4. Honest negative results as findings (label-Dirichlet dead end; gate failing in the
   informative direction).

### B.3 Engineering vs science — be explicit in the paper
- Engineering (not claims of novelty): the Flower harness, data plumbing, seeding fix,
  notebooks, tests.
- Scientific claims (need lit review + statistics): §B.2 items above. **Do not present
  the system build itself as the contribution.**

### B.4 Claims supported vs unverified
- **Supported:** pipeline works; local 0.71/0.84; FedAvg≈pooled; specimen/organism wins;
  gate FAIL direction with mechanism; topology arm passes Path-B gate on 2 splits × 3
  seeds; stability observation; determinism to ~1e-4.
- **UNVERIFIED / weak:** statistical significance (n=3 seeds; effect ≈ gate threshold ≪
  std ~0.016–0.021); only 2 splits tested for Phase 5; no FedProx/FedGTA head-to-head; no
  ablations; temperature/mode not swept live (only T=1.0 distinctiveness run); T=inf
  equivalence asserted by unit test, not shown empirically; single dataset; simulated
  (not real) federation; best-round selection inflates both arms; no cost analysis.

### B.5 What would strengthen the paper (prioritized)
1. Seeds 42–46 (5 seeds) + paired per-seed tests / bootstrap CIs on the worst-hospital delta.
2. Temperature × mode sweep (consensus vs distinctiveness × T∈{1,2,5,∞}) on both splits.
3. FedProx and FedGTA-style baselines under identical protocol.
4. Ablations: fingerprint without history; without homophily/hubness; size-only weights.
5. Final-round (not just best-round) comparison table.
6. Louvain evaluation via pairwise/clustering logic (centroid weighting shown insufficient).
7. Compute/communication cost and fingerprint privacy analysis.

---

## C. Three candidate paper titles

1. **Topology-Aware Federated Aggregation for Antimicrobial-Resistance Prediction: Weighting the Odd Hospital Out** — most accurate to the actual mechanism and finding.
2. **Worst-Hospital Fairness in Federated Graph Learning for AMR Prediction via Fingerprint-Similarity Aggregation** — foregrounds the fairness framing and metric.
3. **When Federated Averaging Beats Centralized Training: Topology-Divergent Hospitals and Similarity-Weighted Aggregation for AMR Knowledge Graphs** — foregrounds the counterintuitive pooled-vs-FedAvg finding; slightly longer, best for a venue that values empirical insight.

---

## D. Detailed section-by-section outline

### 1. Abstract
- **Purpose:** 150–200 words: problem (AMR + privacy), method (heterogeneous KG + federated GNN + fingerprint-similarity aggregation), key results with numbers, conclusion.
- **Must include (from evidence only):** ARMD 1.6M tests; local 0.71 macro-F1 / 0.84 AUROC; FedAvg ≈99% of pooled; organism-community FedAvg 0.719 (+0.037 worst); the pooled-under-trains-rare-communities finding; topology-aware worst-hospital gains +0.0054 / +0.0050 with the fragility caveat (std ≫ effect, n=3).
- **Missing:** final numbers after the §B.5 experiments; venue page limit.
- **Do not invent:** significance claims, second dataset, clinical deployment.

### 2. Index Terms
- Proposed: federated learning; graph neural networks; antimicrobial resistance; heterogeneous graphs; fairness; non-IID data; model aggregation.
- **Missing:** venue-specific keyword taxonomy check.

### 3. Introduction
- **Purpose:** motivate AMR + data-privacy conflict; state why federation + graphs; preview the counterintuitive finding and the contribution.
- **Points:** AMR burden and empiric-therapy need (cite clinical sources — NEEDED); hospitals can't share EHR (cite FL/healthcare surveys — NEEDED); GNNs fit relational EHR but prior work is centralized (cite AMR-ML literature incl. Corbin et al. 2022 — VERIFY details); research gap: aggregation rules ignore client graph structure and optimize means, not underserved sites.
- **Objectives/questions:** (a) can FedAvg match centralized without sharing data? (b) where does averaging hurt a hospital? (c) does similarity-weighted aggregation rescue the worst hospital?
- **Contributions:** list §B.1–B.2, clearly separating system from science.
- **Evidence:** `docs/2026-07-19-eda-decision-brief.md`, `Dataset.md`.
- **Missing:** clinical-motivation citations; exact venue framing.

### 4. Related Work
- **Purpose:** place the work; establish what is/isn't new.
- **Subsections:** (a) ML for AMR prediction (EHR/GNN/antibiogram — NEEDED); (b) federated learning basics: FedAvg, FedProx, clustered/personalized FL — NEEDED; (c) federated *graph* learning: FedGTA, AdaFGL, FedSSA, FedStar, GCFL, FedSA-GCL, FedDAG, benchmarks (OpenFGL, FedGraphNN) — CANDIDATE refs from prior web research, all NEED verification; (d) fairness in FL (worst-client metrics — NEEDED); (e) positioning table: method vs ours on (similarity signal, weighting rule, fairness target).
- **Missing:** full literature search — this section cannot be drafted honestly yet (see §I).

### 5. Proposed Methodology / System Design
- **Purpose:** everything needed to reimplement.
- **5.1 Problem formulation:** binary edge classification on `(organism, tested, antibiotic)`; supervision triples `(patient, organism, antibiotic)`; macro-F1 + AUROC + worst-hospital metrics. Evidence: `config.py`, `docs/2026-07-19-eda-decision-brief.md`.
- **5.2 Data and preprocessing:** ARMD tables used, join keys, row filter → 1.60M/67k/rate 0.1952, binary label map, patient-level splits. Evidence: `data_loader.py`, `tests/test_data_loader.py`, notebook 06 dataset printout. Missing: none (cite dataset paper).
- **5.3 Heterogeneous graph:** node/edge types, core-vs-enrichment, feature families + leakage-safety rules (train-only rates, temporally strict history, no identity one-hots). Evidence: `graph_build.py`, `tests/test_graph_build.py`. Figure: schema diagram (to draw — nodes/edges/table sample).
- **5.4 AMR-SAGE model:** HeteroConv-of-SAGEConv encoder + triple MLP decoder; identical architecture constraint for averaging. Evidence: `model.py`, `train_local.py`.
- **5.5 Hospital simulation:** partition functions by family; determinism + no-leakage guarantees. Evidence: `partition.py`, `tests/test_partition.py`. Figure: split schematic per family.
- **5.6 Federated protocol:** Flower simulation, matched budgets, pooled/local-only/FedAvg definitions, seeding fix. Evidence: `federated/{run,task,client_app,server_app}.py`.
- **5.7 Topology-aware aggregation (the method):** 14-feature fingerprint (table + hygiene: dedup + variance floor), z-score → distance-to-centroid → softmax weights, consensus vs distinctiveness, T→∞=FedAvg anchor; client/server contract (JSON in fit metrics; fallback). Evidence: `topology.py`, `federated/strategy.py`, `tests/test_topology*.py`, research doc. Figures: fingerprint heatmap; weight schematic; example weight vectors (organism H3 0.74).
- **Missing:** notation table; final hyperparameter table (after §B.5 runs).

### 6. Experimental Setup
- **Purpose:** reproducibility.
- **Points:** full-cohort frame; splits used per experiment with n_clients/seeds; protocols (Phase-1 60 epochs hidden-64; fed v2 8r×4e hidden-128; specimen 10r×6e); baselines (majority 0.446, pooled, local-only, FedAvg); metrics + best/final reporting; hardware (Colab GPU T4/P100; topology on CPU); software versions (**UNKNOWN** — `requirements.txt` is unpinned; record exact versions before final runs); determinism procedure (seed+cid).
- **Evidence:** `run.py`, notebooks 03/04/07, `res.txt`.
- **Missing:** pinned environment file; wall-clock/compute costs (**UNKNOWN** — never logged); second-seed-set plan.

### 7. Results and Discussion
- **Purpose:** report honestly, interpret, limit.
- **7.1 Local model:** 0.663 core → 0.71/0.84 with history; 48-model search table (Top-10 0.665–0.668; kitchen-sink +0.0002). Evidence: `docs/2026-08-05-phase1-results.md`, notebook 03.
- **7.2 Federation baselines:** α-sweep 9/9 wins but tiny gains; specimen +0.019; organism-community +0.023/+0.037 worst; label-Dirichlet dead end as reported negative. Evidence: `docs/2026-08-06-federation-results.md`, `res.txt`.
- **7.3 The pooled-vs-FedAvg finding:** hard-split table (FedAvg beats pooled +0.014–+0.042); Louvain tiny-hospital numbers; mechanism (size-weighted pooled loss). Discuss, don't oversell.
- **7.4 Topology-aware evaluation:** the two PASS tables + per-seed breakdown (seed-43 win vs seed-42 loss); stability finding; AUROC side-table. Evidence: notebook-07 JSON output (pasted 2026 run).
- **7.5 Ablations/limits:** what was NOT run (§B.4) — present as explicit limitations + future work, not hidden.
- **Figures/tables:** see §F.
- **Missing:** §B.5 reruns; statistical tests; final-round table.

### 8. Conclusion and Future Work
- **Purpose:** restate contribution + honest limits + next steps (consensus sweep, more seeds, FedProx/FedGTA, Louvain clustering, explainability dashboard, second dataset).
- **Missing:** nothing (write last).

### 9. References
- **Purpose:** complete, verified IEEE-style references.
- **Currently verified from repo:** ARMD dataset citation (Dataset.md); Corbin et al. 2022 (cited in docs — re-verify bibliographic details); Hsu et al. 2019 / Li et al. 2022 NIID-Bench (cited in docs — re-verify); FLamby 2022 (cited in docs — re-verify).
- **Unverified candidates:** FedGTA, AdaFGL, FedSSA, FedStar, GCFL/GCFL+, FedSA-GCL, FedDAG, OpenFGL, FedGraphNN, q-FFL/AFL fairness, IFCA/Ditto — all need lookup before citing (see §I).

---

## E. Existing experimental evidence (with locations)

| Evidence | Location | Status |
|---|---|---|
| Local 0.663 → 0.71/0.84; 48-model search | `docs/2026-08-05-phase1-results.md`, nb 03 | logged |
| FedAvg α-sweep 9/9; specimen/organism wins; label-dir dead end | `docs/2026-08-06-federation-results.md`, `res.txt` | logged |
| Hard splits + gate FAIL + Louvain smoking gun | `docs/2026-08-11-splits-and-results.md` (§3 results + §7 design/gate, merged 2026-10-23 from the deleted 2026-08-09 calibration/design docs) | logged |
| Real-data fingerprints + similarity matrices + weights | notebook-06 JSON (Drive `amr_topology_fingerprints.json`; pasted in chat) | logged |
| Topology vs FedAvg, both PASS + per-seed breakdown | notebook-07 JSON (Drive `amr_topology_comparison.json`; pasted in chat) | logged |
| Determinism (~1e-4 across arms) | notebook-07 pooled/local columns | logged |
| Offline correctness (60+ tests) | `tests/` (6 files) | runnable |
| Implementation | `src/amr_fed/` (12 modules) | on `main` |

---

## F. Recommended tables and figures

| # | Artifact | Evidence available? | Still needed |
|---|---|---|---|
| T1 | Dataset summary (tables used, rows, patients, label rate, splits) | ✅ measured | nothing |
| T2 | Local-model results (baseline/core/history + 48-model top-10) | ✅ logged | nothing |
| T3 | Federation baselines per split (local/FedAvg/pooled/worst + seeds) | ✅ logged | 5-seed extension |
| T4 | Hard-split gate table (FedAvg vs pooled + smoking-gun row) | ✅ logged | nothing |
| T5 | Topology vs FedAvg (both splits, mean±std, worst, AUROC, per-seed) | ✅ logged (n=3) | n=5 + final-round cols |
| T6 | Ablation table (no-history / no-homophily / size-only / T sweep) | ❌ | new runs |
| T7 | Cost table (runtime, communication rounds, fingerprint bytes) | ❌ | log timings |
| F1 | System architecture diagram (data → KG → GNN → Flower → strategy) | ❌ | draw from §A.2 |
| F2 | KG schema + triple-supervision diagram | ❌ | draw from `graph_build.py` |
| F3 | Split-family schematic (one row per split type) | ❌ | draw from `partition.py` |
| F4 | Fingerprint heatmap (hospitals × 14 features, organism-community) | ✅ data in nb-06 JSON | plot it |
| F5 | Weight illustration (FedAvg size-bars vs topo distinctiveness bars, H3 0.74) | ✅ data in nb-06 JSON | plot it |
| F6 | Per-round F1 curves (FedAvg vs topo, best+final marked; seed-42 specimen decay vs topo stability) | ✅ data in nb-07 JSON (`fed_by_round`) | plot it |
| F7 | Worst-hospital gain forest-style plot with seed spread | ✅ data (nb-07) | plot after n=5 |

---

## G. Missing information and questions for you

1. **Target venue + page limit?** (Determines depth: workshop 4pp vs conference 8pp.)
2. **Author list and order?** (Docs name Vikram, Saffiya, Sanjana, Arshia + guide Dr. Swati Jagdale — confirm order and corresponding author.)
3. **Compute budget for §B.5 reruns?** (5 seeds × sweeps ≈ long Colab hours.)
4. **Exact software versions?** (`requirements.txt` is unpinned — needed for reproducibility section.)
5. **Code/data release plan?** (Code is on GitHub; ARMD is public via Dryad — confirm what the paper will link.)
6. **Clinical framing?** (Is there a clinician co-author or deployment context, or is this purely ML-methodology?)
7. **Which title direction** (§C) do you prefer?
8. **Should the dashboard/explainability work be in this paper or a follow-up?**

---

## H. Recommended additional experiments (in order)

1. Seeds 42–46 + paired tests/bootstrap CIs on worst-hospital delta (both splits).
2. Temperature × mode sweep (consensus/distinctiveness × T∈{1,2,5,∞}).
3. FedProx (proximal term — small code change) and FedGTA-style baselines, identical protocol.
4. Ablations: fingerprint minus history / minus homophily-hubness / size-only / no-hygiene.
5. Final-round comparison table alongside best-round.
6. Louvain + pairwise/clustering aggregation (centroid weighting shown insufficient there).
7. Timing/communication/fingerprint-size measurements for T7.

---

## I. Potential related-work search terms and citation needs

Search terms: "federated graph learning topology-aware aggregation", "personalized federated learning client similarity", "clustered federated learning healthcare", "graph neural networks antimicrobial resistance", "federated learning fairness worst-client", "non-IID partitioning graphs Louvain", "label propagation homophily federated".

- **Repo-attested (re-verify details before citing):** ARMD dataset paper; Corbin et al. 2022 personalized antibiogram; Hsu et al. 2019 Dirichlet protocol; Li et al. 2022 NIID-Bench; FLamby 2022.
- **Candidates from prior web research (VERIFY — do not cite yet):** FedGTA (VLDB 2024); AdaFGL (2024); FedSSA; FedStar; GCFL/GCFL+; FedSA-GCL; FedDAG; OpenFGL (VLDB 2025); FedGraphNN (2021); q-FFL/AFL; IFCA; Ditto; FedProx (Li et al. 2020).
- **Needed from scratch:** AMR burden/therapy-motivation clinical sources; healthcare-FL privacy surveys; GNN-for-EHR baselines.

---

## J. Risks, limitations, and claims that should not be made yet

1. **Do not claim statistical significance** — n=3, effect ≈ threshold ≪ std.
2. **Do not claim SOTA or generality** — one dataset, simulated federation, 2 splits, no external baselines yet.
3. **Do not claim the mean improves** — mean deltas (+0.001) are noise; the claim is worst-hospital only.
4. **Do not present best-round-only results without final-round alongside** — both arms decay across rounds.
5. **Do not claim pooled is "beaten" as a victory** — the correct statement is the mechanism (size-weighting starves rare communities), with FedAvg-as-rescue.
6. **Do not cite any §I candidate reference until verified** against the actual paper.
7. **Novelty is provisional** until the Related Work search (§I) is done — say "to our knowledge" nowhere yet.
8. **Methodological weakness to disclose:** temperature/mode chosen from fingerprint inspection before the live runs (mild selection); report the full sweep in §H.2 to address it.

---

## Summary and recommended first section

**Most important findings:** (1) an honest, reproducible end-to-end system with a strong local model; (2) FedAvg matches pooled without sharing data, with the largest gains on the worst hospitals; (3) the counterintuitive, mechanistically explained pooled-vs-FedAvg reversal on topology-divergent splits; (4) a working similarity-weighted aggregator that passes its pre-registered worst-hospital gate on two splits — with fragility openly quantified.

**Draft first: §5 Methodology** — it is fully determined by the code, needs no new decisions, and writing it will surface exactly which details (versions, timings, notation) must be nailed down before the Results section. After that: §F tables T1–T5, then Results, then Introduction, with Related Work running in parallel once the §I search is done.
