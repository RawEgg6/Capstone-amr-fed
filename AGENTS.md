# AGENTS.md — AMR Federated KG capstone

**Primary entry point for AI agents (OpenCode, Claude Code, Codex, …). Read this whole file
before making any change.** Human-facing setup lives in `README.md`, the data dictionary in
`Dataset.md`, research history in `docs/`. `CLAUDE.md` is a one-line pointer to this file.

- Repo folder: `Capstone-amr-fed` (older docs/README say `capstone` — same repo).
- Importable package: `amr_fed` under `src/` (installed editable: `pip install -e .`).
- Remote: `https://github.com/RawEgg6/Capstone-amr-fed.git`, working branch `main`.
- Team: Vikram, Saffiya, Sanjana, Arshia · Guide: Dr. Swati Jagdale.

## What we're building

A **topology-aware federated learning** system for antimicrobial-resistance (AMR) prediction.
Each simulated hospital builds a heterogeneous knowledge graph from ARMD, trains a local GNN,
and a central server aggregates models by **weighting each client on graph topology** (not
uniform FedAvg). Privacy-preserving: only model weights + a small topology fingerprint ever
leave a client.

**Status snapshot (2026-10-24):** Phases 0–5 are done and evaluated (local model, partitions,
FedAvg baseline, topology fingerprint, topology-aware `Strategy` — Path-B gate PASS on 2 splits
at n=3, fragile). Phase 6 (baseline comparisons) is active: the Louvain split was re-run with
per-hospital output (nb11) and the earlier headline was corrected (see Results landscape); the
10-hospital six-method comparison (nb12) has run (`docs/2026-10-25-ten-hospital-results.md`), but its topo arm ran on a Mac CPU while the other arms ran on Colab, so the topo-vs-control gap needs a same-machine rerun. Phase 7 (dashboard)
not started; Phase 8 (paper) is outline-only (`research_outline.md`).

## Locked decisions — do not silently change (source of truth: `src/amr_fed/config.py`)

- **Dataset:** ARMD (Stanford, single institution), 10 of 16 CSVs. Path via `config.DATA_DIR`.
- **Graph:** heterogeneous, 5 node types — patient, organism, antibiotic, comorbidity, procedure.
- **Edges (6):** core = `(organism,tested,antibiotic)` [label], `(patient,grew,organism)`,
  `(organism,known_resistant,antibiotic)`; enrichment = `(patient,underwent,procedure)`,
  `(patient,has,comorbidity)`, `(patient,prior_exposure,antibiotic)`.
- **Target:** **binary** edge classification — Resistant+Intermediate vs Susceptible — on the
  `(organism, tested, antibiotic)` edge. Class-weighted BCE; report macro-F1 (+ AUROC).
  Keep only `was_positive=1` + S/I/R rows → 1.60M rows, 67,008 patients, label rate 0.1952.
- **"Hospitals":** simulated by partitioning **patients** 1:1 (no leakage). Baseline =
  Dirichlet(α) ward-mixture (α-swept) + plain ward + ADI; many harder splits added since
  (see "Splits" below).
- **GNN:** heterogeneous GraphSAGE ("AMR-SAGE"), **identical architecture across all clients**
  (required for weight averaging). Hidden 64 (Phase 1) / 128 (federation v2 protocol).
- **Federation:** Flower (`flwr[simulation]`, Ray); `federated/strategy.py` adds a topology-aware
  `Strategy` subclass on top of FedAvg.

Rationale for every locked decision: `docs/2026-07-19-eda-decision-brief.md` (the reference).

## Data access (team)

- Code → **GitHub**; data → **one shared Google Drive folder** (the 10 ARMD CSVs). Never commit data.
- Each person sets `ARMD_DIR` to their own mount; `config.py` resolves it
  (env var → Colab Drive mount `/content/drive/MyDrive/ARMD` → local `data/raw/`).

## Environment & setup

**Local dev (Mac/Linux/WSL):**
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements.txt && pip install -e .
export ARMD_DIR="/path/to/your/ARMD"   # persist in ~/.zshrc
```
- `requirements.txt` is **unpinned** (by design so far); `pyproject.toml` requires Python >=3.11.
  The repo's `.venv` is Python 3.13 and currently has **no torch/flwr** — 4 of 6 test files run
  locally; the two federated tests print `SKIP` without the FL stack.
- **Colab (where all training runs; versions logged 2026-10-23):** python 3.13.15,
  torch 2.11.0+cu130, torch_geometric 2.8.1, flwr 1.41.0, ray 2.55.1, 1 GPU (T4/P100, high-RAM).
  Actor env: `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=0`, `TF_FORCE_GPU_ALLOW_GROWTH=true`.
  Flower 1.41 warns `run_simulation` is deprecated (still works; `flwr run` is the future path).

## How to run

| What | Command |
|---|---|
| Setup check | run `notebooks/00_setup_check.ipynb` (local or Colab) |
| Unit tests (plain scripts, no pytest) | `python tests/test_<name>.py` — each prints `OK: …` or `SKIP` |
| Module self-checks (need `ARMD_DIR`) | `python -m amr_fed.data_loader` / `graph_build` / `partition` / `topology` |
| Local training (Phase 1) | `notebooks/03_train_local.ipynb` (Colab GPU) |
| FedAvg / split experiments | `notebooks/04_federated.ipynb` (Colab GPU) |
| Fingerprints / Phase-5 comparison / pre-registered experiment | `notebooks/06`, `07`, `08` (Colab GPU) |
| Louvain per-hospital / 10-hospital six-method comparison | `notebooks/11`, `12` (Colab GPU) |
| Raw committed run log | `res.txt` (572 lines, Colab output) |

Tests: 11 files — `test_data_loader`, `test_graph_build`, `test_partition` (32 tests),
`test_val_selection` + `test_final_protocol` + `test_final_analysis` + `test_regimes` + `test_simft` (torch-free; round/cutoff selection, per-hospital cutoffs, v3 weights, analysis, data regimes), `test_topology` (17), `test_topology_strategy` (needs torch), `test_federated_seeding`
(needs flwr). All green as of `d05b736` (federated pair green on Colab, skip locally).

## Architecture (data flow)

```
ARMD CSVs (Drive, never committed)
  → config.py         locked schema: 5 node types, 6 edges, binary label, SEED, DATA_DIR
  → data_loader.py    10/16 tables joined on 4 keys → 1.60M-row per-test frame, label {0,1}
  → graph_build.py    frame → PyG HeteroData (patient/organism/antibiotic core + enrichment)
  → model.py          AMR-SAGE: HeteroConv of SAGEConv + MLP decoder over triples
                      (dropout() draws masks from the CPU generator — actor CUDA-RNG guard)
  → train_local.py    class-weighted BCE; macro-F1, AUROC, threshold tuning, sweeps
  → partition.py      1:1 patient→hospital assignment (11 split fns; pure pandas, no torch)
  → topology.py       14-feature hospital fingerprint → z-score → distance-to-centroid
                      → softmax aggregation weights (consensus | distinctiveness, T dial)
  → federated/        run.py (run_local_only/run_pooled/run_fedavg/run_multiseed/headroom_gate)
                      task.py (graphs via disk, seeding, init/train/eval) · client_app.py ·
                      server_app.py · strategy.py (TopologyAwareStrategy extends FedAvg)
  → notebooks/ 00–08  drivers + EDA (import modules; no logic defined here)
  → tests/            offline asserts (no data/GPU needed except FL-stack skips)
```

**Splits in `partition.py`:** `dirichlet_ward_mixture`, `label_dirichlet`, `specimen_baseline`,
`organism_community`, `ward_baseline`, `adi_baseline`, `homophily_split`, `degree_skew_split`,
`topology_split` (2-D corners), `louvain_split`, `antibiogram_conflict` (pre-registered).
Contract: `partition_fn(df, n_clients, seed) -> Series(index=patient → client id)`, 1:1, balanced.

**Topology arm wiring:** driver computes fingerprints into `run_config.json` → client forwards
its vector in fit metrics → `TopologyAwareStrategy.aggregate_fit` converts similarity to weights.
`temperature=inf` reproduces FedAvg exactly (unit-tested). Missing payloads fall back to size-proportional.

## Results landscape (evidence: docs/ + res.txt + notebook JSON on Drive)

| Milestone | Headline numbers | Where |
|---|---|---|
| Local model | majority 0.446 → core **0.663** → +patient-history **0.707–0.712** (AUROC 0.836; 48-model search found nothing better) | `docs/2026-08-05-phase1-results.md`, `2026-08-06-federation-results.md` |
| FedAvg baseline | beats local-only 9/9 ward runs; specimen +0.019 (worst +0.023); organism-community +0.023 (**worst +0.037**); label-Dirichlet = reported dead end | `docs/2026-08-11-splits-and-results.md` |
| Hard topology splits | headroom gate **FAIL in the informative direction**: on topology-divergent splits FedAvg *beats* pooled (size-weighted pooled loss starves rare communities; Louvain 5% community: local 0.7423 → pooled 0.6735 → FedAvg 0.7308) | `docs/2026-08-11-splits-and-results.md` §3, §7 |
| Phase 5 (novelty) | Path-B gate PASS: worst-hospital +0.0054 (organism-community) / +0.0050 (specimen); mean deltas noise-level; determinism fixed (`0af90d0`); fragile at n=3 (std ≫ effect) | `docs/2026-10-18-topology-phase5-results.md`, nb 07 |
| Experiment #1 (done) | antibiogram-conflict split, 5 seeds: **C1 FAIL, C2 FAIL, C3 marginal PASS** — condition not reproduced (FedAvg *helps* minority +0.028); topo tail +0.0045 within noise | `docs/2026-10-18-antibiogram-conflict-split.md` §8 |
| Louvain topology arm (done, corrected 2026-10-24) | `louvain_split(5)` actually yields **3 hospitals** (~73/21/5%). Topo-aware beats FedAvg on weighted best-round F1 (0.6987 vs 0.6785, std 0.006 vs 0.023) and beats its own local model on the largest hospital in 5/5 seeds (+0.024); on the middle/smallest hospitals both FedAvg and topo are *worse than local-only* (topo less so). Topo weights are near-uniform (~1/3 each) → **uniform-weight control (T=inf) not yet run**. C1/C2/C3 and "minority hospital" framing retired | `docs/2026-10-23-louvain-topology-results.md` |
| 5-way baseline comparison (nb10, aggregates only) | FedMedian / FedTrimmedAvg / Topo are indistinguishable on best-round (≈0.698) and worst-hospital (≈0.67); final-round Topo 0.6786 ≈ FedTrimmedAvg 0.6777 > FedMedian 0.6542 > FedProx 0.6490 > FedAvg 0.6457. Topo does **not** clearly beat the robust baselines; earlier "wins all three metrics / final 0.6907" claim withdrawn | `docs/2026-10-23-louvain-topology-results.md` §6 |

**Do-not-claim list** (binding, from `research_outline.md` §J): no statistical-significance
claims (n=3), no SOTA/generality, no "mean improves" (mean deltas are noise), no best-round-only
tables without final-round, no "we beat centralized" headline, no citing unverified references.

## Notebook index (drivers; output stripped — re-run on Colab)

| # | Notebook | Role |
|---|---|---|
| 00 | setup_check | env bootstrap + import/CSV checks (local & Colab) |
| 01 | eda | first-look EDA (executed, outputs kept) |
| 02 | graph_eda | comprehensive decision-support EDA (executed) — the source of the decision brief |
| 03 | train_local | Phase-1 local training driver (incl. 48-model grid) |
| 04 | federated | Phase-3 FedAvg experiments, cells 4–11 = split calibration |
| 05 | demo_modules | guided walkthrough of every module (demo for panel) |
| 06 | topology_fingerprints | fingerprint + similarity-matrix inspection (saves JSON to Drive) |
| 07 | topology_comparison | Phase-5 FedAvg-vs-topology arms, verdict, T×mode sweep (added 2026-10-18) |
| 08 | antibiogram_comparison | pre-registered experiment #1 (done, null result); cells D1–D3 are SIGFPE forensics |
| 11 | louvain_per_hospital | Louvain re-run, every per-hospital number (source of truth for the Louvain doc) |
| 14 | louvain_val_selection | Louvain, six arms, **validation-selected protocol** (round + cutoff chosen on val, test read once); Colab or local; writes `results/amr_louvain_val.json` |
| 16 | louvain_final | **FINAL protocol** on Louvain (3 hospitals): per-hospital cutoffs, FedAvg + fine-tune, topo v3 (3 temperatures + shuffled control), plus Uniform / Topo v1 / FedProx / FedMedian (9 federated runs per seed); writes `results/amr_louvain_final_<TAG>.json` |
| 17 | org10_final | **FINAL protocol** on `organism_community` n=10, same 9 arms; `RUN_ARMS`/`TAG` split work across accounts, cell 9 merges; writes `results/amr_org10_final_<TAG>.json` |
| 19 | simft | **SimFT** (pre-registered in `docs/2026-10-28-simft-preregistration.md`): personalized aggregation weighted by function-space similarity (hospital i scores every hospital's model on its validation half A) + final fine-tune; settings: louvain (baselines nb16), org10 (nb17), ward10 at 10% data (nb18) — only the 3 SimFT arms are trained, baselines are read from those notebooks' files; writes `results/amr_simft_<setting>_<TAG>.json` |
| 18 | low_data_and_new_site | **Where federation helps** (pre-registered in `docs/2026-10-28-low-data-and-new-site-preregistration.md`): ward-Dirichlet 10 hospitals; Part A low-data curve (keep 100/25/10/3% of train+val patients; local-only vs FedAvg vs FedAvg+fine-tune), Part B new hospital (hold one out; zero-shot / fine-tuned / scratch at k labelled patients); writes `results/amr_regimes_<TAG>.json` |
| 15 | org10_val_controls | `organism_community` n=10, **validation-selected protocol**, 8 arms: the six from nb12 plus two topology controls — `topo_shuffled` (each hospital gets another's fingerprint) and `invsize` (weights ∝ 1/n_train); `RUN_ARMS`/`TAG` split work across accounts, cell 8 merges; writes `results/amr_org10_val_<TAG>.json` |
| 12 | ten_hospitals | `organism_community` n=10: FedAvg / FedProx / FedMedian / FedTrimmedAvg / uniform control / Topo (run 2026-10-25; topo arm on Mac CPU, rest on Colab) |

## Docs index (append-only history; wrong claims are deleted, not left in place)

- **Active results:** `docs/2026-10-23-louvain-topology-results.md` (rewritten 2026-10-24 from nb11),
  `docs/2026-10-18-topology-phase5-results.md`, `docs/2026-10-18-antibiogram-conflict-split.md`
  (pre-registration + null result), `research_outline.md` (paper outline; §G open questions, §J claim guardrails).
- **Design record:** `docs/2026-07-19-eda-decision-brief.md` (locked decisions + rationale),
  `docs/2026-09-28-topology-module-research.md` (Phase-5 design).
- **Results history:** `docs/2026-08-05-phase1-results.md`, `docs/2026-08-06-federation-results.md`,
  `docs/2026-08-11-splits-and-results.md` (**canonical consolidated record** of all splits and the headroom gate).
- **Reference:** `Dataset.md` (ARMD data card), `README.md` (setup), `res.txt` (raw log).
- *Removed 2026-10-24 (recoverable from git history): notebooks 09/10 (superseded by nb11/12; carried
  the incorrect "k=5" framing), `prompts/baseline-comparison-handoff.md` (stale handoff). Earlier
  removals: `2026-07-19-next-steps.md`, the three `2026-08-09-*` docs, `2026-08-11-handoff.md`. Do not re-create them.*

## Active work & TODOs

**A. Active / blocked**
1. Louvain re-run with per-hospital output (nb11) **done 2026-10-24** — see corrected results landscape.
2. Next priority: (a) rerun nb12 topo and uniform arms on one machine; (b) confirm the 10-hospital topo weight
   concentration (0.73 on one 3,322-patient hospital) is not a fingerprint artifact; (c) personalization.
   Louvain is 3 hospitals (`n_clients=3`); nb11/nb13 previously passed `n_clients=5`, which gave the same 3 but misstated the setup.
3. Keep `os.environ.pop("AMR_RNG_PROBE", None)` at the top of every arm cell (SIGFPE guard).

**B. Next experiments (priority order)**
1. Same-machine rerun of nb12 topo + uniform; the Louvain uniform control is in nb13 (not yet run).
2. Personalization (local fine-tune after the last round); round-decay control (server LR / best-val round).
3. Ablations: fingerprint without history / without homophily-hubness / size-only weights.
4. Bootstrap CIs on per-hospital deltas; timing/cost logging.

**C. Paper (`research_outline.md`)**
- §G open questions (venue, author order, compute budget, code/data release, title choice).
- Verify every §I citation before use; pin exact software versions for the reproducibility section.

**D. Housekeeping (team decisions, not unilateral)**
- Reconcile `res.txt` specimen FedAvg numbers vs notebook-07 comparison table (documented mismatch).
- Stale branches: `phase1-core-pipeline`, `phase3-federated`, `fix/spliting-tries`,
  `phase5-topology-aware`, `topology-aware-aggregation`.
- Unpinned `requirements.txt`; deprecated `run_simulation` in flwr 1.41.

## Evaluation protocol (federated runs)

- **Old (legacy keys, kept for comparison):** `fedavg_best` = best round by *test* macro-F1 at a fixed
  0.5 cutoff — optimistic; local-only (`local_only`) is final-epoch @0.5, no selection.
- **Current (`val_*` keys, `federated/selection.py`):** each hospital scores the global model on its
  val and test rows at 29 cutoffs (0.150–0.850) every round; the server picks the (round, cutoff)
  with the best size-weighted **validation** macro-F1 and the test score is read once there.
  Local-only is scored every `local_epochs` epochs with the same rule (per hospital). Reported keys:
  `val_fed`, `val_fed_at05`, `val_fed_final`, `val_worst_fed`, `val_fed_auroc`, `val_local`, ...
  Quote these, plus final-round and AUROC; do not quote `fedavg_best` alone.
- **Final protocol (`phc_*` keys, notebooks 16/17, pre-registered in
  `docs/2026-10-28-final-protocol-preregistration.md`):** same as above, but every hospital picks its
  **own cutoff** on its own validation rows (`selection.select_federated_phc`) — what local-only always did.
  Primary metric `phc_f1`; also `phc_worst`, `val_fed` (shared cutoff), `ft_f1` (FedAvg + 8-epoch
  per-hospital fine-tune), `phc_val_score` (used to pick v3's temperature). Local-only is trained once per
  seed, in the FedAvg arm (`compute_local=False` elsewhere). Results store confusion counts per cutoff and
  probability histograms (`records`, `local_detail`), weights, fingerprints (all rows and train rows),
  assignments and provenance, so nothing needs retraining. Analysis lives in `federated/analysis.py`,
  the arm list / run loop in `federated/final_protocol.py`.
- **Topo v3 (`strategy="personalized"`, `strategy.PersonalizedStrategy`, `topology.personalized_weights`):**
  every hospital gets its own model, a similarity-weighted average of all hospitals' trained models
  (`W[i,j] = softmax_j(-D[i,j]/T)`, fingerprints from train rows, size features dropped). T=inf equals Uniform
  exactly (verified), T→0 is local training.
- Only per-hospital counts at the cutoff grid travel to the server (no logits, no patient data).

## Conventions & constraints

- Core logic = modules under `src/amr_fed/`. Notebooks only import + call them (exceptions:
  historical notebook 04 cells carry experiment config — that is accepted state, don't refactor).
- **Never commit `data/`, CSVs, `.pt` caches, or any patient data** (gitignored). Never
  commit or push unless explicitly asked.
- **Features:** do NOT use one-hot identity features (caps a GNN at ~0.49 F1). Use resistance
  ratios, counts, degrees, demographics, ADI, temporal recency, patient-history (temporally
  strict — leakage-tested).
- Set seeds from `config.SEED` (42); federated seeding = `seed + cid` in `client_fn`
  (`0af90d0`). Keep client architectures identical (FedAvg alignment). Client feature dims must
  match — no per-client schema changes.
- `partition.py` stays pure pandas/numpy (no torch import).
- Docs are dated `YYYY-MM-DD-name.md`; results docs are historical records — append dated
  updates, never rewrite past numbers.
- Pre-registration docs: criteria fixed before runs; any change after seeing results is an
  **iteration** and must be appended with date + rationale, reporting both runs.
- Colab is the compute; local venv is for partition/topology tests + code authoring only.
- Existing agent-specific config: `.claude/settings.local.json` (permissions allowlist — keep),
  `.superpowers/sdd/` (spec-driven-dev task history — keep, do not delete).

## Known ambiguities (flagged, not resolved)

1. Repo folder name: `Capstone-amr-fed` vs `capstone` in README/old AGENTS.md.
2. README recommends Python 3.11; actual runs are on 3.13 (local + Colab); pyproject allows >=3.11.
3. Per-patient vs per-bug-drug prediction framing (2026-08-05 open team question) was resolved
   *de facto* by the code (per-test triples `(patient, organism, antibiotic)`), but no explicit
   decision record exists in `docs/`.
4. `docs/2026-10-18-topology-phase5-results.md` numbers are transcribed from pasted notebook
   output and flagged **not re-verified** against Drive JSON.
