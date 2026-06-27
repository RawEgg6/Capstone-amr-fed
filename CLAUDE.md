# CLAUDE.md — AMR Federated KG capstone

Context for Claude Code. Read before making changes.
(Repo folder is `capstone`; the importable Python package is `amr_fed`.)

## What we're building
A **topology-aware federated learning** system for antimicrobial-resistance prediction.
Each simulated hospital builds a heterogeneous knowledge graph from ARMD, trains a local
GNN, and a central server aggregates models by **weighting each client on graph topology**
(not uniform FedAvg). Privacy-preserving: only model weights + a small topology fingerprint
ever leave a client.

## Locked decisions — do not silently change (see `src/amr_fed/config.py`)
- **Dataset:** ARMD (Stanford, single institution). 10 of 16 CSVs used. Data found via `config.DATA_DIR`.
- **Graph:** heterogeneous, 5 node types — patient, organism, antibiotic, comorbidity, procedure.
- **Target:** edge classification of S/I/R on the `(organism, tested, antibiotic)` edge.
- **"Hospitals":** simulated by **ward** (ICU/ER/IP/OP) since ARMD is one site.
- **GNN:** heterogeneous GraphSAGE ("AMR-SAGE"), **identical architecture across all clients**.
- **Federation:** Flower (`flwr[simulation]`); a custom topology-aware `Strategy` subclass aggregates.

## Data access (team)
- Code is shared via **GitHub**; data via a **shared Google Drive folder** (never committed).
- Each person sets `ARMD_DIR` to their own mount of that shared folder; `config.py` resolves it
  (env var -> Colab Drive mount -> local `data/raw/`). Same code runs for everyone.

## Build order
0. Env + schema (this scaffold)
1. Single-ward pipeline: data -> KG -> local GNN with strong macro-F1
2. Ward partitioning (IID + non-IID)
3. FedAvg baseline (must be >= local-only before adding novelty)
4. Topology profiling module
5. Topology-aware aggregation (the novel contribution)
6. Evaluation vs FedAvg / FedProx / FedGTA
7. Explainability + clinical dashboard
8. Paper / thesis / deck

## Intended structure
```
src/amr_fed/
  config.py        # locked constants (done)
  data_loader.py   # load + join the 10 tables on the 4 keys
  graph_build.py   # build the 5-node heterogeneous KG (PyG HeteroData)
  partition.py     # ward-based client partitioning
  model.py         # AMR-SAGE heterogeneous GNN
  train_local.py   # single-client training + eval
  topology.py      # Phase 4: per-client topology fingerprint
  federated/
    client_app.py  # Flower ClientApp
    server_app.py  # Flower ServerApp
    strategy.py    # Phase 5: topology-aware aggregation Strategy
notebooks/         # EDA + final plots ONLY (import modules, don't define logic here)
tests/
```

## Conventions
- Core logic = modules under `src/amr_fed/`. Notebooks only import + call them.
- **Never** commit `data/` or any patient data.
- **Features:** do NOT use one-hot identity features — a GNN on them tops out ~0.49 F1.
  Use resistance/susceptibility ratios, counts, degrees, demographics, ADI, temporal recency.
- Set seeds from `config.SEED`. Keep client architectures identical.
