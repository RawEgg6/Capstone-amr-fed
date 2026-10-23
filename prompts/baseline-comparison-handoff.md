# Handoff prompt — federated baseline comparison (paste this into a new Claude session)

---

## What this project is

A topology-aware federated learning system for antimicrobial-resistance (AMR) prediction.
5 simulated hospitals each build a heterogeneous knowledge graph (patients / organisms /
antibiotics / comorbidities / procedures) from the ARMD dataset and train a local
heterogeneous GraphSAGE (AMR-SAGE). A central server aggregates model weights. The novel
contribution: instead of uniform weight averaging (FedAvg), the server weights each hospital
by how *topologically distinct* its patient graph is (Phase-5 `TopologyAwareStrategy`).

Repo: `https://github.com/RawEgg6/Capstone-amr-fed.git` · branch `main`  
Package: `amr_fed` under `src/` (installed editable)  
Primary agent entry point: `AGENTS.md` in the repo root — read it before making any change.  
Conventions live there; a few critical ones are repeated below.

---

## Current results (the context for this task)

**Best result so far (Louvain k=5 split, notebook 09, committed 2026-10-23):**

| Arm | Best-round macro-F1 (mean ± std, 5 seeds) | Worst-hosp F1 |
|---|---|---|
| Local-only | — | 0.6742 ± 0.022 |
| FedAvg | 0.6785 ± 0.023 | 0.6528 ± 0.030 |
| Topology-aware (ours) | **0.6987 ± 0.006** | **0.6712 ± 0.010** |

- C1 PASS: FedAvg hurts the minority hospital (worst_fed 0.6528 < worst_local 0.6742)
- C2 PASS: topology-aware Δmean +0.0202, Δworst +0.0184 (4× the Phase-5 effect)
- C3 PASS: tail hospital topo 0.7201 vs FedAvg 0.7058; nearly recovers to local 0.7215
- Pooled arm was NOT run on this split — that is task 0 below.

The Louvain split creates structurally divergent hospitals (each hospital inherits a
distinct organism-antibiotic community subgraph), which is exactly the regime where
size-weighted averaging fails the minority. This is the paper's headline result.

---

## Your tasks

### Task 0 — add pooled arm to notebook 09 (small)

Open `notebooks/09_louvain_topology.ipynb`. The verdict cell (cell 10) computes C1/C2/C3
but `compute_pooled` was not explicitly set, so pooled may or may not have run. Check the
existing cells: if `lv5_fed` already has `pooled` key populated (not None), add one line
to the verdict print. If it's None, add a new cell **before cell 6** that runs
`run_pooled(n_clients, cfg, epochs=32)` explicitly and prints the result. The three-way
table should read: local / pooled / FedAvg / topo.

### Task 1 — implement three federated baselines

Add three new `strategy=` options to `run_fedavg` in `src/amr_fed/federated/run.py`.
Accepted strategies currently: `"fedavg"`, `"topology"`. Add: `"fedprox"`, `"fedmedian"`,
`"fedtrimmedavg"`. Details below.

**1a. FedMedian (server-side only, easiest)**

Flower 1.41 ships `flwr.server.strategy.FedMedian`. It aggregates by coordinate-wise
median instead of weighted mean — robust to outlier client updates.

Change needed: `src/amr_fed/federated/server_app.py` — wherever the strategy is
instantiated, add a branch for `"fedmedian"` that creates `FedMedian(...)` with the same
`evaluate` / `on_fit_config_fn` / `on_evaluate_config_fn` as the existing FedAvg branch.
Then in `run_fedavg` allow `strategy="fedmedian"` (the ValueError guard currently rejects
it — remove it from the guard).

**1b. FedTrimmedAvg (server-side only, easy)**

Flower ships `flwr.server.strategy.FedTrimmedAvg`. It drops the top and bottom
`beta` fraction of client updates before averaging (default beta=0.1 drops 10% each side).

Same change pattern as 1a. Add `"fedtrimmedavg"` strategy; instantiate
`FedTrimmedAvg(beta=0.1, ...)` in `server_app.py`.

**1c. FedProx (client + server, medium)**

FedProx (Li et al., ICLR 2020, arXiv:1907.02189) adds a proximal term to each client's
local loss: `L_local + (μ/2) * ||w - w_global||²`. This penalises client drift away from
the global model and is the most widely cited FedAvg improvement in the literature.

Changes needed:

*server side* (`server_app.py`): when `strategy="fedprox"`, pass `{"proximal_mu": 0.01}`
in `on_fit_config_fn` so each client receives μ in its fit config. Use the regular
`FedAvg` strategy (or Flower's `FedProx` strategy if available — check
`flwr.server.strategy.FedProx`; if not available fall back to `FedAvg` with the config).

*client side* (`src/amr_fed/federated/task.py`): modify `local_train` to accept an
optional `proximal_mu: float = 0.0` and optional `global_weights` (a list of numpy arrays,
same format as `get_weights`). When `proximal_mu > 0`, after computing the normal loss add:
```python
prox = sum((p - g.to(p.device)).pow(2).sum()
           for p, g in zip(model.parameters(), global_tensors))
loss = loss + (proximal_mu / 2.0) * prox
```
where `global_tensors` are the global weights converted to tensors once before the epoch
loop (use `torch.tensor(w, dtype=p.dtype, device=p.device)` for each numpy array `w`).

*client_app.py* (`FlowerClient.fit`): read `proximal_mu = float(config.get("proximal_mu", 0.0))`
from the Flower fit config dict, and pass it + the current weights (before `set_weights`)
to `local_train`.

**Important:** `local_train` signature in `task.py` currently is:
```python
def local_train(model, data, epochs: int):
```
Extend it to:
```python
def local_train(model, data, epochs: int, proximal_mu: float = 0.0, global_weights=None):
```
Default behaviour (μ=0, no global_weights) must be identical to today — all existing
callers pass only `model, data, epochs`.

### Task 2 — create notebook 10 with the 5-way comparison

Create `notebooks/10_baseline_comparison.ipynb`. It runs all 5 methods on the Louvain k=5
split (the same setup as notebook 09), collects results, and prints a comparison table.

**NOTE: FedAvg and Topology arms are already done** (notebook 09, results saved to Drive).
Do NOT rerun them. Load the saved results from Drive in cell 6, then run only the three
new baselines.

**Cells to include (in order):**

```
Cell 1:  setup (git clone/pull, pip install)         — copy from nb 09 cell 1
Cell 2:  mount Drive + ARMD_DIR                      — copy from nb 09 cell 2
Cell 3:  imports + sanity check                      — copy from nb 09 cell 3
Cell 4:  load cohort once (df = load_cohort_frame()) — copy from nb 09 cell 4
Cell 5:  smoke check (Louvain k=5 validity)           — copy from nb 09 cell 5, k=5 only
Cell 6:  load existing FedAvg + Topo results from Drive (no rerun)
Cell 7:  FedProx arm     — new (lv5_fedprox)
Cell 8:  FedMedian arm   — new (lv5_fedmedian)
Cell 9:  FedTrimmedAvg arm — new (lv5_fedtrimmedavg)
Cell 10: comparison table + verdict
```

Cell 6 loads the already-computed results instead of rerunning:
```python
# 6) Load existing FedAvg + Topo results from notebook 09 (already computed)
import json
with open('/content/drive/MyDrive/amr_louvain_k5_verdict.json') as f:
    nb09 = json.load(f)
# Reconstruct the run-dict shape run_multiseed returns, using the saved numbers
lv5_fed = {
    'fedavg_best':  nb09['fed']['best'],
    'worst_fed':    nb09['fed']['worst'],
    'worst_local':  nb09['fed']['worst_local'],
    'fedavg_final': nb09['fed']['final'],
    'runs': [],  # per-seed breakdown not needed for the summary table
}
lv5_topo = {
    'fedavg_best':  nb09['topo']['best'],
    'worst_fed':    nb09['topo']['worst'],
    'worst_local':  nb09['topo']['worst_local'],
    'fedavg_final': nb09['topo']['final'],
    'runs': [],
}
print('Loaded FedAvg best:', nb09['fed']['best'], '| Topo best:', nb09['topo']['best'])
```

Each new arm cell (7, 8, 9) follows the exact same pattern as nb 09 cells 6/7:
- `os.environ.pop('AMR_RNG_PROBE', None)` at the top (mandatory — crashes if set)
- `ray.shutdown()` before the run
- reload amr_fed modules
- call `run_multiseed` with `strategy='fedprox'` / `'fedmedian'` / `'fedtrimmedavg'`
- same seeds (42–46), same split, same protocol (rounds=8, local_epochs=4, hidden=128)

Example cell 7:
```python
# 7) FedProx arm (μ=0.01)
import os, ray
os.environ.pop('AMR_RNG_PROBE', None)
os.environ['AMR_DEBUG_INIT'] = '1'
ray.shutdown()
import sys as _s
for _m in [m for m in list(_s.modules) if m.startswith('amr_fed')]:
    del _s.modules[_m]
from amr_fed.federated.run import run_multiseed
from amr_fed.partition import louvain_split

lv5_fedprox = run_multiseed(
    seeds=(42, 43, 44, 45, 46), df=df,
    label='louvain-k5/FedProx',
    partition_fn=lambda d, s: louvain_split(d, n_clients=5, seed=s),
    n_clients=5, rounds=8, local_epochs=4,
    local_only_epochs=32, hidden=128,
    strategy='fedprox'
)
```

The comparison table cell (11) should print:

```
Method         | Best-round F1 (mean±std) | Worst-hosp F1 (mean±std) | Final F1 (mean±std)
Local-only     | ...                      | ...                      | —
Pooled         | ...                      | ...                      | ...
FedAvg         | ...                      | ...                      | ...
FedProx(μ=0.01)| ...                      | ...                      | ...
FedMedian      | ...                      | ...                      | ...
FedTrimmedAvg  | ...                      | ...                      | ...
Topo-aware     | ...                      | ...                      | ...
```

Save the full results dict to `/content/drive/MyDrive/amr_baseline_comparison.json`.
Print `=== COPY EVERYTHING BELOW THIS LINE BACK TO CHAT ===` then the JSON then `=== END ===`.

---

## Key constraints (binding)

- **Never commit `data/`, CSVs, `.pt` caches, or patient data.**
- **Never commit or push unless the user explicitly asks.**
- **Do not rewrite existing notebooks.** Add cells only. Notebook 09 cells already run and
  produced valid results — do not change them.
- Default `proximal_mu=0.0` in `local_train` must leave existing behaviour **identical**.
  Run the existing tests (`python tests/test_federated_seeding.py`) to verify no regression.
- Do not add `strategy="fedprox"` / `"fedmedian"` / `"fedtrimmedavg"` to `run_multiseed`'s
  ValueError guard — just pass them through. The guard currently rejects unknown strategy
  strings; remove only those three from it (or restructure the check).
- Keep client architectures identical across all methods (required for weight averaging).
- `partition.py` must stay pure pandas/numpy (no torch import).
- The `AMR_RNG_PROBE` env var must never be set during training arms. Every arm cell must
  have `os.environ.pop('AMR_RNG_PROBE', None)` as its first line.

---

## File map (what to touch)

```
src/amr_fed/federated/
  run.py         — remove "fedprox"/"fedmedian"/"fedtrimmedavg" from ValueError guard;
                   pass strategy param through to server_app
  server_app.py  — add strategy branches for fedmedian, fedtrimmedavg, fedprox
  task.py        — extend local_train signature (proximal_mu, global_weights)
  client_app.py  — read proximal_mu from fit config, pass to local_train

notebooks/
  09_louvain_topology.ipynb  — add pooled-arm cell if missing (Task 0 only)
  10_baseline_comparison.ipynb — create new (Task 2)
```

Do NOT touch `strategy.py` (TopologyAwareStrategy), `partition.py`, `model.py`,
`graph_build.py`, `data_loader.py`, `config.py`, or any existing notebook cells.

---

## How to verify before running on Colab

```bash
# local (no GPU, no flwr needed for these):
python tests/test_federated_seeding.py   # should still print OK
python -c "from amr_fed.federated.task import local_train; print('local_train OK')"
python -c "from amr_fed.federated.run import run_fedavg; print('run_fedavg OK')"
```

If `test_federated_seeding.py` prints SKIP (no flwr installed locally), that is fine —
it means the federated stack isn't installed; just verify the imports don't error.

---

## Order to run cells on Colab (notebook 10)

1. Run cells 1–5 (setup → smoke). Confirm `SMOKE PASS k=5`.
2. Run cell 6 (load saved FedAvg + Topo from Drive) — instant, no training.
3. Run cell 7 (FedProx). FedProx should improve on FedAvg slightly or match it.
4. Run cell 8 (FedMedian).
5. Run cell 9 (FedTrimmedAvg).
6. Run cell 10 (table + verdict). Paste the JSON block back into the original session.

Each new arm takes ~15–20 min on Colab T4 (5 seeds × 8 rounds). Total runtime ≈ 45–60 min
(3 arms only, not 5). Run cells sequentially so Ray actors don't compete for GPU memory.

---

## After getting results

Paste the `=== COPY EVERYTHING BELOW ... ===` block into chat. The next step is to record
results in `docs/2026-10-23-louvain-topology-results.md` (append a §2 with the 5-way table)
and update `AGENTS.md` results landscape. The user will ask Claude to do that.
