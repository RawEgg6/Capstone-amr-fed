# Capstone 171 — AMR Federated KG

Topology-aware federated learning for antimicrobial-resistance prediction on the ARMD dataset.
(Repo folder: `capstone`. Python package: `amr_fed`.)

## How the team shares things
- **Code → GitHub.** Everyone clones this repo, pushes/pulls. Code never goes in Drive.
- **Data → one shared Google Drive folder** (the 10 ARMD CSVs). Never committed to git.
- Each person points `ARMD_DIR` at their own mount of that shared folder. The code is identical
  for everyone — `config.py` resolves the path (env var → Colab Drive mount → local `data/raw/`).

---

## Setup A — Local (Mac/Linux/WSL, for development with Claude Code)

```bash
git clone https://github.com/<your-org>/capstone.git
cd capstone

python3.11 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# point at YOUR mount of the shared ARMD Drive folder (add to ~/.zshrc to persist)
export ARMD_DIR="/Users/<you>/Library/CloudStorage/GoogleDrive-.../ARMD"
```

Then run `notebooks/00_setup_check.ipynb` — it confirms imports work and finds all 10 CSVs.

## Setup B — Colab (for a teammate, or GPU training)

Open `notebooks/00_setup_check.ipynb` in Colab and run it top to bottom. The first cell:
clones the repo, installs deps (torch is already there with CUDA), mounts Drive, and sets
`ARMD_DIR` to the shared folder. Then the same checks run. Enable GPU via
Runtime → Change runtime type → GPU when you reach training.

---

## Workflow
Author in **Claude Code** (local) → push to **GitHub** → teammates / Colab pull and run.
Notebooks import modules and show output; all pipeline logic lives in `src/amr_fed/`.

## Structure
```
src/amr_fed/   core modules (config.py has all locked decisions)
notebooks/     00_setup_check.ipynb, then EDA + experiments
data/raw/      local fallback for ARMD CSVs (gitignored)
tests/
CLAUDE.md      project context for Claude Code
```

See `CLAUDE.md` for the locked schema and the phase plan.
