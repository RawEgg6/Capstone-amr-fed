"""Shared task code for the Flower FedAvg simulation (Phase 3).

Each simulated hospital is a patient-subset **core** graph. Two things must be
identical across clients for weight averaging to work: the feature dims (fixed by
graph_build) and the model structure. We pin the model to a canonical edge-type
set (padding missing edges as empty) so every client's state_dict lines up.

Flower's Ray simulation runs each client in its own process, so we hand graphs off
via disk (fixed path) rather than in-memory globals.
"""
from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import config
from ..data_loader import PK
from ..graph_build import build_arrays, to_hetero_data
from ..model import AMRSAGE
from ..train_local import _auroc, _macro_f1

# Canonical core edge types — exactly what to_hetero_data emits for the 3 core edges
# (including reverse edges). Pinned so every client's model has identical submodules.
CANONICAL_EDGES = [
    ("organism", "tested", "antibiotic"),
    ("antibiotic", "rev_tested", "organism"),
    ("patient", "grew", "organism"),
    ("organism", "rev_grew", "patient"),
    ("organism", "known_resistant", "antibiotic"),
    ("antibiotic", "rev_known_resistant", "organism"),
]

CLIENTS_DIR = Path(tempfile.gettempdir()) / "amr_fed_clients"
FED_HISTORY = CLIENTS_DIR / "fed_history.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- federated metric history (flwr 1.23 run_simulation returns None, so the
#      strategy appends each round's aggregated macro-F1 here for run.py to read) ----
def reset_fed_history() -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    FED_HISTORY.write_text("")


def append_fed_metric(macro_f1: float, per_client: dict | None = None,
                      auroc: float | None = None, per_client_auroc: dict | None = None) -> None:
    with open(FED_HISTORY, "a") as f:
        f.write(json.dumps({"macro_f1": macro_f1, "per_client": per_client or {},
                            "auroc": auroc, "per_client_auroc": per_client_auroc or {}}) + "\n")


def read_fed_history() -> list[float]:
    if not FED_HISTORY.exists():
        return []
    return [json.loads(x)["macro_f1"] for x in FED_HISTORY.read_text().splitlines() if x.strip()]


def read_fed_records() -> list[dict]:
    """Full per-round records ({'macro_f1', 'per_client': {cid: f1}}) — for the
    per-hospital / worst-client breakdown at the best round."""
    if not FED_HISTORY.exists():
        return []
    return [json.loads(x) for x in FED_HISTORY.read_text().splitlines() if x.strip()]


# ---- disk hand-off (Ray clients are separate processes) --------------------
def _client_path(cid: int) -> Path:
    return CLIENTS_DIR / f"client_{cid}.pt"


def write_run_config(cfg: dict) -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    (CLIENTS_DIR / "run_config.json").write_text(json.dumps(cfg))


def read_run_config() -> dict:
    return json.loads((CLIENTS_DIR / "run_config.json").read_text())


def save_client_graph(cid: int, data) -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, _client_path(cid))


def load_client_graph(cid: int):
    # weights_only=False is required for PyG HeteroData (not a plain tensor dict).
    # Safe here: the file is a self-produced local artifact written by this same run
    # to a temp dir (build_and_save_clients), never untrusted external input.
    return torch.load(_client_path(cid), weights_only=False)


def _pad_canonical_edges(data):
    for (s, r, d) in CANONICAL_EDGES:
        if (s, r, d) not in data.edge_index_dict:
            data[s, r, d].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data


def build_and_save_clients(df, assignment, n_clients: int, seed: int = config.SEED,
                           patient_history: bool = True) -> list[int]:
    """Build one core graph per hospital (patient subset), pad edges, save to disk.
    patient_history adds the per-test prior-resistance decoder features (the Phase-1
    winning feature) to every client. Returns per-client patient counts."""
    sizes = []
    for c in range(n_clients):
        sub = df[df[PK].map(assignment) == c]
        data = _pad_canonical_edges(to_hetero_data(
            build_arrays(sub, seed=seed, patient_history=patient_history)))
        save_client_graph(c, data)
        sizes.append(int(data["patient"].num_nodes))
    return sizes


# ---- model + train/eval ----------------------------------------------------
def _tf_dim(data) -> int:
    tf = getattr(data, "triple_feat", None)
    return tf.shape[1] if tf is not None else 0


def make_model(cfg: dict, triple_feat_dim: int = 0):
    return AMRSAGE(CANONICAL_EDGES, hidden=cfg["hidden"], layers=cfg["layers"],
                   aggr=cfg["aggr"], triple_feat_dim=triple_feat_dim).to(DEVICE)


def init_model_on(data, cfg: dict):
    """Build a model and materialise its lazy params via one forward pass.
    triple_feat_dim is read from the graph so every client's decoder matches (must be
    identical across clients for FedAvg weight aggregation — guaranteed since all clients
    use the same patient_history setting)."""
    data = data.to(DEVICE)
    model = make_model(cfg, _tf_dim(data))
    with torch.no_grad():
        model.encode(data.x_dict, data.edge_index_dict)
    return model


def free_gpu() -> None:
    """Release cached GPU memory. Ray reuses each client actor across rounds, so without
    this the per-round activation cache accumulates and the big client (e.g. the ~38k-patient
    organism hospital) OOMs when co-scheduled with another. Cheap; no-op on CPU."""
    if DEVICE == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def get_weights(model) -> list[np.ndarray]:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights) -> None:
    sd = OrderedDict((k, torch.as_tensor(w)) for k, w in zip(model.state_dict().keys(), weights))
    model.load_state_dict(sd, strict=True)


def local_train(model, data, epochs: int, lr: float = 1e-3, weight_decay: float = 1e-4) -> float:
    data = data.to(DEVICE)
    tr = data.train_mask.to(DEVICE)
    tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
    tf = getattr(data, "triple_feat", None)
    if tf is not None:
        tf = tf.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.train_pos_weight.to(DEVICE))
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, tri[:, tr],
                    tf[tr] if tf is not None else None)
        loss = loss_fn(out, y[tr])
        loss.backward()
        opt.step()
    out = float(loss.item())
    free_gpu()  # drop this round's activation cache so a reused actor doesn't grow unbounded
    return out


@torch.no_grad()
def local_eval(model, data, mask_name: str = "test_mask") -> tuple[float, float, int]:
    """Returns (macro_f1, auroc, n) on the masked triples. AUROC is NaN if the masked
    set is single-class (undefined)."""
    data = data.to(DEVICE)
    mask = getattr(data, mask_name).to(DEVICE)
    tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
    tf = getattr(data, "triple_feat", None)
    if tf is not None:
        tf = tf.to(DEVICE)
    model.eval()
    if int(mask.sum()) == 0:
        return 0.0, float("nan"), 0
    logits = model(data.x_dict, data.edge_index_dict, tri[:, mask],
                   tf[mask] if tf is not None else None)
    result = _macro_f1(y[mask], logits), _auroc(y[mask], logits), int(mask.sum())
    free_gpu()
    return result
