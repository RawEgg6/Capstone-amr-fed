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
import random
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
from .selection import THRESHOLDS, confusion_counts, macro_f1_grid, prob_hist

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

# Contract for topology-aware aggregation (Phase 5): the client's fit() metrics carry
# its fingerprint — a JSON list of floats in topology.FEATURE_NAMES order — under this
# key. The server strategy reads it back; missing/malformed values mean FedAvg fallback.
FIT_METRIC_KEY = "topology_fingerprint"


# ---- federated metric history (flwr 1.23 run_simulation returns None, so the
#      strategy appends each round's aggregated macro-F1 here for run.py to read) ----
def reset_fed_history() -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    FED_HISTORY.write_text("")


def append_fed_metric(macro_f1: float, per_client: dict | None = None,
                      auroc: float | None = None, per_client_auroc: dict | None = None,
                      extra: dict | None = None) -> None:
    """`extra` carries the validation/test cutoff grids (selection.grid_record); the legacy
    fields (test macro-F1 @0.5, AUROC) are unchanged so old readers keep working."""
    rec = {"macro_f1": macro_f1, "per_client": per_client or {},
           "auroc": auroc, "per_client_auroc": per_client_auroc or {}}
    if extra:
        rec.update(extra)
    with open(FED_HISTORY, "a") as f:
        f.write(json.dumps(rec) + "\n")


SIMFT_ROUNDS = CLIENTS_DIR / "simft_rounds.jsonl"


def reset_simft_files() -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    SIMFT_ROUNDS.write_text("")
    for f in CLIENTS_DIR.glob("simft_personal_*.npz"):
        f.unlink()


def append_simft_round(rec: dict) -> None:
    with open(SIMFT_ROUNDS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_simft_rounds() -> list[dict]:
    if not SIMFT_ROUNDS.exists():
        return []
    return [json.loads(x) for x in SIMFT_ROUNDS.read_text().splitlines() if x.strip()]


def save_personal_params(cid: int, weights) -> None:
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CLIENTS_DIR / f"simft_personal_{cid}.npz", *weights)


def load_personal_params(cid: int) -> list[np.ndarray]:
    z = np.load(CLIENTS_DIR / f"simft_personal_{cid}.npz")
    return [z[f"arr_{i}"] for i in range(len(z.files))]


GLOBAL_PARAMS = CLIENTS_DIR / "global_final.npz"


def save_global_params(weights) -> None:
    """Persist the server's latest global model (overwritten every round) so the driver can
    fine-tune it per hospital after the simulation ends."""
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(GLOBAL_PARAMS, *weights)


def load_global_params() -> list[np.ndarray]:
    z = np.load(GLOBAL_PARAMS)
    return [z[f"arr_{i}"] for i in range(len(z.files))]


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


def build_graph(sub, seed: int, patient_history: bool = True, split_code: dict | None = None):
    """One hospital's padded core graph from its frame (optionally with a precomputed split code)."""
    return _pad_canonical_edges(to_hetero_data(
        build_arrays(sub, seed=seed, patient_history=patient_history, split_code=split_code)))


def build_and_save_clients(df, assignment, n_clients: int, seed: int = config.SEED,
                           patient_history: bool = True, train_frac: float = 1.0) -> list[int]:
    """Build one core graph per hospital (patient subset), pad edges, save to disk.
    patient_history adds the per-test prior-resistance decoder features (the Phase-1
    winning feature) to every client. ``train_frac`` < 1 keeps only that fraction of every
    hospital's TRAIN and VALIDATION patients (test patients are untouched, so the test set is
    identical at every data level; see regimes.subsample_frame). Returns per-client patient counts."""
    from .regimes import subsample_frame
    sizes = []
    for c in range(n_clients):
        sub = df[df[PK].map(assignment) == c]
        code = None
        if train_frac < 1.0:
            sub, code = subsample_frame(sub, seed=seed, cid=c, frac=train_frac)
        data = build_graph(sub, seed, patient_history, code)
        save_client_graph(c, data)
        sizes.append(int(data["patient"].num_nodes))
    return sizes


# ---- model + train/eval ----------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch **in the current process**.

    Reproducibility fix (2026): `run_fedavg` seeds the parent process, but Flower's Ray
    simulation runs each client — and the server — in their own processes with fresh RNGs,
    so the parent seed never reaches them. Each process must seed itself, otherwise client
    models are initialised with different random weights every run and FedAvg is
    non-deterministic run-to-run (the ±0.007 noise that sat above the gate's 0.01 threshold).
    """
    random.seed(seed)
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def client_seed(cfg: dict, cid: int) -> int:
    """Deterministic per-client seed: distinct per hospital, identical across runs."""
    return int(cfg.get("seed", config.SEED)) + int(cid)


def seed_client(cfg: dict, cid: int) -> None:
    """Seed a client process from its (seed + cid). Call before building the model."""
    seed_everything(client_seed(cfg, cid))


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
    # Materialise the lazy conv weights under eval(): this forward exists only to
    # build shapes, and a train-mode pass would draw dropout masks for nothing.
    model.eval()
    with torch.no_grad():
        model.encode(data.x_dict, data.edge_index_dict)
    model.train()
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


def local_train(model, data, epochs: int, lr: float = 1e-3, weight_decay: float = 1e-4,
                proximal_mu: float = 0.0, global_weights=None) -> float:
    """Local epochs of class-weighted BCE. proximal_mu > 0 adds the FedProx term
    (mu/2)*||w - w_global||^2 to every step's loss, where global_weights is the
    server's parameter list (get_weights format, numpy) captured before training.
    Defaults (mu=0, no global_weights) are bit-identical to the pre-FedProx trainer."""
    data = data.to(DEVICE)
    tr = data.train_mask.to(DEVICE)
    tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
    tf = getattr(data, "triple_feat", None)
    if tf is not None:
        tf = tf.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.train_pos_weight.to(DEVICE))
    # FedProx: pin the reference ONCE before the epoch loop (the global model does
    # not move while this client trains).
    global_tensors = None
    if proximal_mu > 0.0 and global_weights is not None:
        global_tensors = [torch.as_tensor(w, dtype=p.dtype, device=p.device)
                          for w, p in zip(global_weights, model.parameters())]
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, tri[:, tr],
                    tf[tr] if tf is not None else None)
        loss = loss_fn(out, y[tr])
        if global_tensors is not None:
            prox = sum((p - g).pow(2).sum()
                       for p, g in zip(model.parameters(), global_tensors))
            loss = loss + (proximal_mu / 2.0) * prox
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


def _val_mask(data, half: str | None):
    """Validation mask, optionally one half of the validation PATIENTS (parity of the patient node id):
    half "A" is used to score other hospitals' models (SimFT weights), half "B" to select round / cutoff /
    epoch, so the weights never see the rows that choose the reported number. None = all validation rows."""
    mask = data.val_mask
    if half is None:
        return mask
    parity = (data.triple_index[0] % 2) == 0
    return mask & (parity if half == "A" else ~parity)


@torch.no_grad()
def cross_eval_losses(models, cfg: dict, half: str = "A") -> np.ndarray:
    """L[i, j] = class-weighted BCE of model j on hospital i's validation half ``half`` (NaN when that half
    is empty). One hospital graph is in memory at a time. In a deployment each hospital would score the
    received models itself and return only these numbers."""
    n = len(models)
    L = np.full((n, n), np.nan)
    for i in range(n):
        data = load_client_graph(i).to(DEVICE)
        mask = _val_mask(data, half).to(DEVICE)
        if int(mask.sum()) > 0:
            tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
            tf = getattr(data, "triple_feat", None)
            tf = tf.to(DEVICE) if tf is not None else None
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.train_pos_weight.to(DEVICE))
            model = init_model_on(data, cfg)
            model.eval()
            for j in range(n):
                set_weights(model, models[j])
                out = model(data.x_dict, data.edge_index_dict, tri[:, mask], tf[mask] if tf is not None else None)
                L[i, j] = float(loss_fn(out, y[mask]))
            del model
        del data
        free_gpu()
    return L


@torch.no_grad()
def local_eval_full(model, data, val_half: str | None = None) -> dict:
    """Score the model on this hospital's VAL and TEST rows in one pass.

    Returns n_test / n_val, the legacy test macro-F1 @0.5 and AUROC (identical to
    `local_eval`), and the macro-F1 of both splits at every cutoff in selection.THRESHOLDS
    (`val_grid` / `test_grid`). The cutoff/round are chosen from the val grid only."""
    data = data.to(DEVICE)
    tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
    tf = getattr(data, "triple_feat", None)
    if tf is not None:
        tf = tf.to(DEVICE)
    model.eval()
    out = {}
    for name in ("val", "test"):
        mask = (_val_mask(data, val_half) if name == "val" else data.test_mask).to(DEVICE)
        n = int(mask.sum())
        out[f"n_{name}"] = n
        if n == 0:
            out[f"{name}_grid"] = [float("nan")] * len(THRESHOLDS)
            out[f"{name}_cm"] = [[0, 0, 0, 0]] * len(THRESHOLDS)
            out[f"{name}_hist"] = [[0] * 20, [0] * 20]
            out[f"{name}_auroc"] = float("nan")
            if name == "test":
                out["macro_f1"], out["auroc"] = 0.0, float("nan")
            continue
        logits = model(data.x_dict, data.edge_index_dict, tri[:, mask],
                       tf[mask] if tf is not None else None)
        p = torch.sigmoid(logits).double().cpu().numpy()
        yn = y[mask].cpu().numpy()
        out[f"{name}_grid"] = macro_f1_grid(yn, p).tolist()
        # raw detail for later analysis: confusion counts at every cutoff + probability histograms
        out[f"{name}_cm"] = confusion_counts(yn, p).tolist()
        out[f"{name}_hist"] = prob_hist(yn, p)
        out[f"{name}_auroc"] = _auroc(y[mask], logits)
        if name == "test":
            out["macro_f1"], out["auroc"] = _macro_f1(y[mask], logits), out["test_auroc"]
    free_gpu()
    return out


def local_train_tracked(model, data, epochs: int, eval_every: int, lr: float = 1e-3,
                        weight_decay: float = 1e-4, eval_at_start: bool = False,
                        val_half: str | None = None) -> list[dict]:
    """`local_train` (same optimiser, same one-full-batch-step-per-epoch loop) that also scores
    the model on val+test every `eval_every` epochs and at the last epoch. Evaluation runs in
    eval() mode under no_grad, so it consumes no RNG and leaves training bit-identical to
    `local_train`. Returns [{"epoch": e, **local_eval_full(...)}, ...]."""
    data = data.to(DEVICE)
    tr = data.train_mask.to(DEVICE)
    tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
    tf = getattr(data, "triple_feat", None)
    if tf is not None:
        tf = tf.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.train_pos_weight.to(DEVICE))
    ckpts = sorted(set(range(eval_every, epochs + 1, eval_every)) | {epochs})
    history = []
    if eval_at_start:   # epoch 0 = the model as handed in (used by the fine-tune baseline)
        history.append({"epoch": 0, **local_eval_full(model, data, val_half)})
    model.train()
    for e in range(1, epochs + 1):
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, tri[:, tr],
                    tf[tr] if tf is not None else None)
        loss_fn(out, y[tr]).backward()
        opt.step()
        if e in ckpts:
            history.append({"epoch": e, **local_eval_full(model, data, val_half)})
            model.train()
    free_gpu()
    return history
