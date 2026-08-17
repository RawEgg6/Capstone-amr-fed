"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.

Seeding note: Flower/Ray spawns each client in its own subprocess. The
``torch.manual_seed`` call in run_fedavg only seeds the *parent* process, so
lazy model params and any stochastic ops inside client processes are
non-deterministic across runs. ``_seed_client`` below re-seeds every relevant
RNG at the start of ``client_fn`` using the global seed stored in run_config,
mixed with the partition-id so clients still have independent randomness while
being fully reproducible run-to-run.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import (
    get_weights, init_model_on, load_client_graph, local_eval, local_train,
    read_run_config, set_weights,
)


def _seed_client(base_seed: int, partition_id: int) -> None:
    """Seed every relevant RNG in this client subprocess.

    We XOR the base seed with the partition-id so clients are independent
    (different random shuffles within local_train etc.) while still being
    fully reproducible across runs with the same base seed.
    """
    seed = int(base_seed) ^ (int(partition_id) * 2654435761)  # Knuth hash, stays positive
    seed = seed & 0xFFFF_FFFF  # keep it 32-bit so all seeders accept it
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlowerClient(NumPyClient):
    def __init__(self, data, cfg: dict, cid: int):
        self.data = data
        self.cfg = cfg
        self.cid = cid
        self.model = init_model_on(data, cfg)   # materialises lazy params

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        local_train(self.model, self.data, self.cfg["local_epochs"])
        return get_weights(self.model), int(self.data.train_mask.sum()), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        f1, auc, n = local_eval(self.model, self.data, "test_mask")
        # cid -> per-hospital breakdown; auroc alongside macro_f1
        return 0.0, n, {"macro_f1": f1, "auroc": auc, "cid": self.cid}


def client_fn(context: Context):
    cfg = read_run_config()
    pid = int(context.node_config["partition-id"])
    # Seed this subprocess BEFORE building the model so lazy-param initialisation
    # and all subsequent ops are deterministic for this (seed, partition) pair.
    _seed_client(cfg.get("seed", 42), pid)
    return FlowerClient(load_client_graph(pid), cfg, pid).to_client()


app = ClientApp(client_fn=client_fn)
