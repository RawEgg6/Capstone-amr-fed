"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.

TKPA-FL additions (round 1 only):
  - Always returns partition_id in fit() metrics.
  - Returns a serialised topology profile when the server sets send_profile=True.
"""
from __future__ import annotations

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import (
    get_weights, init_model_on, load_client_graph, local_eval, local_train,
    read_run_config, set_weights,
)
from ..topology import build_hospital_profile, serialize_profile


class FlowerClient(NumPyClient):
    def __init__(self, data, cfg: dict, partition_id: int):
        self.data = data
        self.cfg = cfg
        self.partition_id = partition_id
        self.model = init_model_on(data, cfg)   # materialises lazy params

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        local_train(self.model, self.data, self.cfg["local_epochs"])
        weights = get_weights(self.model)
        num_examples = int(self.data.train_mask.sum())

        metrics = {"partition_id": self.partition_id}

        # Round 1: server requests the topology profile
        if config.get("send_profile", False):
            profile = build_hospital_profile(self.data)
            metrics["profile"] = serialize_profile(profile)

        return weights, num_examples, metrics

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        f1, n = local_eval(self.model, self.data, "test_mask")
        return 0.0, n, {"macro_f1": f1, "partition_id": self.partition_id}


def client_fn(context: Context):
    cfg = read_run_config()
    pid = int(context.node_config["partition-id"])
    return FlowerClient(load_client_graph(pid), cfg, pid).to_client()


app = ClientApp(client_fn=client_fn)
