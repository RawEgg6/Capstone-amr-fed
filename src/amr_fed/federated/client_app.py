"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.
"""
from __future__ import annotations

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import (
    get_weights, init_model_on, load_client_graph, local_eval, local_train,
    read_run_config, set_weights,
)


class FlowerClient(NumPyClient):
    def __init__(self, data, cfg: dict, cid: int):
        self.data = data
        self.cfg = cfg
        self.cid = cid
        self.model = init_model_on(data, cfg)

        # Compact topology fingerprint computed when the client graph was built.
        self.topology_homophily = float(data.topology_homophily)
        self.topology_hubness = float(data.topology_hubness)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        local_train(self.model, self.data, self.cfg["local_epochs"])

        metrics = {
            "topology_homophily": self.topology_homophily,
            "topology_hubness": self.topology_hubness,
            "cid": self.cid,
        }

        return (
            get_weights(self.model),
            int(self.data.train_mask.sum()),
            metrics,
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        f1, auc, n = local_eval(self.model, self.data, "test_mask")

        return (
            0.0,
            n,
            {
                "macro_f1": f1,
                "auroc": auc,
                "cid": self.cid,
                "topology_homophily": self.topology_homophily,
                "topology_hubness": self.topology_hubness,
            },
        )


def client_fn(context: Context):
    cfg = read_run_config()
    pid = int(context.node_config["partition-id"])
    return FlowerClient(load_client_graph(pid), cfg, pid).to_client()


app = ClientApp(client_fn=client_fn)
