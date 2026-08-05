"""Flower ServerApp — plain FedAvg (Phase 3 baseline).

Aggregates client weights by data size (FedAvg) and reports a test-weighted
macro-F1 across hospitals each round. Initial weights come from a model
materialised on client 0's graph (all clients share the same param shapes).

Phase 5 will swap FedAvg for a topology-aware Strategy subclass here.
"""
from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .task import get_weights, init_model_on, load_client_graph, read_run_config


def _weighted_macro_f1(metrics: list) -> dict:
    """metrics: list of (num_examples, {"macro_f1": ...}) -> test-weighted mean."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"macro_f1": 0.0}
    return {"macro_f1": sum(n * m["macro_f1"] for n, m in metrics) / total}


def server_fn(context: Context):
    cfg = read_run_config()
    init_weights = get_weights(init_model_on(load_client_graph(0), cfg))
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=cfg["n_clients"],
        initial_parameters=ndarrays_to_parameters(init_weights),
        evaluate_metrics_aggregation_fn=_weighted_macro_f1,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg["rounds"]))


app = ServerApp(server_fn=server_fn)
