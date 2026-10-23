"""Flower ServerApp — plain FedAvg (Phase 3 baseline).

Aggregates client weights by data size (FedAvg) and reports a test-weighted
macro-F1 across hospitals each round. Initial weights come from a model
materialised on client 0's graph (all clients share the same param shapes).

Phase 5 will swap FedAvg for a topology-aware Strategy subclass here.
"""
from __future__ import annotations

import os

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .strategy import TopologyAwareStrategy
from .task import (
    append_fed_metric, get_weights, init_model_on, load_client_graph, read_run_config,
    seed_everything,
)


def _weighted_metrics(metrics: list) -> dict:
    """metrics: list of (num_examples, {"macro_f1", "auroc", "cid"}) -> test-weighted means.
    Appends the round's aggregates AND per-hospital F1/AUROC to the history file
    (run_simulation returns None), so run.py can report the worst-client gap. AUROC is
    averaged only over hospitals where it is finite (single-class clients report NaN)."""
    total = sum(n for n, _ in metrics)
    f1 = sum(n * m["macro_f1"] for n, m in metrics) / total if total else 0.0
    # nan-safe weighted AUROC (a == a is False for NaN)
    auc_pairs = [(n, m["auroc"]) for n, m in metrics if m.get("auroc") is not None and m["auroc"] == m["auroc"]]
    auc = sum(n * a for n, a in auc_pairs) / sum(n for n, _ in auc_pairs) if auc_pairs else float("nan")
    per_client = {str(m.get("cid", i)): round(m["macro_f1"], 4) for i, (_, m) in enumerate(metrics)}
    per_client_auroc = {str(m.get("cid", i)): (round(m["auroc"], 4) if m.get("auroc") == m.get("auroc") else None)
                        for i, (_, m) in enumerate(metrics)}
    append_fed_metric(f1, per_client, auroc=auc, per_client_auroc=per_client_auroc)
    return {"macro_f1": f1, "auroc": auc}


def server_fn(context: Context):
    cfg = read_run_config()
    # Seed the server process so the initial parameters are deterministic. Without this,
    # the server (a separate process from the driver) inits weights with a fresh RNG and
    # every FedAvg run starts from different weights (see task.seed_everything).
    seed_everything(int(cfg.get("seed", 42)))
    dbg = os.environ.get("AMR_DEBUG_INIT", "").strip().lower() not in ("", "0", "false", "no")
    if dbg:
        print(f"[server-init] pid={os.getpid()} "
              f"vis={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"env AMR_RNG_PROBE={os.environ.get('AMR_RNG_PROBE')}", flush=True)
    init_weights = get_weights(init_model_on(load_client_graph(0), cfg))
    if dbg:
        # real-RNG init (no zero-init wrap here): "model OK" = the server side built
        # its weights fine; compare pid with the notebook's "driver pid" print —
        # same pid means the server runs in the driver process, not a Ray actor.
        print("[server-init] model OK (real RNG, no wrap)", flush=True)
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=cfg["n_clients"],
        initial_parameters=ndarrays_to_parameters(init_weights),
        evaluate_metrics_aggregation_fn=_weighted_metrics,
    )
    # Phase 5: "topology" swaps plain FedAvg for fingerprint-similarity mixing.
    # Phase 6 baselines (2026-10-23): "fedmedian" / "fedtrimmedavg" swap the
    # aggregation rule for Flower's robust aggregators; "fedprox" keeps a FedAvg
    # server but ships proximal_mu to every client's fit config (the proximal term
    # itself lives in task.local_train). Everything else (sampling, evaluation,
    # initial weights) is identical, so the comparison is exactly apples-to-apples.
    # Default stays plain FedAvg.
    strategy_name = cfg.get("strategy", "fedavg")
    if strategy_name == "topology":
        strategy = TopologyAwareStrategy(
            temperature=float(cfg.get("topo_temperature", 1.0)),
            mode=cfg.get("topo_mode", "distinctiveness"),
            **common,
        )
    elif strategy_name == "fedmedian":
        from flwr.server.strategy import FedMedian
        strategy = FedMedian(**common)
    elif strategy_name == "fedtrimmedavg":
        from flwr.server.strategy import FedTrimmedAvg
        strategy = FedTrimmedAvg(beta=float(cfg.get("trimmed_beta", 0.1)), **common)
    elif strategy_name == "fedprox":
        # Flower's FedProx injects proximal_mu into the fit config itself; fall back
        # to FedAvg + an on_fit_config_fn that does the same if it's unavailable.
        mu = float(cfg.get("proximal_mu", 0.01))
        try:
            from flwr.server.strategy import FedProx
            strategy = FedProx(proximal_mu=mu, **common)
        except ImportError:
            common["on_fit_config_fn"] = lambda rnd: {"proximal_mu": mu, "round": rnd}
            strategy = FedAvg(**common)
    else:
        strategy = FedAvg(**common)
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg["rounds"]))


app = ServerApp(server_fn=server_fn)
