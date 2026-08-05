"""Phase 3 driver: local-only baseline + Flower FedAvg, compared to pooled (Phase 1).

The three numbers this produces, at a given alpha:
  pooled      -- Phase-1 model on all data (the "everyone shares data" ceiling)
  local-only  -- each hospital trains alone (weighted-avg macro-F1)
  FedAvg      -- hospitals train locally, server averages weights each round

FedAvg's per-round local epochs x rounds is set equal to local-only's epoch budget
so the comparison is fair. Run on Colab (needs torch + flwr[simulation]).
"""
from __future__ import annotations

import numpy as np

from ..data_loader import PK, load_cohort_frame
from ..partition import dirichlet_ward_mixture
from . import client_app, server_app
from .task import (
    DEVICE, build_and_save_clients, init_model_on, load_client_graph,
    local_eval, local_train, write_run_config,
)

POOLED_REFERENCE = 0.663  # Phase-1 core macro-F1 (all data, one model)


def _run_config(n_clients: int, rounds: int, local_epochs: int) -> dict:
    # canonical Phase-1 architecture (best clean config from the grid)
    return {"n_clients": n_clients, "rounds": rounds, "local_epochs": local_epochs,
            "hidden": 128, "layers": 2, "aggr": "mean"}


def run_local_only(n_clients: int, cfg: dict, epochs: int = 60):
    """Train each hospital's model on its own data alone; eval on its own test set."""
    f1s, ns = [], []
    for c in range(n_clients):
        data = load_client_graph(c)
        model = init_model_on(data, cfg)
        local_train(model, data, epochs)
        f1, n = local_eval(model, data, "test_mask")
        f1s.append(round(f1, 4))
        ns.append(n)
    wavg = float(np.average(f1s, weights=ns)) if sum(ns) else 0.0
    return f1s, wavg


def run_fedavg(alpha: float = 0.5, n_clients: int = 5, rounds: int = 10,
               local_epochs: int = 6, local_only_epochs: int = 60, df=None):
    from flwr.simulation import run_simulation

    df = load_cohort_frame() if df is None else df
    assign = dirichlet_ward_mixture(df, n_clients=n_clients, alpha=alpha)
    cfg = _run_config(n_clients, rounds, local_epochs)
    write_run_config(cfg)
    sizes = build_and_save_clients(df, assign, n_clients)
    print(f"alpha={alpha} | {n_clients} hospitals | patients each: {sizes}")

    lo_f1s, lo_avg = run_local_only(n_clients, cfg, epochs=local_only_epochs)
    print(f"LOCAL-ONLY per-hospital macro-F1: {lo_f1s} | weighted avg = {lo_avg:.4f}")

    ngpu = (0.9 / n_clients) if DEVICE == "cuda" else 0.0
    hist = run_simulation(
        server_app=server_app.app,
        client_app=client_app.app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": ngpu}},
    )
    fed = hist.metrics_distributed.get("macro_f1", [])
    print("FEDAVG macro-F1 by round:", [(r, round(v, 4)) for r, v in fed])
    fed_final = round(fed[-1][1], 4) if fed else None

    print(f"\n=== Phase 3 comparison (alpha={alpha}) ===")
    print(f"  pooled (Phase 1, all data)    : {POOLED_REFERENCE}")
    print(f"  FedAvg (federated)            : {fed_final}")
    print(f"  local-only (alone, weighted)  : {lo_avg:.4f}")
    if fed_final is not None:
        print(f"  => FedAvg {'BEATS' if fed_final > lo_avg else 'does NOT beat'} local-only")
    return {"alpha": alpha, "pooled": POOLED_REFERENCE, "fedavg": fed_final,
            "local_only": lo_avg, "local_only_per_client": lo_f1s, "sizes": sizes}
