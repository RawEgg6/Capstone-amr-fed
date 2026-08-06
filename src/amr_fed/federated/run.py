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
import pandas as pd

from .. import config
from ..data_loader import PK, load_cohort_frame
from ..partition import dirichlet_ward_mixture
from . import client_app, server_app
from .task import (
    DEVICE, build_and_save_clients, init_model_on, load_client_graph,
    local_eval, local_train, read_fed_history, read_fed_records, reset_fed_history,
    write_run_config,
)

POOLED_REFERENCE = 0.71  # Phase-1 pooled macro-F1 with patient-history features (all data)


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
               local_epochs: int = 6, local_only_epochs: int = 60,
               seed: int = config.SEED, patient_history: bool = True, df=None,
               partition_fn=None, label: str | None = None):
    """partition_fn(df, seed) -> patient->client Series lets us swap the split
    (ward-Dirichlet default, or label_dirichlet / specimen_baseline). Any client
    labels (str/int) are normalised to contiguous 0..k-1 ids and n_clients is
    derived from the split, so specimen (3-4 hospitals) works without extra args."""
    import torch
    from flwr.simulation import run_simulation

    torch.manual_seed(seed)
    df = load_cohort_frame() if df is None else df
    raw = (dirichlet_ward_mixture(df, n_clients=n_clients, alpha=alpha, seed=seed)
           if partition_fn is None else partition_fn(df, seed))
    codes, _ = pd.factorize(raw)                 # str/int client labels -> 0..k-1
    assign = pd.Series(codes, index=raw.index)
    n_clients = int(assign.max()) + 1
    tag = label or f"alpha={alpha}"
    cfg = _run_config(n_clients, rounds, local_epochs)
    write_run_config(cfg)
    sizes = build_and_save_clients(df, assign, n_clients, seed=seed, patient_history=patient_history)
    print(f"{tag} | {n_clients} hospitals | patients each: {sizes}")

    lo_f1s, lo_avg = run_local_only(n_clients, cfg, epochs=local_only_epochs)
    print(f"LOCAL-ONLY per-hospital macro-F1: {lo_f1s} | weighted avg = {lo_avg:.4f}")

    ngpu = (0.9 / n_clients) if DEVICE == "cuda" else 0.0
    reset_fed_history()
    run_simulation(  # returns None in flwr 1.23; the strategy logs per-round F1 to disk
        server_app=server_app.app,
        client_app=client_app.app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": ngpu}},
    )
    records = read_fed_records()
    fed = [r["macro_f1"] for r in records]
    print("FEDAVG macro-F1 by round:", [round(v, 4) for v in fed])
    best_idx = max(range(len(fed)), key=fed.__getitem__) if fed else None
    fed_best = round(fed[best_idx], 4) if fed else None  # best round (FedAvg drifts on non-IID)
    fed_final = round(fed[-1], 4) if fed else None
    # per-hospital FedAvg F1 at the best round (global model, each client's own test set)
    fed_pc = records[best_idx].get("per_client", {}) if fed else {}
    fed_f1s = [fed_pc.get(str(c)) for c in range(n_clients)]

    print(f"\n=== Phase 3 comparison ({tag}) ===")
    print(f"  pooled (Phase 1, all data)     : {POOLED_REFERENCE}")
    print(f"  FedAvg  best-round / final     : {fed_best} / {fed_final}")
    print(f"  local-only (alone, weighted)   : {lo_avg:.4f}")
    if fed_best is not None:
        print(f"  => FedAvg (best) {'BEATS' if fed_best > lo_avg else 'does NOT beat'} local-only")
    # per-hospital breakdown + worst-client (the fairness story: FedAvg helps small/weak sites most)
    print("  per-hospital  (n | local-only -> FedAvg | delta):")
    for c in range(n_clients):
        fv = fed_f1s[c]
        delta = f"{fv - lo_f1s[c]:+.4f}" if fv is not None else "   n/a"
        fvs = f"{fv:.4f}" if fv is not None else " n/a "
        print(f"    H{c}: n={sizes[c]:>6} | {lo_f1s[c]:.4f} -> {fvs} | {delta}")
    worst_local = round(min(lo_f1s), 4)
    fed_ok = [v for v in fed_f1s if v is not None]
    worst_fed = round(min(fed_ok), 4) if fed_ok else None
    if worst_fed is not None:
        print(f"  worst-hospital: local {worst_local:.4f} -> FedAvg {worst_fed:.4f} "
              f"({worst_fed - worst_local:+.4f})")
    return {"alpha": alpha, "pooled": POOLED_REFERENCE, "fedavg_best": fed_best,
            "fedavg_final": fed_final, "local_only": lo_avg,
            "local_only_per_client": lo_f1s, "fedavg_per_client": fed_f1s,
            "worst_local": worst_local, "worst_fed": worst_fed,
            "sizes": sizes, "fed_by_round": fed}


def run_multiseed(alpha: float = 0.5, n_clients: int = 5, rounds: int = 10,
                  local_epochs: int = 6, seeds=(42, 43, 44), patient_history: bool = True,
                  df=None, partition_fn=None, label: str | None = None):
    """Run FedAvg over several seeds; report mean +/- std to denoise the partition +
    training randomness. partition_fn(df, seed) swaps the split (default ward-Dirichlet
    at `alpha`; pass label_dirichlet / specimen_baseline for the non-IID settings).
    Loads the cohort ONCE and reuses it."""
    df = load_cohort_frame() if df is None else df
    tag = label or f"alpha={alpha}"
    runs = []
    for s in seeds:
        print(f"\n########## {tag}  seed={s} ##########")
        runs.append(run_fedavg(alpha=alpha, n_clients=n_clients, rounds=rounds,
                               local_epochs=local_epochs, seed=s,
                               patient_history=patient_history, df=df,
                               partition_fn=partition_fn, label=label))

    def ms(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    gains = [r["fedavg_best"] - r["local_only"] for r in runs
             if r["fedavg_best"] is not None]
    lo, fb, ff = ms("local_only"), ms("fedavg_best"), ms("fedavg_final")
    wl, wf = ms("worst_local"), ms("worst_fed")   # worst-hospital (fairness) metric
    gmean, gstd = round(float(np.mean(gains)), 4), round(float(np.std(gains)), 4)
    worst_gains = [r["worst_fed"] - r["worst_local"] for r in runs
                   if r.get("worst_fed") is not None]
    wgmean = round(float(np.mean(worst_gains)), 4) if worst_gains else None
    wgstd = round(float(np.std(worst_gains)), 4) if worst_gains else None

    print(f"\n=== MULTI-SEED SUMMARY: {tag}, {len(seeds)} seeds ===")
    print(f"  local-only         : {lo[0]} +/- {lo[1]}")
    print(f"  FedAvg best-round  : {fb[0]} +/- {fb[1]}")
    print(f"  FedAvg final       : {ff[0]} +/- {ff[1]}")
    print(f"  gain (best-local)  : {gmean} +/- {gstd}   (pooled ref = {POOLED_REFERENCE})")
    print(f"  worst-hosp local   : {wl[0]} +/- {wl[1]}")
    print(f"  worst-hosp FedAvg  : {wf[0]} +/- {wf[1]}")
    print(f"  worst-hosp gain    : {wgmean} +/- {wgstd}")
    return {"alpha": alpha, "seeds": list(seeds), "local_only": lo, "fedavg_best": fb,
            "fedavg_final": ff, "gain_mean": gmean, "gain_std": gstd,
            "worst_local": wl, "worst_fed": wf,
            "worst_gain_mean": wgmean, "worst_gain_std": wgstd, "runs": runs}
