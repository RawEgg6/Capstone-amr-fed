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
from ..partition import _call_partition, dirichlet_ward_mixture
from . import client_app, server_app
from .task import (
    DEVICE, build_and_save_clients, free_gpu, init_model_on, load_client_graph,
    local_eval, local_train, read_fed_history, read_fed_records, reset_fed_history,
    write_run_config,
)

POOLED_REFERENCE = 0.71  # Phase-1 pooled macro-F1 with patient-history features (all data)


def _wavg_finite(vals, weights) -> float:
    """Weighted mean over finite entries (skips None / NaN); NaN if none are finite.
    Used for AUROC, which is NaN for any single-class hospital."""
    pairs = [(v, w) for v, w in zip(vals, weights) if v is not None and v == v]  # v==v: False for NaN
    if not pairs:
        return float("nan")
    v, w = zip(*pairs)
    return float(np.average(v, weights=w))


def _run_config(n_clients: int, rounds: int, local_epochs: int) -> dict:
    # canonical Phase-1 architecture (best clean config from the grid)
    return {"n_clients": n_clients, "rounds": rounds, "local_epochs": local_epochs,
            "hidden": 128, "layers": 2, "aggr": "mean"}


def run_local_only(n_clients: int, cfg: dict, epochs: int = 60):
    """Train each hospital's model on its own data alone; eval on its own test set.
    Returns (f1_per_client, weighted_f1, auroc_per_client, weighted_auroc)."""
    f1s, aucs, ns = [], [], []
    for c in range(n_clients):
        data = load_client_graph(c)
        model = init_model_on(data, cfg)
        local_train(model, data, epochs)
        f1, auc, n = local_eval(model, data, "test_mask")
        f1s.append(round(f1, 4))
        aucs.append(round(auc, 4) if auc == auc else None)  # None for single-class client
        ns.append(n)
        del model, data
        free_gpu()   # release before the next (possibly huge) client's model
    wavg = float(np.average(f1s, weights=ns)) if sum(ns) else 0.0
    return f1s, wavg, aucs, _wavg_finite(aucs, ns)


def run_pooled(n_clients: int, cfg: dict, epochs: int = 60):
    """Centralized 'pooled' baseline, scored with the SAME protocol as FedAvg so the
    comparison is apples-to-apples: ONE model trained jointly on every hospital's train
    triples, then evaluated on each hospital's OWN test set and size-weighted (+ worst).

    Differs from FedAvg only in *how* it trains (joint full-batch GD on the union, no
    weight-averaging rounds) — same graphs, same train/test masks, same architecture, same
    per-hospital-averaged metric. Memory-safe: graphs live on CPU, one is moved to GPU at a
    time and gradients are accumulated across hospitals before each step."""
    import torch
    from torch import nn

    graphs = [load_client_graph(c) for c in range(n_clients)]  # keep on CPU
    model = init_model_on(graphs[0], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # global class weight (pooled sees the union, so pos_weight is the union's neg/pos)
    pos = sum(int(g.triple_label[g.train_mask].sum()) for g in graphs)
    neg = sum(int((g.triple_label[g.train_mask] == 0).sum()) for g in graphs)
    pos_weight = torch.tensor(neg / max(pos, 1), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n_tr = [int(g.train_mask.sum()) for g in graphs]
    total_tr = max(sum(n_tr), 1)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        for c in range(n_clients):
            data = graphs[c].to(DEVICE)
            tr = data.train_mask.to(DEVICE)
            tri, y = data.triple_index.to(DEVICE), data.triple_label.to(DEVICE).float()
            tf = getattr(data, "triple_feat", None)
            if tf is not None:
                tf = tf.to(DEVICE)
            out = model(data.x_dict, data.edge_index_dict, tri[:, tr],
                        tf[tr] if tf is not None else None)
            # size-weight each hospital's mean loss -> matches a per-example mean over the union
            loss = loss_fn(out, y[tr]) * (n_tr[c] / total_tr)
            loss.backward()  # accumulate grads across hospitals
            del data
            free_gpu()
        opt.step()

    f1s, aucs, ns = [], [], []
    for c in range(n_clients):
        f1, auc, n = local_eval(model, graphs[c], "test_mask")
        f1s.append(round(f1, 4))
        aucs.append(round(auc, 4) if auc == auc else None)
        ns.append(n)
        free_gpu()
    wavg = float(np.average(f1s, weights=ns)) if sum(ns) else 0.0
    worst = round(min(f1s), 4) if f1s else None
    return f1s, wavg, aucs, _wavg_finite(aucs, ns), worst


def run_fedavg(alpha: float = 0.5, n_clients: int = 5, rounds: int = 10,
               local_epochs: int = 6, local_only_epochs: int = 60,
               seed: int = config.SEED, patient_history: bool = True, df=None,
               partition_fn=None, label: str | None = None, compute_pooled: bool = True):
    """partition_fn(df, ...) -> patient->client Series lets us swap the split (ward-
    Dirichlet default, or label_dirichlet / specimen_baseline / topology_split). The fn is
    dispatched by keyword match on its signature (see partition._call_partition): n_clients
    and seed are injected when declared. Any client labels (str/int) are normalised to
    contiguous 0..k-1 ids and n_clients is derived from the split, so specimen (3-4
    hospitals) works without extra args."""
    import torch
    from flwr.simulation import run_simulation

    torch.manual_seed(seed)
    df = load_cohort_frame() if df is None else df
    raw = (dirichlet_ward_mixture(df, n_clients=n_clients, alpha=alpha, seed=seed)
           if partition_fn is None
           else _call_partition(partition_fn, df, n_clients=n_clients, seed=seed))  # n_clients injected only when the split declares it (see partition._call_partition)
    codes, _ = pd.factorize(raw)                 # str/int client labels -> 0..k-1
    assign = pd.Series(codes, index=raw.index)
    n_clients = int(assign.max()) + 1
    tag = label or f"alpha={alpha}"
    cfg = _run_config(n_clients, rounds, local_epochs)
    write_run_config(cfg)
    sizes = build_and_save_clients(df, assign, n_clients, seed=seed, patient_history=patient_history)
    print(f"{tag} | {n_clients} hospitals | patients each: {sizes}")

    lo_f1s, lo_avg, lo_aucs, lo_auc = run_local_only(n_clients, cfg, epochs=local_only_epochs)
    print(f"LOCAL-ONLY per-hospital macro-F1: {lo_f1s} | weighted avg = {lo_avg:.4f} "
          f"| AUROC = {lo_auc:.4f}")

    # pooled baseline, scored with the SAME per-hospital protocol (apples-to-apples)
    if compute_pooled:
        pl_f1s, pl_avg, pl_aucs, pl_auc, pl_worst = run_pooled(n_clients, cfg, epochs=local_only_epochs)
        print(f"POOLED (centralized, same protocol): {pl_f1s} | weighted avg = {pl_avg:.4f} "
              f"| AUROC = {pl_auc:.4f} | worst = {pl_worst}")
    else:
        pl_f1s, pl_avg, pl_aucs, pl_auc, pl_worst = None, POOLED_REFERENCE, None, float("nan"), None

    ngpu = (0.9 / n_clients) if DEVICE == "cuda" else 0.0
    reset_fed_history()
    free_gpu()  # clear the local-only phase's allocations before the Ray clients start
    run_simulation(  # returns None in flwr 1.23; the strategy logs per-round F1 to disk
        server_app=server_app.app,
        client_app=client_app.app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": ngpu}},
    )
    records = read_fed_records()
    fed = [r["macro_f1"] for r in records]
    fed_auc = [r.get("auroc") for r in records]
    print("FEDAVG macro-F1 by round:", [round(v, 4) for v in fed])
    print("FEDAVG AUROC by round:   ", [round(v, 4) if v is not None and v == v else None for v in fed_auc])
    best_idx = max(range(len(fed)), key=fed.__getitem__) if fed else None  # best round by macro-F1
    fed_best = round(fed[best_idx], 4) if fed else None  # (FedAvg drifts on non-IID)
    fed_final = round(fed[-1], 4) if fed else None
    fed_auc_best = (round(fed_auc[best_idx], 4)
                    if fed and fed_auc[best_idx] is not None and fed_auc[best_idx] == fed_auc[best_idx]
                    else None)
    # per-hospital FedAvg F1/AUROC at the best round (global model, each client's own test set)
    fed_pc = records[best_idx].get("per_client", {}) if fed else {}
    fed_pc_auc = records[best_idx].get("per_client_auroc", {}) if fed else {}
    fed_f1s = [fed_pc.get(str(c)) for c in range(n_clients)]
    fed_aucs = [fed_pc_auc.get(str(c)) for c in range(n_clients)]

    pooled_str = f"{pl_avg:.4f}" + ("" if pl_worst is None else f" (worst {pl_worst})")
    print(f"\n=== Phase 3 comparison ({tag}) ===  [macro-F1 | AUROC]")
    print(f"  pooled (centralized, same protocol): {pooled_str} | AUROC {pl_auc:.4f}")
    print(f"  FedAvg  best-round / final     : {fed_best} / {fed_final} | AUROC {fed_auc_best}")
    print(f"  local-only (alone, weighted)   : {lo_avg:.4f} | AUROC {lo_auc:.4f}")
    if fed_best is not None:
        print(f"  => FedAvg (best) {'BEATS' if fed_best > lo_avg else 'does NOT beat'} local-only; "
              f"{'MATCHES/beats' if fed_best >= pl_avg else 'below'} pooled")
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
    return {"alpha": alpha, "pooled": pl_avg, "pooled_worst": pl_worst,
            "pooled_per_client": pl_f1s, "fedavg_best": fed_best,
            "fedavg_final": fed_final, "local_only": lo_avg,
            "local_only_per_client": lo_f1s, "fedavg_per_client": fed_f1s,
            "worst_local": worst_local, "worst_fed": worst_fed,
            "sizes": sizes, "fed_by_round": fed,
            # AUROC (weighted, NaN-safe): threshold-free, literature-comparable
            "pooled_auroc": pl_auc, "local_only_auroc": lo_auc, "fedavg_auroc_best": fed_auc_best,
            "fedavg_auroc_per_client": fed_aucs, "fedavg_auroc_by_round": fed_auc}


def run_multiseed(alpha: float = 0.5, n_clients: int = 5, rounds: int = 10,
                  local_epochs: int = 6, seeds=(42, 43, 44), patient_history: bool = True,
                  df=None, partition_fn=None, label: str | None = None,
                  compute_pooled: bool = True):
    """Run FedAvg over several seeds; report mean +/- std to denoise the partition +
    training randomness. partition_fn swaps the split (default ward-Dirichlet at `alpha`;
    pass label_dirichlet / specimen_baseline for the non-IID settings).
    Note: n_clients is forwarded to the split only when it declares an n_clients param
    (topology_split / homophily_split); splits without it (specimen_baseline) keep their
    own hospital count. topology_split needs n_clients a multiple of 4, so pass e.g.
    n_clients=4 or 8. Loads the cohort ONCE and reuses it."""
    df = load_cohort_frame() if df is None else df
    tag = label or f"alpha={alpha}"
    runs = []
    for s in seeds:
        print(f"\n########## {tag}  seed={s} ##########")
        runs.append(run_fedavg(alpha=alpha, n_clients=n_clients, rounds=rounds,
                               local_epochs=local_epochs, seed=s,
                               patient_history=patient_history, df=df,
                               partition_fn=partition_fn, label=label,
                               compute_pooled=compute_pooled))

    def ms(key):
        vals = [r[key] for r in runs if r.get(key) is not None and r[key] == r[key]]  # drop None/NaN
        if not vals:
            return None, None
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    gains = [r["fedavg_best"] - r["local_only"] for r in runs
             if r["fedavg_best"] is not None]
    lo, fb, ff = ms("local_only"), ms("fedavg_best"), ms("fedavg_final")
    pl, plw = ms("pooled"), ms("pooled_worst")    # centralized, same protocol
    wl, wf = ms("worst_local"), ms("worst_fed")   # worst-hospital (fairness) metric
    pla, loa, fba = ms("pooled_auroc"), ms("local_only_auroc"), ms("fedavg_auroc_best")  # AUROC
    gmean, gstd = round(float(np.mean(gains)), 4), round(float(np.std(gains)), 4)
    worst_gains = [r["worst_fed"] - r["worst_local"] for r in runs
                   if r.get("worst_fed") is not None]
    wgmean = round(float(np.mean(worst_gains)), 4) if worst_gains else None
    wgstd = round(float(np.std(worst_gains)), 4) if worst_gains else None

    print(f"\n=== MULTI-SEED SUMMARY: {tag}, {len(seeds)} seeds ===")
    print(f"  pooled (same protocol) : {pl[0]} +/- {pl[1]}  | worst {plw[0]} +/- {plw[1]}")
    print(f"  local-only             : {lo[0]} +/- {lo[1]}")
    print(f"  FedAvg best-round      : {fb[0]} +/- {fb[1]}")
    print(f"  FedAvg final           : {ff[0]} +/- {ff[1]}")
    print(f"  gain (best-local)      : {gmean} +/- {gstd}")
    print(f"  FedAvg-best vs pooled  : {round(fb[0] - pl[0], 4) if pl[0] is not None else 'n/a'}")
    print(f"  worst-hosp local       : {wl[0]} +/- {wl[1]}")
    print(f"  worst-hosp FedAvg      : {wf[0]} +/- {wf[1]}")
    print(f"  worst-hosp gain        : {wgmean} +/- {wgstd}")
    print(f"  --- AUROC (weighted) ---")
    print(f"  pooled AUROC           : {pla[0]} +/- {pla[1]}")
    print(f"  local-only AUROC       : {loa[0]} +/- {loa[1]}")
    print(f"  FedAvg (best) AUROC    : {fba[0]} +/- {fba[1]}")
    return {"alpha": alpha, "seeds": list(seeds), "pooled": pl, "pooled_worst": plw,
            "local_only": lo, "fedavg_best": fb, "fedavg_final": ff,
            "gain_mean": gmean, "gain_std": gstd, "worst_local": wl, "worst_fed": wf,
            "worst_gain_mean": wgmean, "worst_gain_std": wgstd,
            "pooled_auroc": pla, "local_only_auroc": loa, "fedavg_auroc_best": fba,
            "runs": runs}
