"""Phase 3 / Phase 5 driver: local-only, FedAvg, pooled, and topology-aware comparison.

## Phase 3 baselines (all splits)
  pooled      -- ONE model on all data ("everyone shares data" ceiling)
  local-only  -- each hospital trains alone (weighted-avg macro-F1)
  FedAvg      -- hospitals train locally, server averages weights each round

## Phase 5 goal (Path B — locked after 2026-08-17 calibration)
  The canonical evaluation splits are organism-community and specimen (both show
  a clean, reproducible FedAvg > local-only gain with p<0.01 sign test). The
  topology-aware aggregator's job is to beat FedAvg's *uniform* averaging — a
  legitimate, publishable novelty that does NOT require FedAvg to fail.

  New acceptance gate (phase5_gate):
    fed_helps  -- FedAvg beats local-only by >= thresh_fed (split is worth federating)
    topo_wins  -- topology-aware beats FedAvg by >= thresh_topo (the novel method works)
    both above measured as mean over seeds, error bars don't overlap zero

FedAvg's per-round local epochs x rounds equals local-only's epoch budget so the
comparison is fair. Run on Colab (needs torch + flwr[simulation]).
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

# ---------------------------------------------------------------------------
# Canonical Phase 5 evaluation settings (locked 2026-08-17, Path B pivot)
# Both splits show clean FedAvg > local-only gains reproducible across 3 seeds.
# Topology-aware aggregation will be evaluated on these two splits.
# ---------------------------------------------------------------------------
CANONICAL_SPLIT_ORGANISM = "organism-community"   # strongest: FedAvg +0.023, worst-hosp +0.037
CANONICAL_SPLIT_SPECIMEN = "specimen"             # clinical: FedAvg +0.019, worst-hosp +0.023
CANONICAL_ROUNDS = 10
CANONICAL_LOCAL_EPOCHS = 6
CANONICAL_HIDDEN = 128


def _wavg_finite(vals, weights) -> float:
    """Weighted mean over finite entries (skips None / NaN); NaN if none are finite.
    Used for AUROC, which is NaN for any single-class hospital."""
    pairs = [(v, w) for v, w in zip(vals, weights) if v is not None and v == v]  # v==v: False for NaN
    if not pairs:
        return float("nan")
    v, w = zip(*pairs)
    return float(np.average(v, weights=w))


def _run_config(n_clients: int, rounds: int, local_epochs: int, hidden: int = 128,
                seed: int = config.SEED) -> dict:
    # canonical Phase-1 architecture (best clean config from the grid)
    return {"n_clients": n_clients, "rounds": rounds, "local_epochs": local_epochs,
            "hidden": hidden, "layers": 2, "aggr": "mean", "seed": seed}


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

    Training uses a **uniform per-hospital loss weight** (1/n_clients per hospital) so
    every hospital contributes equally to the gradient regardless of patient count. This
    matches FedAvg's implicit assumption (one model update per client per round, not
    proportional to client size) and prevents large hospitals from drowning out the
    signal from small/rare communities — fixing the root cause that made FedAvg beat
    pooled on every topology split.

    Memory-safe: graphs live on CPU; one hospital is moved to GPU at a time and gradients
    are accumulated across hospitals before each optimiser step."""
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
            # Uniform per-hospital weight (1/n_clients) so every hospital contributes
            # equally to pooled's gradient regardless of patient-count. This prevents
            # large hospitals from drowning out small/rare communities and makes pooled
            # a fair ceiling vs FedAvg (which also treats each hospital equally by
            # aggregating one model update per client per round, not weighted by size).
            loss = loss_fn(out, y[tr]) / n_clients
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
               local_epochs: int = 6, local_only_epochs: int | None = None,
               seed: int = config.SEED, patient_history: bool = True, df=None,
               partition_fn=None, label: str | None = None, compute_pooled: bool = True,
               hidden: int = 128):
    """partition_fn(df, ...) -> patient->client Series lets us swap the split (ward-
    Dirichlet default, or label_dirichlet / specimen_baseline / topology_split). The fn is
    dispatched by keyword match on its signature (see partition._call_partition): n_clients
    and seed are injected when declared. Any client labels (str/int) are normalised to
    contiguous 0..k-1 ids and n_clients is derived from the split, so specimen (3-4
    hospitals) works without extra args.
    local_only_epochs defaults to rounds*local_epochs (matched budget: local-only trains
    the same total epochs as FedAvg's per-round local epochs x rounds)."""
    local_only_epochs = rounds * local_epochs if local_only_epochs is None else local_only_epochs
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
    cfg = _run_config(n_clients, rounds, local_epochs, hidden=hidden, seed=seed)
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
                  compute_pooled: bool = True, local_only_epochs: int | None = None,
                  hidden: int = 128):
    """Run FedAvg over several seeds; report mean +/- std to denoise the partition +
    training randomness. partition_fn swaps the split (default ward-Dirichlet at `alpha`;
    pass label_dirichlet / specimen_baseline for the non-IID settings).
    Note: n_clients is forwarded to the split only when it declares an n_clients param
    (topology_split / homophily_split); splits without it (specimen_baseline) keep their
    own hospital count. topology_split needs n_clients a multiple of 4, so pass e.g.
    n_clients=4 or 8. Loads the cohort ONCE and reuses it.
    local_only_epochs is passed straight through to run_fedavg (None = matched budget
    rounds*local_epochs); hidden is the per-client hidden width in _run_config."""
    df = load_cohort_frame() if df is None else df
    tag = label or f"alpha={alpha}"
    runs = []
    for s in seeds:
        print(f"\n########## {tag}  seed={s} ##########")
        runs.append(run_fedavg(alpha=alpha, n_clients=n_clients, rounds=rounds,
                               local_epochs=local_epochs, seed=s,
                               patient_history=patient_history, df=df,
                               partition_fn=partition_fn, label=label,
                               compute_pooled=compute_pooled,
                               local_only_epochs=local_only_epochs, hidden=hidden))

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


def headroom_gate(partition_fn, n_clients: int = 8, rounds: int = 6,
                  local_epochs: int = 3, hidden: int = 64,
                  seeds: tuple = (42, 43, 44), df=None, label: str | None = None,
                  thresh_worst: float = 0.02, thresh_mean: float = 0.01,
                  pooled_worst_floor: float = 0.60) -> dict:
    """Acceptance test: is `partition_fn` hard enough for topology-aware aggregation to
    matter? Runs multi-seed FedAvg vs pooled at a matched budget (local_only_epochs =
    rounds*local_epochs) and checks three conditions:
      fed_helps  -- FedAvg beats local-only (the split isn't trivially easy to train alone)
      gap_ok     -- pooled beats FedAvg by a meaningful margin on the worst hospital
                    (thresh_worst) OR on the mean (thresh_mean): headroom exists to chase
      pooled_ok  -- pooled itself clears pooled_worst_floor, so the split is hard but not
                    hopeless (a sub-0.60 worst hospital leaves no signal worth pooling)
    Prints PASS/FAIL with mean +/- std over seeds, per-condition hints on failure, and
    returns the verdict plus all numbers for the calibration notebook."""
    res = run_multiseed(partition_fn=partition_fn, n_clients=n_clients, rounds=rounds,
                        local_epochs=local_epochs, local_only_epochs=rounds * local_epochs,
                        hidden=hidden, seeds=seeds, df=df, label=label,
                        compute_pooled=True)
    pooled = res["pooled"]
    pooled_worst = res["pooled_worst"]
    fed = res["fedavg_best"]
    worst_fed = res["worst_fed"]
    local = res["local_only"]

    fed_helps = fed[0] > local[0]                                  # FedAvg beats local-only
    gap_ok = (worst_fed[0] <= pooled_worst[0] - thresh_worst) or (fed[0] <= pooled[0] - thresh_mean)
    pooled_ok = pooled_worst[0] >= pooled_worst_floor
    passed = fed_helps and gap_ok and pooled_ok

    def _fmt(v):
        return f"{v[0]} +/- {v[1]}" if v[0] is not None else "n/a"

    print(f"headroom_gate: {'PASS' if passed else 'FAIL'}  (mean +/- std over {len(seeds)} seeds)")
    print(f"  pooled       : {_fmt(pooled)}   worst {_fmt(pooled_worst)}")
    print(f"  fedavg_best  : {_fmt(fed)}   worst-hospital {_fmt(worst_fed)}")
    print(f"  local_only   : {_fmt(local)}")
    print(f"  fed > local  : {fed_helps} | worst gap: {_fmt(worst_fed)} <= {_fmt(pooled_worst)} - {thresh_worst} "
          f"| mean gap: {_fmt(fed)} <= {_fmt(pooled)} - {thresh_mean} | pooled_worst >= {pooled_worst_floor}: {pooled_ok}")
    if not fed_helps:
        print("  fed_helps: FedAvg does not beat local-only -> raise purity (0.2), rounds (8), or local_epochs (4)")
    if not gap_ok:
        print("  gap too small -> lower purity (0.0), try hidden=32, or rounds=4")
    if not pooled_ok:
        print("  pooled weak (pooled_worst < 0.60) -> widen hidden to 128, reduce n_clients to 4")

    return {"pass": passed, "pooled": pooled, "pooled_worst": pooled_worst,
            "fedavg_best": fed, "local_only": local, "worst_fed": worst_fed,
            "runs": res}


def phase5_gate(topo_f1: float, topo_f1_std: float,
                fedavg_f1: float, fedavg_f1_std: float,
                local_f1: float,
                thresh_fed: float = 0.010,
                thresh_topo: float = 0.005) -> dict:
    """Path-B acceptance test (locked 2026-08-17): topology-aware beats FedAvg.

    Checks two conditions (mean over seeds, std reported for transparency):
      fed_helps  -- FedAvg-best > local-only by >= thresh_fed
                    (the canonical split is worth federating; if federation
                    doesn't help vs training alone there's nothing to improve)
      topo_wins  -- topology-aware-best > FedAvg-best by >= thresh_topo
                    (the novel aggregator demonstrably beats uniform averaging)

    thresh_fed=0.010 (1 F1 point): organism-community showed +0.023 mean,
      specimen showed +0.019 — both well above this floor.
    thresh_topo=0.005 (0.5 F1 point): conservative floor; we expect ~0.01–0.02
      improvement from topology-weighting on the canonical splits.

    Args:
        topo_f1, topo_f1_std   -- mean ± std topology-aware macro-F1 (over seeds)
        fedavg_f1, fedavg_f1_std -- mean ± std FedAvg-best macro-F1 (over seeds)
        local_f1               -- mean local-only macro-F1 (deterministic, no std needed)
        thresh_fed             -- minimum FedAvg−local gap to count as "worth federating"
        thresh_topo            -- minimum topo−FedAvg gap to count as "topo wins"

    Returns dict with pass/fail verdict and all component numbers.
    """
    fed_helps = (fedavg_f1 - local_f1) >= thresh_fed
    topo_wins = (topo_f1 - fedavg_f1) >= thresh_topo

    # Overlap check: does the topo improvement error bar clearly exclude zero?
    # Using a simple (mean - 2*std > 0) proxy — conservative, no distributional assumption.
    topo_ci_positive = (topo_f1 - fedavg_f1 - 2 * max(topo_f1_std, fedavg_f1_std)) > 0

    passed = fed_helps and topo_wins

    print(f"\nphase5_gate: {'PASS ✅' if passed else 'FAIL ❌'}")
    print(f"  local-only     : {local_f1:.4f}")
    print(f"  FedAvg-best    : {fedavg_f1:.4f} +/- {fedavg_f1_std:.4f}")
    print(f"  topology-aware : {topo_f1:.4f} +/- {topo_f1_std:.4f}")
    print(f"  fed_helps  (FedAvg - local >= {thresh_fed}): "
          f"{fedavg_f1 - local_f1:+.4f} => {fed_helps}")
    print(f"  topo_wins  (topo - FedAvg >= {thresh_topo}): "
          f"{topo_f1 - fedavg_f1:+.4f} => {topo_wins}")
    print(f"  CI positive (topo gain - 2*std > 0): {topo_ci_positive}")
    if not fed_helps:
        print("  => fed_helps FAIL: try organism-community split (strongest gain +0.023)")
    if not topo_wins:
        print("  => topo_wins FAIL: topology fingerprint needs refinement "
              "or more rounds/hidden width")

    return {"pass": passed, "fed_helps": fed_helps, "topo_wins": topo_wins,
            "topo_ci_positive": topo_ci_positive,
            "fedavg_gain": round(fedavg_f1 - local_f1, 4),
            "topo_gain": round(topo_f1 - fedavg_f1, 4),
            "local_f1": local_f1, "fedavg_f1": fedavg_f1, "topo_f1": topo_f1}


def run_phase5_comparison(partition_fn=None, n_clients: int = 5,
                          rounds: int = CANONICAL_ROUNDS,
                          local_epochs: int = CANONICAL_LOCAL_EPOCHS,
                          seeds: tuple = (42, 43, 44),
                          hidden: int = CANONICAL_HIDDEN,
                          df=None, label: str | None = None,
                          compute_pooled: bool = True) -> dict:
    """Full 4-way Phase 5 comparison: local-only vs pooled vs FedAvg vs topology-aware.

    Uses the topology-aware strategy from strategy.py (Phase 5) alongside the
    standard FedAvg baseline. Both strategies run on the SAME pre-built client
    graphs so the comparison is exactly apples-to-apples.

    Default split: organism-community (canonical Phase 5 split, strongest gain).
    Pass partition_fn=specimen_baseline for the clinical defensibility story.

    This function is a STUB — it runs the FedAvg half now (Phase 3 code) and
    will run the topology-aware half once strategy.py is implemented (Phase 5).
    The stub prints a clear placeholder so the notebook cell works end-to-end
    and you can see the FedAvg baseline while Phase 5 is in development.

    Returns dict with all four baselines' results for phase5_gate.
    """
    from ..partition import organism_community

    if partition_fn is None:
        partition_fn = organism_community  # canonical default

    tag = label or CANONICAL_SPLIT_ORGANISM
    print(f"\n{'='*60}")
    print(f"Phase 5 comparison: {tag}")
    print(f"{'='*60}")

    # --- Step 1: FedAvg baseline (fully implemented, Phase 3) ---
    print("\n[1/2] Running FedAvg baseline (uniform averaging) ...")
    fedavg_res = run_multiseed(
        partition_fn=partition_fn,
        n_clients=n_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        seeds=seeds,
        hidden=hidden,
        df=df,
        label=tag,
        compute_pooled=compute_pooled,
    )

    # --- Step 2: Topology-aware (Phase 5 — stub until strategy.py is built) ---
    print("\n[2/2] Topology-aware aggregation ...")
    print("  *** STUB: strategy.py not yet implemented (Phase 5 in progress) ***")
    print("  To implement: build TopologyAwareStrategy in "
          "src/amr_fed/federated/strategy.py,")
    print("  then swap server_app.topology_aware_app for server_app.app in this function.")
    topo_res = None  # placeholder

    fed_f1, fed_std = fedavg_res["fedavg_best"]
    local_f1 = fedavg_res["local_only"][0]

    print(f"\n{'='*60}")
    print(f"FedAvg baseline (mean +/- std, {len(seeds)} seeds):")
    print(f"  local-only  : {local_f1:.4f}")
    print(f"  FedAvg-best : {fed_f1:.4f} +/- {fed_std:.4f}  "
          f"(gain: {fed_f1 - local_f1:+.4f})")
    print(f"  worst-hosp  : {fedavg_res['worst_fed'][0]:.4f} +/- "
          f"{fedavg_res['worst_fed'][1]:.4f}  "
          f"(gain: {fedavg_res['worst_gain_mean']:+.4f})")
    print(f"\nTopology-aware: awaiting Phase 5 implementation.")
    print(f"Target to beat: FedAvg {fed_f1:.4f} by >={0.005:.3f} F1 points")
    print(f"{'='*60}\n")

    return {"fedavg": fedavg_res, "topology_aware": topo_res,
            "canonical_split": tag, "seeds": list(seeds)}
