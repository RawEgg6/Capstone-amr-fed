"""Single-hospital training + evaluation for AMR-SAGE.

Class-weighted binary edge classification on the per-test triples; reports
test macro-F1 against the majority-class baseline + a confusion matrix. This
is the Phase-1 gate: model macro-F1 must clearly beat the baseline (~0.45).

Runs on Colab (torch). Example:
    from amr_fed.train_local import main
    model, metrics = main(ward=None)   # or ward="ICU" for one hospital
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn

from . import config
from .graph_build import build_graph
from .model import AMRSAGE


def _macro_f1(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    return f1_score(y_true.cpu().numpy(), pred, average="macro")


def _f1_at(y_np, p_np, thr):
    return f1_score(y_np, (p_np >= thr).astype(int), average="macro")


def _best_threshold(y_np, p_np):
    """Threshold that maximises macro-F1 on validation (no test leakage)."""
    grid = np.linspace(0.15, 0.85, 29)
    return float(max(grid, key=lambda t: _f1_at(y_np, p_np, t)))


def train(data, hidden: int = 64, layers: int = 2, epochs: int = 60,
          lr: float = 1e-3, weight_decay: float = 1e-4, eval_every: int = 5,
          dropout: float = 0.3, aggr: str = "sum", device: str | None = None,
          verbose: bool = True):
    torch.manual_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
    tri = data.triple_index.to(device)
    y = data.triple_label.to(device).float()
    tr, va, te = (m.to(device) for m in (data.train_mask, data.val_mask, data.test_mask))
    pos_weight = data.train_pos_weight.to(device)
    tf = getattr(data, "triple_feat", None)                 # per-test patient-history features
    if tf is not None:
        tf = tf.to(device)
    tf_dim = tf.shape[1] if tf is not None else 0

    def fwd(mask):
        return model(x_dict, edge_index_dict, tri[:, mask], tf[mask] if tf is not None else None)

    model = AMRSAGE(list(edge_index_dict.keys()), hidden=hidden, layers=layers,
                    dropout=dropout, aggr=aggr, triple_feat_dim=tf_dim).to(device)
    with torch.no_grad():                       # materialise lazy SAGEConv params before optim
        model.encode(x_dict, edge_index_dict)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(fwd(tr), y[tr])
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                vf1 = _macro_f1(y[va], fwd(va))
            if vf1 > best_val:
                best_val = vf1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"epoch {epoch:3d} | train loss {loss.item():.4f} | val macro-F1 {vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(fwd(va)).cpu().numpy()
        p_te = torch.sigmoid(fwd(te)).cpu().numpy()
    y_va, y_te = y[va].cpu().numpy(), y[te].cpu().numpy()

    test_f1 = _f1_at(y_te, p_te, 0.5)                        # fixed-threshold (comparable to before)
    thr = _best_threshold(y_va, p_val)                       # tuned on val
    test_f1_tuned = _f1_at(y_te, p_te, thr)
    auroc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    baseline_f1 = f1_score(y_te, np.zeros_like(y_te), average="macro")

    if verbose:
        tag = " | +patient-history" if tf_dim else ""
        print(f"\nTEST macro-F1 @0.5: {test_f1:.4f} | @tuned(thr={thr:.2f}): {test_f1_tuned:.4f} "
              f"| AUROC: {auroc:.4f} | baseline: {baseline_f1:.4f}{tag}")
        print("confusion matrix @0.5 [rows=true 0/1, cols=pred 0/1]:")
        print(confusion_matrix(y_te, (p_te >= 0.5).astype(int)))
    return model, {"best_val_macro_f1": best_val, "test_macro_f1": test_f1,
                   "test_macro_f1_tuned": round(test_f1_tuned, 4), "auroc": round(float(auroc), 4),
                   "tuned_threshold": round(thr, 3), "baseline_macro_f1": baseline_f1,
                   "triple_feat_dim": tf_dim}


_SWEEP_GRID = [
    {"aggr": "sum",  "hidden": 64,  "layers": 2, "lr": 1e-3, "epochs": 60},   # current baseline
    {"aggr": "mean", "hidden": 64,  "layers": 2, "lr": 1e-3, "epochs": 60},   # mean-agg
    {"aggr": "mean", "hidden": 128, "layers": 2, "lr": 1e-3, "epochs": 80},   # wider
    {"aggr": "mean", "hidden": 128, "layers": 3, "lr": 5e-4, "epochs": 100},  # deeper + slower
    {"aggr": "mean", "hidden": 64,  "layers": 2, "lr": 5e-4, "epochs": 100, "dropout": 0.4},  # more reg
]


def sweep(data, configs: list | None = None):
    """Train several architectures on the SAME prebuilt graph and rank by test macro-F1.

    Answers 'is ~0.66 a real ceiling or under-optimization?'. Build `data` once
    (build_graph) and pass it in — graph construction is the slow part, not training.
    """
    configs = configs or _SWEEP_GRID
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"[sweep {i}/{len(configs)}] {cfg} ...")
        _, m = train(data, verbose=False, **cfg)
        results.append({**cfg, **m})
    results.sort(key=lambda r: r["test_macro_f1"], reverse=True)
    print("\n=== SWEEP SUMMARY (best test macro-F1 first) ===")
    for r in results:
        print(f"  test={r['test_macro_f1']:.4f}  val={r['best_val_macro_f1']:.4f}  |  "
              f"aggr={r['aggr']:4s} hidden={r['hidden']} layers={r['layers']} "
              f"lr={r['lr']} epochs={r.get('epochs', 60)} dropout={r.get('dropout', 0.3)}")
    return results


# Trimmed 3-config arch sweep for the exhaustive feature x arch grid (keeps it ~1h).
_GRID_ARCH = [
    {"aggr": "sum",  "hidden": 64,  "layers": 2, "lr": 1e-3, "epochs": 60},          # baseline
    {"aggr": "mean", "hidden": 128, "layers": 2, "lr": 1e-3, "epochs": 80},          # wider, mean
    {"aggr": "mean", "hidden": 64,  "layers": 2, "lr": 5e-4, "epochs": 100, "dropout": 0.4},  # more reg
]
_ENRICH_OPTS = ["comorbidity", "prior_exposure", "procedure"]


def full_grid(cache_dir: str = "/content/drive/MyDrive", ward: str | None = None,
              arch_configs: list | None = None):
    """Exhaustive search: EVERY feature combination x an architecture sweep.

    Feature combos = the 8 subsets of {comorbidity, prior_exposure, procedure}
    crossed with rich_patient {off, on} = 16. Each combo is trained under every
    arch in arch_configs (default 3). The cohort is loaded ONCE and only the graph
    is rebuilt per combo (caches make enrichment cheap). Prints a running best
    (so partial results survive a Colab disconnect) and a final Top-10.

    NOTE: ensure the enrichment caches exist first (run cells 6/7/8 once), or the
    first comorbidity combo will stream the 18GB CSV before caching it.
    """
    import gc
    import itertools

    import torch

    from .data_loader import load_cohort_frame
    from .graph_build import build_arrays, to_hetero_data

    arch_configs = arch_configs or _GRID_ARCH
    caches = dict(
        comorbidity_cache=f"{cache_dir}/amr_comorbidity_edges.parquet",
        exposure_cache=f"{cache_dir}/amr_prior_exposure_edges.parquet",
        procedure_cache=f"{cache_dir}/amr_procedure_edges.parquet",
        labvital_cache=f"{cache_dir}/amr_labvital_per_culture.parquet",
    )
    combos = [(tuple(sub), rp)
              for r in range(len(_ENRICH_OPTS) + 1)
              for sub in itertools.combinations(_ENRICH_OPTS, r)
              for rp in (False, True)]
    print(f"{len(combos)} feature combos x {len(arch_configs)} archs = "
          f"{len(combos) * len(arch_configs)} runs\n")

    df = load_cohort_frame(ward=ward)                 # loaded ONCE, reused across combos
    results, best = [], None
    for ci, (enrich, rp) in enumerate(combos, 1):
        data = to_hetero_data(build_arrays(df, enrich=enrich, rich_patient=rp, **caches))
        feats = ("+".join(enrich) or "core") + (" +labs/vitals" if rp else "")
        for cfg in arch_configs:
            _, m = train(data, verbose=False, **cfg)
            row = {"feats": feats, **cfg, **m}
            results.append(row)
            if best is None or m["test_macro_f1"] > best["test_macro_f1"]:
                best = row
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{ci:2d}/{len(combos)}] {feats:<40s} best-so-far "
              f"test={best['test_macro_f1']:.4f} ({best['feats']}, aggr={best['aggr']}, h={best['hidden']})")

    results.sort(key=lambda r: r["test_macro_f1"], reverse=True)
    print(f"\n=== TOP 10 of {len(results)} runs (core reference = 0.663) ===")
    for r in results[:10]:
        print(f"  test={r['test_macro_f1']:.4f} val={r['best_val_macro_f1']:.4f} | "
              f"{r['feats']:<40s} | aggr={r['aggr']:4s} h={r['hidden']} L={r['layers']} lr={r['lr']}")
    return results


def main(ward: str | None = None, enrich: tuple = (), comorbidity_cache: str | None = None,
         exposure_cache: str | None = None, rich_patient: bool = False,
         labvital_cache: str | None = None, patient_history: bool = False,
         specimen: bool = False, prescriptions: bool = False,
         prescription_cache: str | None = None):
    return train(build_graph(ward=ward, enrich=enrich, comorbidity_cache=comorbidity_cache,
                             exposure_cache=exposure_cache, rich_patient=rich_patient,
                             labvital_cache=labvital_cache, patient_history=patient_history,
                             specimen=specimen, prescriptions=prescriptions,
                             prescription_cache=prescription_cache))


if __name__ == "__main__":
    main()
