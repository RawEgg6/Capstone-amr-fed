"""New-hospital experiment (notebook 18, part B): how well does a federated model serve a hospital that
never took part in training, with and without a little local data, compared with training from scratch?

For one seed: hold one hospital out, train FedAvg on the others, then for each number of labelled local
patients ``k`` (``regimes.slice_frame``) score on the held-out hospital's FULL test set:
  zero_shot  the global model as is (cutoff chosen on the hospital's own small validation slice)
  finetune   the global model + up to 8 local epochs (epoch and cutoff chosen on the validation slice;
             epoch 0 is the zero-shot model, so fine-tuning can only match or improve on validation)
  scratch    a fresh model trained alone on the same slice (32 epochs, checkpoint every 4; same selection)
All selection uses validation rows only; test is read once at each choice.
"""
from __future__ import annotations

import numpy as np

from ..data_loader import PK
from .regimes import holdout_hospital, slice_frame
from .selection import select_local
from .task import (
    build_graph, free_gpu, init_model_on, load_global_params, local_train_tracked, read_run_config,
    seed_everything, set_weights,
)

KS = (25, 100, 400, None)        # labelled training patients at the new hospital (None = all of them)
_KEEP = ("epoch", "n_val", "n_test", "val_cm", "test_cm", "val_hist", "test_hist", "val_auroc", "test_auroc")


def _summarize(hist, ckpts=None):
    use = list(range(len(hist))) if ckpts is None else ckpts
    sel = select_local([hist[i]["val_grid"] for i in use], [hist[i]["test_grid"] for i in use])
    h = hist[use[sel["ckpt"]]]
    auc = h["test_auroc"]
    return {"f1": float(sel["test_f1"]), "thr": sel["thr"], "val": float(sel["val_score"]), "epoch": h["epoch"],
            "auroc": None if auc != auc else float(auc), "n_val": h["n_val"], "n_test": h["n_test"]}


def run_new_site(*, df, partition_fn, seed: int, rule: str, ks=KS, rounds: int = 8, local_epochs: int = 4,
                 local_only_epochs: int = 32, hidden: int = 128, ft_epochs: int = 8, label: str = "newsite",
                 patient_history: bool = True) -> dict:
    from .run import run_fedavg
    assign_full = partition_fn(df, seed)
    h = holdout_hospital(assign_full, rule)
    held = set(assign_full.index[assign_full == h])
    df_rest = df[~df[PK].isin(held)]
    assign_rest = assign_full[assign_full != h]
    fed = run_fedavg(df=df_rest, partition_fn=lambda d, s: assign_rest, n_clients=int(assign_rest.nunique()),
                     rounds=rounds, local_epochs=local_epochs, seed=seed, hidden=hidden, label=f"{label}/{rule}",
                     compute_pooled=False, compute_local=False, save_global=True, patient_history=patient_history)
    gp = load_global_params()
    cfg = read_run_config()
    held_df = df[df[PK].isin(held)]
    out = {"seed": seed, "rule": rule, "holdout": int(h), "holdout_patients": len(held),
           "other_sizes": fed["assignment_sizes"], "fed_phc_f1": fed["phc_f1"], "fed_wall_seconds": fed["wall_seconds"],
           "ks": {}}
    for k in ks:
        sub, code = slice_frame(held_df, seed, k)
        data = build_graph(sub, seed, patient_history, code)
        n_tr = int((data.train_mask).sum()); n_va = int((data.val_mask).sum()); n_te = int((data.test_mask).sum())
        # fine-tune the federated model (epoch 0 = zero-shot)
        model = init_model_on(data, cfg)
        set_weights(model, gp)
        hist = local_train_tracked(model, data, ft_epochs, eval_every=2, eval_at_start=True)
        zero = _summarize(hist, ckpts=[0])
        tuned = _summarize(hist)
        detail_ft = [{x: hh[x] for x in _KEEP} for hh in hist]
        del model
        free_gpu()
        # from scratch on the same slice
        seed_everything(seed + 5000 + (k or 0))
        model = init_model_on(data, cfg)
        hist2 = local_train_tracked(model, data, local_only_epochs, eval_every=4)
        scratch = _summarize(hist2)
        detail_sc = [{x: hh[x] for x in _KEEP} for hh in hist2]
        del model, data
        free_gpu()
        out["ks"]["all" if k is None else str(k)] = {
            "n_train_patients": len(sub[sub[PK].map(code) == 0][PK].unique()),
            "n_val_patients": len(sub[sub[PK].map(code) == 1][PK].unique()),
            "n_triples": {"train": n_tr, "val": n_va, "test": n_te},
            "zero_shot": zero, "finetune": tuned, "scratch": scratch,
            "detail_finetune": detail_ft, "detail_scratch": detail_sc}
        print(f"{label}/{rule} seed {seed} | held-out H{h} k={'all' if k is None else k}: "
              f"zero-shot {zero['f1']:.4f} | fine-tuned {tuned['f1']:.4f} | scratch {scratch['f1']:.4f}")
    return out
