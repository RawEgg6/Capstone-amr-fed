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
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from . import config
from .graph_build import build_graph
from .model import AMRSAGE


def _macro_f1(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    return f1_score(y_true.cpu().numpy(), pred, average="macro")


def train(data, hidden: int = 64, layers: int = 2, epochs: int = 60,
          lr: float = 1e-3, weight_decay: float = 1e-4, eval_every: int = 5,
          device: str | None = None):
    torch.manual_seed(config.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
    tri = data.triple_index.to(device)
    y = data.triple_label.to(device).float()
    tr, va, te = (m.to(device) for m in (data.train_mask, data.val_mask, data.test_mask))
    pos_weight = data.train_pos_weight.to(device)

    model = AMRSAGE(list(edge_index_dict.keys()), hidden=hidden, layers=layers).to(device)
    with torch.no_grad():                       # materialise lazy SAGEConv params before optim
        model.encode(x_dict, edge_index_dict)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_dict, edge_index_dict, tri[:, tr]), y[tr])
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                vf1 = _macro_f1(y[va], model(x_dict, edge_index_dict, tri[:, va]))
            if vf1 > best_val:
                best_val = vf1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d} | train loss {loss.item():.4f} | val macro-F1 {vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        t_logits = model(x_dict, edge_index_dict, tri[:, te])
    y_te = y[te].cpu().numpy()
    test_f1 = _macro_f1(y[te], t_logits)
    baseline_f1 = f1_score(y_te, np.zeros_like(y_te), average="macro")  # always-Susceptible
    pred = (torch.sigmoid(t_logits) >= 0.5).long().cpu().numpy()

    print(f"\nTEST macro-F1: {test_f1:.4f}  |  majority-baseline macro-F1: {baseline_f1:.4f}")
    print("confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(confusion_matrix(y_te, pred))
    return model, {"best_val_macro_f1": best_val, "test_macro_f1": test_f1,
                   "baseline_macro_f1": baseline_f1}


def main(ward: str | None = None, enrich: tuple = (), comorbidity_cache: str | None = None,
         exposure_cache: str | None = None, rich_patient: bool = False,
         labvital_cache: str | None = None):
    return train(build_graph(ward=ward, enrich=enrich, comorbidity_cache=comorbidity_cache,
                             exposure_cache=exposure_cache, rich_patient=rich_patient,
                             labvital_cache=labvital_cache))


if __name__ == "__main__":
    main()
