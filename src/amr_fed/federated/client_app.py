"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.

SIGFPE forensics (notebook sets AMR_DEBUG_INIT=1 before flwr's ray.init): a
[client-init] line per construction and a [kaiming] line per PyG lazy-weight
reset (the crash site), both flush=True so they survive the fault.
"""
from __future__ import annotations

import ctypes
import importlib
import json
import os

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import (
    DEVICE, FIT_METRIC_KEY, get_weights, init_model_on, load_client_graph, local_eval,
    local_train, read_run_config, seed_client, set_weights,
)


def _debug_on() -> bool:
    return os.environ.get("AMR_DEBUG_INIT", "").strip().lower() not in ("", "0", "false", "no")


def _fenv_mask() -> int | str:
    try:
        return int(ctypes.CDLL("libm.so.6").fegetexcept())
    except Exception as e:
        return f"err {e}"


def _install_kaiming_probe() -> None:
    """Wrap every PyG binding of kaiming_uniform so the SIGFPE site logs first.

    The crash dies inside uniform_ called from kaiming_uniform (pyg inits.py:22);
    this prints shape/dtype/device/fan/bound/mask immediately before that call —
    the last line we see if it faults. Patched in both the inits module and the
    linear module in case reset_weight_ imported the name by value.
    """
    if not _debug_on():
        return
    for name in ("torch_geometric.nn.inits", "torch_geometric.nn.dense.linear"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        orig = getattr(mod, "kaiming_uniform", None)
        if orig is None or getattr(mod, "_amr_probe", False):
            continue

        def _make_probe(orig_fn):
            def _probe(value, *args, **kwargs):
                fan = kwargs.get("fan", args[0] if args else 0)
                a = kwargs.get("a", args[1] if len(args) > 1 else 5 ** 0.5)
                try:
                    bound = (6.0 / ((1 + a ** 2) * float(fan))) ** 0.5
                except BaseException as e:
                    bound = f"ERR {e!r}"
                print(f"[kaiming] shape={tuple(value.shape)} dtype={value.dtype} "
                      f"device={value.device} fan={fan!r} bound={bound} mask={_fenv_mask()}",
                      flush=True)
                return orig_fn(value, *args, **kwargs)
            return _probe

        setattr(mod, "kaiming_uniform", _make_probe(orig))
        setattr(mod, "_amr_probe", True)


class FlowerClient(NumPyClient):
    def __init__(self, data, cfg: dict, cid: int):
        self.data = data
        self.cfg = cfg
        self.cid = cid
        self.model = init_model_on(data, cfg)   # materialises lazy params

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        local_train(self.model, self.data, self.cfg["local_epochs"])
        metrics: dict = {}
        # Phase 5: hand the server our topology fingerprint (driver-computed from our
        # patient subset and stored in run_config.json). Absent -> the server falls back
        # to plain FedAvg weights for us.
        fp = (self.cfg.get("fingerprints") or {}).get(str(self.cid))
        if fp is not None:
            metrics[FIT_METRIC_KEY] = json.dumps([float(v) for v in fp])
        return get_weights(self.model), int(self.data.train_mask.sum()), metrics

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        f1, auc, n = local_eval(self.model, self.data, "test_mask")
        # cid -> per-hospital breakdown; auroc alongside macro_f1
        return 0.0, n, {"macro_f1": f1, "auroc": auc, "cid": self.cid}


def client_fn(context: Context):
    cfg = read_run_config()
    pid = int(context.node_config["partition-id"])
    # Ray runs each client in its own process with a fresh RNG; seed it deterministically
    # (parent seed + cid) BEFORE the model is built, so FedAvg is reproducible run-to-run.
    seed_client(cfg, pid)
    data = load_client_graph(pid)
    if _debug_on():
        import torch
        tf = getattr(data, "triple_feat", None)
        print(f"[client-init cid={pid}] device={DEVICE} cuda={torch.cuda.is_available()} "
              f"vis={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()} "
              f"fenv={_fenv_mask()} cfg={cfg['hidden']}/{cfg['layers']}/{cfg['aggr']} "
              f"rounds={cfg['rounds']} seed={cfg['seed']} "
              f"x={{{', '.join(f'{k}:{tuple(v.shape)}' for k, v in data.x_dict.items())}}} "
              f"e={{{', '.join(f'{str(k)}:{int(v.shape[1])}' for k, v in data.edge_index_dict.items())}}} "
              f"tf={tuple(tf.shape) if tf is not None else None}", flush=True)
    client = FlowerClient(data, cfg, pid)
    if _debug_on():
        print(f"[client-init cid={pid}] model OK", flush=True)
    return client.to_client()


_install_kaiming_probe()
app = ClientApp(client_fn=client_fn)
