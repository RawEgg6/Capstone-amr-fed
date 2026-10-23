"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.

SIGFPE forensics (notebook sets AMR_DEBUG_INIT=1 before flwr's ray.init): a
[client-init] line per construction, a [bisect] env/subprocess probe once per
process, and a [kaiming] line per PyG lazy-weight reset (the crash site), all
flush=True so they survive the fault. FlowerClient also builds its model under
_zero_weight_init so the actor never calls a generator during construction.
"""
from __future__ import annotations

import ctypes
import importlib
import json
import os
from contextlib import contextmanager

import torch
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


# Every RNG-based weight initializer any part of the model construction can hit:
# PyG's inits module (incl. `inits.uniform(size, value)` for biases), PyG's
# linear module binding, and torch.nn.init (decoder Linear).
_RNG_INIT_TARGETS = (
    ("torch_geometric.nn.inits", "kaiming_uniform"),
    ("torch_geometric.nn.inits", "glorot"),
    ("torch_geometric.nn.inits", "uniform"),
    ("torch_geometric.nn.dense.linear", "kaiming_uniform"),
    ("torch.nn.init", "kaiming_uniform_"),
    ("torch.nn.init", "kaiming_normal_"),
    ("torch.nn.init", "xavier_uniform_"),
    ("torch.nn.init", "xavier_normal_"),
    ("torch.nn.init", "uniform_"),
    ("torch.nn.init", "normal_"),
    ("torch.nn.init", "trunc_normal_"),
)


@contextmanager
def _zero_weight_init():
    """Materialise the model WITHOUT touching a generator — the SIGFPE site.

    The actor's own init weights are dead computation regardless: fit() and
    evaluate() both start with set_weights(server parameters), and the server's
    initial global model is built in the driver (where RNG works), so FedAvg
    starts from the same random model as before and no client ever uses these
    values. Neutralising the initializers makes client construction independent
    of whatever the actor's generator state is.
    """
    saved, count = [], [0]

    def _zero(*args, **kwargs):
        hit = None
        for obj in (*args, *kwargs.values()):
            if isinstance(obj, torch.Tensor):
                with torch.no_grad():
                    obj.zero_()
                count[0] += 1
                hit = hit if hit is not None else obj
        return hit

    for name, attr in _RNG_INIT_TARGETS:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        orig = getattr(mod, attr, None)
        if orig is None:
            continue
        saved.append((mod, attr, orig))
        setattr(mod, attr, _zero)
    try:
        yield
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)
    if _debug_on():
        print(f"[zero-init] zeroed {count[0]} weight tensor(s) without RNG", flush=True)


class FlowerClient(NumPyClient):
    def __init__(self, data, cfg: dict, cid: int):
        self.data = data
        self.cfg = cfg
        self.cid = cid
        with _zero_weight_init():
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


_BISECT_DONE = False


def _bisect_once(pid: int) -> None:
    """Once per process: NON-FATAL probes only (a SIGFPE here must not kill the run).

    b1 dumps the actor's env/thread/RNG-relevant state; b2 runs the crashing op
    (cpu + cuda uniform_) in a FRESH subprocess inheriting the actor's env —
    rc!=0 means the environment itself is poisoned, rc=0 means only this
    process's state is. The in-process uniform_ tests were dropped: if they
    faulted they would kill the actor before the zero-init fix is exercised.
    """
    global _BISECT_DONE
    if not _debug_on() or _BISECT_DONE:
        return
    _BISECT_DONE = True
    import subprocess
    import sys

    keys = sorted(k for k in os.environ
                  if any(s in k.upper() for s in
                         ("CUDA", "OMP", "MKL", "THREAD", "RAY", "GPU", "NCCL", "PYTORCH")))
    print(f"[bisect b1] cid={pid} threads={torch.get_num_threads()} "
          f"interop={torch.get_num_interop_threads()} devs={torch.cuda.device_count()} "
          f"fenv={_fenv_mask()} "
          f"env={{{', '.join(f'{k}={os.environ[k][:60]}' for k in keys)}}}", flush=True)
    code = ("import torch\n"
            "print('sub threads', torch.get_num_threads(), 'devs', torch.cuda.device_count())\n"
            "torch.manual_seed(0)\n"
            "torch.empty(128, 3).uniform_(-.1, .1)\n"
            "print('sub cpu-ok')\n"
            "if torch.cuda.is_available():\n"
            "    torch.empty(128, 3, device='cuda').uniform_(-.1, .1)\n"
            "    torch.cuda.synchronize()\n"
            "    print('sub cuda-ok')\n")
    p = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=120)
    print(f"[bisect b2] rc={p.returncode} out={p.stdout.strip()!r} "
          f"err={p.stderr.strip()[-300:]!r}", flush=True)


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
        _bisect_once(pid)
    client = FlowerClient(data, cfg, pid)
    if _debug_on():
        print(f"[client-init cid={pid}] model OK", flush=True)
    return client.to_client()


_install_kaiming_probe()
app = ClientApp(client_fn=client_fn)
