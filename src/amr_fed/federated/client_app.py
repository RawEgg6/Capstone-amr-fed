"""Flower ClientApp — one simulated hospital.

Each client loads its own pre-built graph from disk (by partition id), trains a
few local epochs from the server's current weights, and evaluates the current
global model on its own held-out test triples.

SIGFPE forensics (notebook sets AMR_DEBUG_INIT=1 before flwr's ray.init): a
[client-init] line per construction, a [bisect] env/full-pipeline-subprocess probe
once per process, and a [kaiming] line per PyG lazy-weight reset, all flush=True so
they survive the fault. FlowerClient builds its model under _zero_weight_init so the
actor never calls a generator during construction, and model.py draws dropout masks
from the CPU generator, so an actor's training path has no CUDA-RNG op at all.
AMR_RNG_PROBE=1 (D3 only, never the arms) adds a fatal-if-poisoned in-process
[rng-probe entry] check at client_fn entry — a SIGFPE stack there means the message
path already delivered a poisoned RNG; 'ok' means the poison forms later.
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


def _probe_on() -> bool:
    return os.environ.get("AMR_RNG_PROBE", "").strip().lower() not in ("", "0", "false", "no")


def _fenv_mask() -> int | str:
    try:
        return int(ctypes.CDLL("libm.so.6").fegetexcept())
    except Exception as e:
        return f"err {e}"


def _rng_probe(tag: str) -> None:
    """Fatal-if-poisoned in-process CUDA RNG check (gated on AMR_RNG_PROBE, D3 only).

    SIGFPE cannot be caught, so this may only run where a crash is the datum we
    want: the stack position [rng-probe {tag}] says the state arriving at that
    point is already poisoned (client_fn entry = everything the message path and
    actor startup did), while 'uniform ok' + 'dropout ok' say it is clean and the
    poison — if any — forms later. Arms never set AMR_RNG_PROBE, so a working run
    is never killed by a probe.
    """
    if not _probe_on():
        return
    import torch.nn.functional as F
    print(f"[rng-probe {tag}] start fenv={_fenv_mask()}", flush=True)
    torch.empty(64, device=DEVICE).uniform_(-0.1, 0.1)
    print(f"[rng-probe {tag}] uniform ok", flush=True)
    F.dropout(torch.randn(128, device=DEVICE), p=0.3, training=True)
    print(f"[rng-probe {tag}] dropout ok", flush=True)


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

    b1 dumps the actor's env/thread/RNG-relevant state; b2 runs the WHOLE client
    pipeline (seed -> load graph -> init_model_on with REAL kaiming uniform_ ->
    train-mode encode -> raw CUDA uniform_/dropout) in a FRESH subprocess that
    inherits the actor's env. 'sub ALL-OK' = env + pipeline are clean in a fresh
    process (poison, if any, is live-actor-state only); any failure prints the
    last stage reached before rc!=0. In-process RNG tests stay out: if they
    faulted they would kill the actor before the real forensics run.
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
    if pid != 0:   # one full-pipeline subprocess per run is enough — 5 concurrent
        # GPU-heavy subprocesses (one per actor) would risk OOM on smaller GPUs
        print("[bisect b2] skipped (only cid 0 runs the pipeline probe)", flush=True)
        return
    import amr_fed
    src = os.path.dirname(os.path.dirname(os.path.abspath(amr_fed.__file__)))
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import torch, torch.nn.functional as F\n"
        "from amr_fed.federated.task import (seed_client, read_run_config, "
        "load_client_graph, init_model_on)\n"
        "print('sub start threads=%%s devs=%%s' %% (torch.get_num_threads(), "
        "torch.cuda.device_count()), flush=True)\n"
        "cfg = read_run_config()\n"
        "seed_client(cfg, %d)\n"
        "data = load_client_graph(%d)\n"
        "print('sub graph-ok', flush=True)\n"
        "model = init_model_on(data, cfg)\n"
        "print('sub init-ok (real kaiming uniform_)', flush=True)\n"
        "model.train()\n"
        "with torch.no_grad():\n"
        "    model.encode(data.x_dict, data.edge_index_dict)\n"
        "print('sub encode-train-ok (cpu-mask dropout)', flush=True)\n"
        "if torch.cuda.is_available():\n"
        "    torch.empty(64, device='cuda').uniform_(-.1, .1)\n"
        "    F.dropout(torch.randn(128, device='cuda'), p=0.3, training=True)\n"
        "    print('sub cuda-rng-ok', flush=True)\n"
        "print('sub ALL-OK', flush=True)\n"
    ) % (src, pid, pid)
    p = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=240)
    print(f"[bisect b2] rc={p.returncode} out={p.stdout.strip()!r} "
          f"err={p.stderr.strip()[-300:]!r}", flush=True)


def client_fn(context: Context):
    _rng_probe("entry")   # FIRST: if this SIGFPEs, the poison predates all of our code
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
