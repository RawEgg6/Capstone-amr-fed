"""Does the model need the organism / antibiotic resistance-rate inputs? (2026-11-26, diagnostic, not pre-registered.)

Same setup as the pair-rate split at 3% data, fedavg arm only (gives local-only, FedAvg, FedAvg + fine-tune), seeds 42-46, but the last column of the
organism and antibiotic node features (each hospital's own train-label resistance rate, graph_build.py) is set to 0 (its z-scored mean).
Compare with `results/amr_pairrate_f030.json` (same data, same seeds, rates on).

    python scripts/2026-11-26-rate-input-ablation.py                  # fedavg arm (resume-safe), prints the rates on/off comparison
    python scripts/2026-11-26-rate-input-ablation.py oracle [arm ...]  # more arms with the rates off, prints their gain over FedAvg + fine-tune
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amr_fed.federated import budget_runs as BR  # noqa: E402
from amr_fed.federated import final_protocol as P  # noqa: E402
from amr_fed.federated import task as T  # noqa: E402
from amr_fed.federated.analysis import _fmt_stat, _stat  # noqa: E402

spec = importlib.util.spec_from_file_location("pl", ROOT / "scripts" / "2026-11-25-pair-rate-lowdata.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

FRAC = pl.GATE_FRAC
OUT = ROOT / "results" / "amr_pairrate_norate_f030.json"

_orig = T.build_arrays


def _no_rates(*a, **k):
    out = _orig(*a, **k)
    for t in ("organism", "antibiotic"):
        x = out["x"][t].copy()
        x[:, -1] = 0.0      # the rate column is last (graph_build.build_arrays)
        out["x"][t] = x
    return out


def main():
    import ray
    os.environ.pop("AMR_RNG_PROBE", None)
    T.build_arrays = _no_rates
    res = json.load(open(OUT)) if OUT.exists() else {}
    kw = dict(seeds=pl.SEEDS, df=pl.data(), rounds=8, local_epochs=BR.LOCAL_EPOCHS, local_only_epochs=40, hidden=128,
              n_clients=pl.pr.N_HOSP, train_frac=FRAC, partition_fn=pl.pr.partition_fn)
    arms = ["fedavg", *sys.argv[1:]]
    res = BR.run_part(res, {k: pl.ALL_ARMS[k] for k in arms}, prefix="norate/0.03", kw=kw,
                      checkpoint=lambda r: P.save_json(r, OUT), ray_shutdown=ray.shutdown)
    P.save_json(res, OUT)
    on, off = pl.values(json.load(open(pl.path(FRAC)))), pl.values(res)
    print(f"rate inputs on vs off, shifted pair-rate data, {FRAC * 100:g}% (unweighted mean F1)")
    for k in ("local-only (40)", "fedavg (32)", "fedavg+ft"):
        d = off[k] - on[k]
        print(f"  {k:<18} rates on {on[k].mean():.4f} | rates off {off[k].mean():.4f}")
        print(_fmt_stat("   off - on", _stat(d), d))
    for name, v in (("rates on", on), ("rates off", off)):
        d = v["fedavg+ft"] - v["local-only (40)"]
        print(_fmt_stat(f"fedavg+ft - local-only, {name}", _stat(d), d))
    for k in arms[1:]:
        d = off[f"{k}+ft"] - off["fedavg+ft"]
        print(f"  {k}+ft rates off {off[f'{k}+ft'].mean():.4f}")
        print(_fmt_stat(f"   {k}+ft - fedavg+ft, rates off", _stat(d), d))


if __name__ == "__main__":
    main()
