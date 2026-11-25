"""Label-inversion shift (`shift25`, notebook 22) at 3% data with the shuffled control (pre-registration: docs/2026-11-26-inversion-lowdata-preregistration.md).

    python scripts/2026-11-26-inversion-lowdata.py gate          # fedavg + oracle, prints the headroom gate
    python scripts/2026-11-26-inversion-lowdata.py fingerprint   # v3_ab + v3_ab_shuffled (refuses to run unless the gate passed)
    python scripts/2026-11-26-inversion-lowdata.py report
Resume-safe (checkpoints after every seed). Needs ARMD_DIR.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amr_fed.data_loader import load_cohort_frame  # noqa: E402
from amr_fed.federated import budget_runs as BR  # noqa: E402
from amr_fed.federated import final_protocol as P  # noqa: E402
from amr_fed.federated import shift_runs as S  # noqa: E402
from amr_fed.federated.analysis import _fmt_stat, _stat  # noqa: E402

EXP, FRAC, SEEDS = "shift25", 0.03, (42, 43, 44, 45, 46)
OUT = ROOT / "results" / "amr_inversion_f030.json"


def _res():
    return json.load(open(OUT)) if OUT.exists() else {}


def run(arms):
    import ray
    os.environ.pop("AMR_RNG_PROBE", None)   # SIGFPE guard, see AGENTS.md
    dfx, part = S.build(EXP, load_cohort_frame())
    specs = S.arm_specs()
    kw = dict(seeds=SEEDS, df=dfx, rounds=8, local_epochs=BR.LOCAL_EPOCHS, local_only_epochs=40, hidden=128,
              partition_fn=part, n_clients=10, train_frac=FRAC)
    res = BR.run_part(_res(), {k: specs[k] for k in arms}, prefix=f"inversion/{FRAC}", kw=kw,
                      checkpoint=lambda r: P.save_json(r, OUT), ray_shutdown=ray.shutdown)
    P.save_json(res, OUT)


def gate():
    run(["fedavg", "oracle"])
    return report(only_gate=True)


def report(only_gate=False):
    res = _res()
    print(S.report(res))
    from amr_fed.federated.budget import finetune_score
    import numpy as np
    v = {k: np.array([finetune_score(r)["mean"] for r in res[k]["runs"]]) for k in S.ARMS if k in res}
    d = v["oracle"] - v["fedavg"]
    ok = _stat(d)["lo"] > 0
    if only_gate:
        print("GATE PASSED: the oracle beats FedAvg + fine-tune." if ok else
              "GATE FAILED: the oracle does not beat FedAvg + fine-tune. Stop; report a null result.")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:] or ["report"]
    if a[0] == "gate":
        gate()
    elif a[0] == "fingerprint":
        assert report(only_gate=True), "the headroom gate has not passed; per the pre-registration nothing else is run"
        run(["v3_ab", "v3_ab_shuffled"])
        report()
    else:
        report()
