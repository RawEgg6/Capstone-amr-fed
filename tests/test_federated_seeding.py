"""Offline tests for deterministic client seeding (Phase-3 reproducibility fix).

Seeding lives in `task.py`, which imports torch / torch_geometric. These tests SKIP
cleanly when the federated stack isn't installed, and run properly on Colab/GPU:
    python tests/test_federated_seeding.py

What it guards: FedAvg used to be non-deterministic run-to-run because Flower's Ray
clients init models in separate processes with fresh RNGs. `seed_client(cfg, cid)` must
make each client's RNG stream reproducible (same cfg+cid -> same stream) and distinct
per client / per seed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    try:
        import numpy as np
        import torch
        from amr_fed.federated import task
    except ImportError as exc:  # torch / torch_geometric / flwr not available here
        print(f"SKIP: federated stack not installed ({exc})")
        return

    def stream(cfg, cid):
        task.seed_client(cfg, cid)
        return [round(v, 6) for v in torch.rand(5).tolist()], round(float(np.random.rand()), 6)

    # 1) same cfg + same cid -> identical stream (the reproducibility guarantee)
    a = stream({"seed": 42}, 3)
    b = stream({"seed": 42}, 3)
    assert a == b, f"same seed+cid must reproduce the RNG stream: {a} != {b}"

    # 2) different cid -> different stream
    assert stream({"seed": 42}, 4) != a, "different cid must give a different stream"

    # 3) different base seed -> different stream
    assert stream({"seed": 43}, 3) != a, "different base seed must give a different stream"

    # 4) client_seed is exactly base + cid, with the config default when absent
    assert task.client_seed({"seed": 100}, 7) == 107
    assert task.client_seed({}, 0) == 42          # falls back to config.SEED

    # 5) the seed is carried into run_config.json (server + clients read it back)
    try:
        from amr_fed.federated.run import _run_config
        cfg = _run_config(5, 10, 6, hidden=128, seed=7)
        assert cfg["seed"] == 7, cfg
    except ImportError as exc:
        print(f"  (skipping _run_config check — flwr not installed: {exc})")

    print("OK: federated seeding tests passed.")


if __name__ == "__main__":
    main()
