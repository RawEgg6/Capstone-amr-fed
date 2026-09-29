"""Offline tests for the Phase-5 topology-aware strategy wiring.

Needs torch + flwr (the federated stack); SKIPs cleanly without them and runs on
Colab/GPU:  python tests/test_topology_strategy.py

What it guards (without running any simulation):
- fingerprint payloads decode correctly; garbage -> None (FedAvg fallback)
- weighted_average mixes parameter lists by the given weights
- TopologyAwareStrategy constructs with defaults
- aggregate_fit with NO fingerprints falls back to size-proportional weights
- aggregate_fit WITH fingerprints uses topology weights (T=inf -> uniform)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    try:
        import json

        import numpy as np
        from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
        from amr_fed.federated import strategy as S
        from amr_fed.topology import FEATURE_NAMES
    except ImportError as exc:  # torch / flwr / torch_geometric not available here
        print(f"SKIP: federated stack not installed ({exc})")
        return

    D = len(FEATURE_NAMES)

    def fitres(arrs, n, metrics):
        return FitRes(status=Status(code=Code.OK, message="ok"),
                      parameters=ndarrays_to_parameters(arrs),
                      num_examples=n, metrics=metrics)

    # 1) decode: valid JSON round-trips; garbage -> None
    good = json.dumps([float(i) for i in range(D)])
    assert S.decode_fingerprint(good) is not None
    assert np.allclose(S.decode_fingerprint(good), np.arange(D, dtype=float))
    for bad in (None, "not-json{{{", json.dumps([1.0, 2.0]), "[1,2, three]",
                json.dumps([float("nan")] * D)):
        assert S.decode_fingerprint(bad) is None, bad

    # 2) weighted_average mixes per-client parameter lists by weight
    a = [np.array([1.0, 2.0]), np.array([10.0])]
    b = [np.array([3.0, 4.0]), np.array([20.0])]
    out = S.weighted_average([a, b], np.array([0.25, 0.75]))
    assert np.allclose(out[0], [0.25 * 1 + 0.75 * 3, 0.25 * 2 + 0.75 * 4])
    assert np.allclose(out[1], [0.25 * 10 + 0.75 * 20])

    # 3) strategy constructs with defaults (distinctiveness, T=1.0)
    strat = S.TopologyAwareStrategy()
    assert strat.temperature == 1.0 and strat.mode == "distinctiveness"

    # 4) no fingerprints -> size-proportional fallback (1 and 3 examples)
    r = [(None, fitres([np.array([1.0])], 1, {})),
         (None, fitres([np.array([3.0])], 3, {}))]
    params, metrics = strat.aggregate_fit(1, r, [])
    from flwr.common import parameters_to_ndarrays
    assert np.allclose(parameters_to_ndarrays(params)[0], [0.25 * 1 + 0.75 * 3])
    assert metrics["topology_mode"] == "fedavg-fallback"

    # 5) fingerprints present -> topology weights; T=inf must be uniform
    fp0 = json.dumps([0.0] * D)
    fp1 = json.dumps([5.0] * D)
    r2 = [(None, fitres([np.array([1.0])], 1, {S.FIT_METRIC_KEY: fp0})),
          (None, fitres([np.array([3.0])], 3, {S.FIT_METRIC_KEY: fp1}))]
    _, m2 = strat.aggregate_fit(2, r2, [])
    assert m2["topology_mode"] == "topology-distinctiveness"
    assert abs(m2["topo_weight_client_0"] + m2["topo_weight_client_1"] - 1.0) < 1e-9
    # size-proportional would be exactly (0.25, 0.75); topology must differ here
    assert abs(m2["topo_weight_client_0"] - 0.25) > 1e-6

    strat_inf = S.TopologyAwareStrategy(temperature=float("inf"))
    _, m3 = strat_inf.aggregate_fit(3, r2, [])
    assert abs(m3["topo_weight_client_0"] - 0.5) < 1e-9
    assert abs(m3["topo_weight_client_1"] - 0.5) < 1e-9

    # 6) one malformed payload poisons nothing -> whole round falls back safely
    r3 = [(None, fitres([np.array([1.0])], 1, {S.FIT_METRIC_KEY: fp0})),
          (None, fitres([np.array([3.0])], 3, {S.FIT_METRIC_KEY: "junk"}))]
    _, m4 = strat.aggregate_fit(4, r3, [])
    assert m4["topology_mode"] == "fedavg-fallback"
    assert abs(m4["topo_weight_client_0"] - 0.25) < 1e-9

    print("OK: topology strategy tests passed.")


if __name__ == "__main__":
    main()
