"""Phase 5 (v0.5) — topology-aware aggregation strategy.

Fresh implementation (the old branch scaffold is abandoned). ``TopologyAwareStrategy``
subclasses Flower's ``FedAvg`` and overrides *only* ``aggregate_fit``: instead of
size-proportional weights it mixes client models by fingerprint similarity
(see ``amr_fed.topology``).

Client contract: each client's ``fit()`` metrics must carry
``metrics["topology_fingerprint"] = json.dumps(list_of_14_floats)``
in ``FEATURE_NAMES`` order, written by ``client_app`` from ``run_config.json``
(computed driver-side in ``run_fedavg``). Missing or malformed fingerprints fall back
to size-proportional weights (plain FedAvg) with a warning, so a run never dies
mid-simulation.
"""
from __future__ import annotations

import json
from typing import Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..topology import FEATURE_NAMES, TopologyFingerprint, aggregation_weights
from .task import FIT_METRIC_KEY


def decode_fingerprint(raw) -> np.ndarray | None:
    """Parse a client's fingerprint payload back into a vector.

    Returns None for anything missing/malformed (triggers the FedAvg fallback)."""
    try:
        if raw is None:
            return None
        vals = json.loads(raw) if isinstance(raw, str) else list(raw)
        arr = np.asarray(vals, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(FEATURE_NAMES):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr
    except (ValueError, TypeError):
        return None


def size_proportional_weights(results: list[tuple[ClientProxy, FitRes]]) -> np.ndarray:
    """Plain-FedAvg fallback weights (matches flwr's FedAvg default)."""
    nums = np.array([float(r.num_examples) for _, r in results], dtype=np.float64)
    total = nums.sum()
    if total <= 0:
        return np.full(len(results), 1.0 / len(results), dtype=np.float64)
    return nums / total


def weighted_average(ndarrays_list: list[list[np.ndarray]],
                     weights: np.ndarray) -> list[np.ndarray]:
    """Weighted average of per-client parameter lists (weights must sum to 1)."""
    w = np.asarray(weights, dtype=np.float64)
    assert abs(w.sum() - 1.0) < 1e-6, f"weights must sum to 1, got {w.sum()}"
    return [sum(layer * wi for layer, wi in zip(layers, w))
            for layers in zip(*ndarrays_list)]


class TopologyAwareStrategy(FedAvg):
    """FedAvg with topology-similarity mixing weights.

    Everything except ``aggregate_fit`` is inherited from FedAvg unchanged, so the
    comparison against the baseline is exactly apples-to-apples (same sampling,
    same evaluation, same initial parameters).

    Args:
        temperature: softmax temperature for the similarity weights.
            ``inf`` -> uniform weights (exactly FedAvg; sanity-check anchor).
        mode: "consensus" (up-weight typical hospitals) or "distinctiveness"
            (up-weight the odd hospital out). Default "distinctiveness": our
            failure mode is the rare hospital that averaging erases.
        exclude / cv_floor: forwarded to ``topology.aggregation_weights``.
    """

    def __init__(self, temperature: float = 1.0, mode: str = "distinctiveness",
                 exclude=("resistance_rate",), cv_floor: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.mode = mode
        self.exclude = tuple(exclude)
        self.cv_floor = cv_floor

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Mix client weights by fingerprint similarity (fallback: size-proportional)."""
        if not results:
            return None, {}

        decoded = [decode_fingerprint(fit_res.metrics.get(FIT_METRIC_KEY))
                   for _, fit_res in results]
        if any(fp is None for fp in decoded):
            print(f"  [topology] round {server_round}: "
                  f"{sum(fp is None for fp in decoded)}/{len(decoded)} clients sent no "
                  f"usable fingerprint -> size-proportional (FedAvg) fallback")
            weights = size_proportional_weights(results)
            used = "fedavg-fallback"
        else:
            fps = [TopologyFingerprint(fp) for fp in decoded]
            weights = aggregation_weights(
                fps, temperature=self.temperature, mode=self.mode,
                exclude=self.exclude, cv_floor=self.cv_floor)
            used = f"topology-{self.mode}"

        ndarrays_list = [parameters_to_ndarrays(fit_res.parameters)
                         for _, fit_res in results]
        parameters = ndarrays_to_parameters(weighted_average(ndarrays_list, weights))

        metrics: dict[str, Scalar] = {
            f"topo_weight_client_{i}": float(w) for i, w in enumerate(weights)}
        metrics["topology_mode"] = used
        metrics["temperature"] = float(self.temperature)
        return parameters, metrics
