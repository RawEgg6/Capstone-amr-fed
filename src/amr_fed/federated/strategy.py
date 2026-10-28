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
from flwr.common import (EvaluateIns, FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..topology import FEATURE_NAMES, TopologyFingerprint, aggregation_weights
from .task import FIT_METRIC_KEY, save_global_params


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
            "invsize" is a CONTROL, not a method: weights proportional to 1/n_train,
            ignoring the fingerprint entirely.
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

        if self.mode == "invsize":
            # Control arm: weight each hospital by 1/n_train (small hospitals count more), with
            # NO graph information. Tests whether topo is just "don't size-weight".
            inv = 1.0 / np.maximum(np.array([float(r.num_examples) for _, r in results]), 1.0)
            weights = inv / inv.sum()
            ndarrays_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            parameters = ndarrays_to_parameters(weighted_average(ndarrays_list, weights))
            metrics: dict[str, Scalar] = {f"topo_weight_client_{i}": float(w)
                                          for i, w in enumerate(weights)}
            metrics["topology_mode"] = "invsize"
            return parameters, metrics

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


class SavingFedAvg(FedAvg):
    """Plain FedAvg that also writes the aggregated global model to disk every round, so the driver
    can fine-tune the final model at each hospital (the "FedAvg + fine-tune" baseline). Aggregation
    itself is untouched, so results equal ``FedAvg``."""

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            save_global_params(parameters_to_ndarrays(parameters))
        return parameters, metrics


class PersonalizedStrategy(FedAvg):
    """Topo v3: every hospital gets its OWN model each round.

    After local training, hospital i's model is ``sum_j W[i, j] * theta_j`` (theta_j = hospital j's
    trained weights); the next round's training starts from that personalized model, and evaluation
    scores each hospital on its own model. ``W`` is row-stochastic (see
    ``topology.personalized_weights``): the diagonal is largest, and hospitals with similar
    fingerprints get more weight. Hospitals are identified by the ``cid`` each client reports in its
    fit metrics, because Flower's own client ids are not the partition ids.
    """

    def __init__(self, W, **kwargs):
        super().__init__(**kwargs)
        self.W = np.asarray(W, dtype=np.float64)
        self.personal: dict[int, list[np.ndarray]] = {}   # partition cid -> personalized weights
        self.proxy2cid: dict[str, int] = {}               # flwr proxy id -> partition cid

    def configure_fit(self, server_round, parameters, client_manager):
        base = super().configure_fit(server_round, parameters, client_manager)
        if not self.personal:                              # round 1: everyone starts from the same init
            return base
        out = []
        for proxy, ins in base:
            cid = self.proxy2cid.get(proxy.cid)
            if cid in self.personal:
                ins = FitIns(ndarrays_to_parameters(self.personal[cid]), ins.config)
            out.append((proxy, ins))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        trained: dict[int, list[np.ndarray]] = {}
        for proxy, fit_res in results:
            cid = int(fit_res.metrics["cid"])
            self.proxy2cid[proxy.cid] = cid
            trained[cid] = parameters_to_ndarrays(fit_res.parameters)
        n = self.W.shape[0]
        if sorted(trained) != list(range(n)):
            raise RuntimeError(f"personalized aggregation needs all {n} hospitals, got {sorted(trained)}")
        ordered = [trained[j] for j in range(n)]
        for i in range(n):
            self.personal[i] = weighted_average(ordered, self.W[i])
        mean = weighted_average(ordered, np.full(n, 1.0 / n))   # placeholder "global" model (unused for scoring)
        return ndarrays_to_parameters(mean), {"topology_mode": "personalized"}

    def configure_evaluate(self, server_round, parameters, client_manager):
        base = super().configure_evaluate(server_round, parameters, client_manager)
        out = []
        for proxy, ins in base:
            cid = self.proxy2cid.get(proxy.cid)
            if cid in self.personal:
                ins = EvaluateIns(ndarrays_to_parameters(self.personal[cid]), ins.config)
            out.append((proxy, ins))
        return out
