"""Phase 5 — Topology-Aware Aggregation Strategy.

Replaces FedAvg's uniform weight averaging with a topology-weighted scheme:
each client's model update is weighted by how *similar* its local graph topology
is to the global topology fingerprint, so hospitals whose graph structure is
representative contribute more to the global model.

## Design (Path B — locked 2026-08-17)
Canonical evaluation splits: organism-community (5 hospitals) and specimen (3).
Goal: topology-aware macro-F1 > FedAvg-best by >= 0.005 (phase5_gate threshold).

## Architecture
TopologyAwareStrategy subclasses FedAvg and overrides only aggregate_fit.
Everything else (evaluate, configure_fit/evaluate, initial parameters) is
inherited from FedAvg unchanged — comparison is exactly apples-to-apples.

## Topology fingerprint (to implement in Phase 4 / topology.py)
A small fixed-length vector per hospital summarising its local graph structure.
Candidate features:
  - mean homophily deviation (already computed in partition._tested_edge_homophily)
  - log1p(#organisms), log1p(#antibiotics), log1p(#tested edges)
  - resistance rate (scalar, from train_mask labels)
  - degree distribution stats: mean / std of organism-degree and antibiotic-degree
  - log1p(#patients) for size calibration

Sent from each client as a JSON blob inside fit metrics (no raw patient data).

## Aggregation weight formula
Given fingerprints f_i for client i and global reference f_global:
  sim_i  = cosine_similarity(f_i, f_global)
  w_i    = softmax(sim_i / temperature)
  temperature -> 0 : winner-takes-all (most similar client dominates)
  temperature -> inf: uniform FedAvg (baseline sanity check)

References:
  FedGTA (Li et al. 2023) — topology-aware FL on graphs
  AdaFGL (Li et al. 2024) — adaptive topology weighting
  OpenFGL benchmark (2024) — evaluation framework
"""
from __future__ import annotations

from typing import Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class TopologyAwareStrategy(FedAvg):
    """Topology-weighted FedAvg aggregation for heterogeneous GNNs.

    Inherits all FedAvg behaviour; overrides only ``aggregate_fit`` to replace
    the uniform weight with a topology-similarity weight derived from each
    client's graph fingerprint.

    Args:
        temperature:  Softmax temperature for the similarity weights.
                      temperature=inf -> uniform FedAvg (sanity-check baseline).
                      temperature=1.0 -> default; tune on validation set.
        **kwargs:     Passed to FedAvg (fraction_fit, initial_parameters, etc.)
    """

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    # ------------------------------------------------------------------
    # OVERRIDE: aggregate_fit
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate client weights using topology-similarity weights.

        Steps:
          1. Extract topology fingerprints from each client's metrics dict.
          2. Compute global reference fingerprint (mean of clients').
          3. Compute per-client similarity scores -> softmax weights.
          4. Weighted-average the model parameter arrays by those weights.
          5. Return aggregated parameters (same interface as FedAvg).

        STUB: fingerprint extraction returns None -> falls back to uniform
        FedAvg weights (identical to base class). Replace _extract_fingerprint
        and _compute_weights to activate topology weighting.
        """
        if not results:
            return None, {}

        fingerprints = [self._extract_fingerprint(fit_res.metrics)
                        for _, fit_res in results]
        weights = self._compute_weights(fingerprints, results)

        ndarrays_list = [parameters_to_ndarrays(fit_res.parameters)
                         for _, fit_res in results]
        aggregated = _weighted_average(ndarrays_list, weights)
        parameters = ndarrays_to_parameters(aggregated)

        metrics = {f"topo_weight_client_{i}": float(w)
                   for i, w in enumerate(weights)}
        metrics["temperature"] = self.temperature
        return parameters, metrics

    # ------------------------------------------------------------------
    # Helpers — implement these to activate topology weighting
    # ------------------------------------------------------------------

    def _extract_fingerprint(self, metrics: dict) -> np.ndarray | None:
        """Extract topology fingerprint vector from a client's fit metrics.

        TODO (Phase 4): clients must pack their fingerprint into fit metrics:
            import json
            metrics["topology_fingerprint"] = json.dumps(fingerprint.tolist())
        Then decode here:
            raw = metrics.get("topology_fingerprint")
            return np.array(json.loads(raw), dtype=np.float32) if raw else None

        Returns None -> triggers uniform-weight fallback.
        """
        return None  # STUB

    def _compute_weights(
        self,
        fingerprints: list[np.ndarray | None],
        results: list[tuple[ClientProxy, FitRes]],
    ) -> np.ndarray:
        """Compute per-client aggregation weights from topology fingerprints.

        Falls back to size-proportional FedAvg if any fingerprint is None.

        TODO (Phase 5): replace the fallback block with:
            global_fp = np.mean(valid_fps, axis=0)
            sims = [cosine_sim(fp, global_fp) for fp in fingerprints]
            return softmax(np.array(sims) / self.temperature)
        """
        if any(fp is None for fp in fingerprints):
            # Fallback: size-proportional (matches flwr FedAvg default)
            num_examples = np.array([r.num_examples for _, r in results],
                                    dtype=np.float64)
            total = num_examples.sum()
            return num_examples / total if total > 0 else np.ones(len(results)) / len(results)

        # Topology-similarity weights (activated once fingerprints available)
        fps = np.array(fingerprints, dtype=np.float32)
        global_fp = fps.mean(axis=0)
        sims = np.array([_cosine_sim(fp, global_fp) for fp in fps])
        return _softmax(sims / max(self.temperature, 1e-8))


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _weighted_average(ndarrays_list: list[list[np.ndarray]],
                      weights: np.ndarray) -> list[np.ndarray]:
    """Weighted average of a list of parameter-array-lists (weights sum to 1)."""
    assert abs(weights.sum() - 1.0) < 1e-5, f"weights must sum to 1, got {weights.sum()}"
    return [sum(w * arrays[i] for w, arrays in zip(weights, ndarrays_list))
            for i in range(len(ndarrays_list[0]))]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0 if either is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()
