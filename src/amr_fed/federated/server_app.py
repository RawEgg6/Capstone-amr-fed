"""Flower ServerApp — topology-aware federated aggregation (Phase 5).

Each client sends:
    - its model parameters
    - number of training examples
    - a compact topology fingerprint

The server uses topology similarity to modify the FedAvg weights.
No patient-level graph or raw patient data is sent to the server.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from flwr.common import (
    Context,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import (
    append_fed_metric,
    get_weights,
    init_model_on,
    load_client_graph,
    read_run_config,
)


def _weighted_metrics(metrics: list) -> dict:
    """Aggregate test metrics by test-set size.

    Also stores the per-hospital values so run.py can report the
    worst-hospital performance.
    """
    total = sum(n for n, _ in metrics)

    f1 = (
        sum(n * m["macro_f1"] for n, m in metrics) / total
        if total
        else 0.0
    )

    # NaN-safe weighted AUROC.
    auc_pairs = [
        (n, m["auroc"])
        for n, m in metrics
        if m.get("auroc") is not None
        and m["auroc"] == m["auroc"]
    ]

    auc = (
        sum(n * a for n, a in auc_pairs) / sum(n for n, _ in auc_pairs)
        if auc_pairs
        else float("nan")
    )

    per_client = {
        str(m.get("cid", i)): round(m["macro_f1"], 4)
        for i, (_, m) in enumerate(metrics)
    }

    per_client_auroc = {
        str(m.get("cid", i)): (
            round(m["auroc"], 4)
            if m.get("auroc") == m.get("auroc")
            else None
        )
        for i, (_, m) in enumerate(metrics)
    }

    append_fed_metric(
        f1,
        per_client,
        auroc=auc,
        per_client_auroc=per_client_auroc,
    )

    return {
        "macro_f1": f1,
        "auroc": auc,
    }


class TopologyAwareFedAvg(FedAvg):
    """FedAvg modified using client topology similarity.

    The client still contributes according to its amount of training data,
    but clients whose topology is similar to the population receive more
    aggregation weight.

    Similarity is computed from the two-dimensional fingerprint:
        [homophily, hubness]

    The fingerprint itself contains no patient-level information.
    """

    def __init__(
        self,
        *args,
        topology_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topology_temperature = topology_temperature

    @staticmethod
    def _topology_similarity(
        fingerprints: np.ndarray,
    ) -> np.ndarray:
        """Return one similarity score per client.

        Each client's fingerprint is compared with the mean topology
        fingerprint of all clients.

        The two axes are standardized first so that hubness cannot
        dominate homophily merely because it has a larger numeric scale.
        """
        if len(fingerprints) <= 1:
            return np.ones(len(fingerprints), dtype=np.float64)

        mean = fingerprints.mean(axis=0)
        std = fingerprints.std(axis=0)

        # Avoid division by zero for a constant topology axis.
        std = np.where(std < 1e-12, 1.0, std)

        z = (fingerprints - mean) / std
        distances = np.linalg.norm(z, axis=1)

        # Exponential similarity: 1.0 at the population centre,
        # progressively smaller for structurally distant clients.
        similarities = np.exp(-distances)

        return similarities

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Parameters | None, dict]:
        """Aggregate client updates using topology-aware weights."""

        if not results:
            return None, {}

        fingerprints = []
        examples = []

        valid_results = []

        for client, fit_res in results:
            metrics = fit_res.metrics

            hom = metrics.get("topology_homophily")
            hub = metrics.get("topology_hubness")

            if hom is None or hub is None:
                # A client without a fingerprint cannot participate in
                # topology-aware aggregation.
                continue

            fingerprints.append([float(hom), float(hub)])
            examples.append(float(fit_res.num_examples))
            valid_results.append((client, fit_res))

        if not valid_results:
            return super().aggregate_fit(
                server_round,
                results,
                failures,
            )

        fingerprints_np = np.asarray(
            fingerprints,
            dtype=np.float64,
        )

        examples_np = np.asarray(
            examples,
            dtype=np.float64,
        )

        similarity = self._topology_similarity(fingerprints_np)

        # Base FedAvg weight = number of training examples.
        #
        # Topology-aware weight = data size × topology similarity.
        weights = examples_np * similarity

        if weights.sum() <= 0:
            weights = examples_np.copy()

        weights = weights / weights.sum()

        print(
            f"\n[TopologyAwareFedAvg] round={server_round}"
        )

        for i, ((client, fit_res), sim, weight) in enumerate(
            zip(valid_results, similarity, weights)
        ):
            cid = fit_res.metrics.get("cid", client.cid)
            print(
                f"  client={cid} "
                f"hom={fingerprints_np[i, 0]:.4f} "
                f"hub={fingerprints_np[i, 1]:.4f} "
                f"similarity={sim:.4f} "
                f"weight={weight:.4f}"
            )

        # Convert each client's parameters to ndarrays and perform
        # the topology-aware weighted average.
        client_ndarrays = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in valid_results
        ]

        num_layers = len(client_ndarrays[0])

        aggregated = []

        for layer_idx in range(num_layers):
            layer = sum(
                weights[i] * client_ndarrays[i][layer_idx]
                for i in range(len(client_ndarrays))
            )
            aggregated.append(layer)

        aggregated_parameters = ndarrays_to_parameters(aggregated)

        return aggregated_parameters, {}


def server_fn(context: Context):
    cfg = read_run_config()

    # Initial global model comes from client 0's graph.
    init_weights = get_weights(
        init_model_on(
            load_client_graph(0),
            cfg,
        )
    )

    strategy = TopologyAwareFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=cfg["n_clients"],
        initial_parameters=ndarrays_to_parameters(init_weights),
        evaluate_metrics_aggregation_fn=_weighted_metrics,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(
            num_rounds=cfg["rounds"]
        ),
    )


app = ServerApp(server_fn=server_fn)