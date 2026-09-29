"""Flower ServerApp.

Two strategies are provided:

1. FedAvg (Phase 3 baseline) -- ``strategy="fedavg"`` in run.py.
   Aggregates client weights by data size and reports a test-weighted
   macro-F1 across hospitals each round.

2. TopologyAwareStrategy (TKPA-FL) -- ``strategy="topology"`` in run.py.
   Round 1: requests topology profiles from all clients, builds JSD-based
   pairwise similarity S and personalised aggregation matrix A (cached).
   Subsequent rounds: computes personalised model V_i = sum_j A_ij * U_j for
   each hospital and dispatches them in configure_fit / configure_evaluate.
"""
from __future__ import annotations

import numpy as np

from flwr.common import (
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import (
    append_fed_metric, get_weights, init_model_on, load_client_graph, read_run_config,
)
from ..topology import (
    build_aggregation_matrix, build_similarity, deserialize_profile,
)


def _weighted_macro_f1(metrics: list) -> dict:
    """metrics: list of (num_examples, {"macro_f1": ...}) -> test-weighted mean.
    Also appends the round's value to the history file (run_simulation returns None)."""
    total = sum(n for n, _ in metrics)
    f1 = sum(n * m["macro_f1"] for n, m in metrics) / total if total else 0.0
    append_fed_metric(f1)
    return {"macro_f1": f1}


# ============================================================================
# Strategy 1: Plain FedAvg (unchanged baseline)
# ============================================================================

def server_fn(context: Context):
    cfg = read_run_config()
    init_weights = get_weights(init_model_on(load_client_graph(0), cfg))
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=cfg["n_clients"],
        initial_parameters=ndarrays_to_parameters(init_weights),
        evaluate_metrics_aggregation_fn=_weighted_macro_f1,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg["rounds"]))


# ============================================================================
# Strategy 2: TKPA-FL -- Topology-Kernel Personalized Aggregation
# ============================================================================

class TopologyAwareStrategy(FedAvg):
    """TKPA-FL: personalised aggregation using JSD over labeled-triple distributions.

    Inherits FedAvg for parameter serialisation utilities and client selection.
    Overrides configure_fit, aggregate_fit, configure_evaluate, aggregate_evaluate.
    """

    def __init__(self, n_clients: int, initial_parameters: Parameters,
                 kappa: float = 100.0, **fedavg_kwargs):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=n_clients,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=_weighted_macro_f1,
            **fedavg_kwargs,
        )
        self.n_clients = n_clients
        self.kappa = kappa

        # Topology state (built on round 1, cached thereafter)
        self._A: np.ndarray | None = None   # [H, H] aggregation matrix
        self._n: np.ndarray | None = None   # [H] training triple counts

        # Personalised models: pid -> list[np.ndarray]
        self._personalized: dict[int, list[np.ndarray]] = {}

        # cid (Flower string) -> partition_id (int) mapping
        self._cid_to_pid: dict[str, int] = {}

        # Per-round temporary storage
        self._round_weights: dict[int, list[np.ndarray]] = {}
        self._round_n: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # configure_fit
    # ------------------------------------------------------------------ #

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager):
        """Send each hospital its own personalised model (or W^0 on round 1)."""
        clients = client_manager.sample(
            num_clients=self.n_clients, min_num_clients=self.n_clients
        )

        fit_configs = []
        for client in clients:
            pid = self._cid_to_pid.get(client.cid)

            if server_round == 1 or pid is None or pid not in self._personalized:
                params_to_send = parameters
                config = {"send_profile": True} if server_round == 1 else {}
            else:
                params_to_send = ndarrays_to_parameters(self._personalized[pid])
                config = {}

            fit_configs.append((client, FitIns(params_to_send, config)))

        return fit_configs

    # ------------------------------------------------------------------ #
    # aggregate_fit
    # ------------------------------------------------------------------ #

    def aggregate_fit(self, server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures):
        """Collect local updates; on round 1 build A; compute V_i per hospital."""
        if not results:
            return None, {}

        # Clear per-round storage
        self._round_weights.clear()
        self._round_n.clear()
        round_profiles: dict[int, dict] = {}

        for client, fit_res in results:
            m = fit_res.metrics or {}

            # Build cid -> partition_id mapping
            pid = int(m.get("partition_id", -1))
            if pid >= 0:
                self._cid_to_pid[client.cid] = pid

            self._round_weights[pid] = parameters_to_ndarrays(fit_res.parameters)
            self._round_n[pid] = fit_res.num_examples

            # Deserialise profile if present (round 1 only)
            if "profile" in m:
                raw = m["profile"]
                if isinstance(raw, bytes):
                    round_profiles[pid] = deserialize_profile(raw)

        # ---- Round 1: build and cache the aggregation matrix ----
        if server_round == 1 and self._A is None:
            profiles_ordered = [
                round_profiles.get(pid, {}) for pid in range(self.n_clients)
            ]
            n_array = np.array([
                float(self._round_n.get(pid, 0)) for pid in range(self.n_clients)
            ])
            S = build_similarity(profiles_ordered, kappa=self.kappa)
            self._A = build_aggregation_matrix(S, n_array)
            self._n = n_array.copy()
            print(f"\n[TKPA-FL] Aggregation matrix built (round {server_round}).")
            print(f"[TKPA-FL] S (diagonal=1, off-diag range): "
                  f"min={S[S < 1].min() if (S < 1).any() else 1:.4f}  "
                  f"max={S[S < 1].max() if (S < 1).any() else 1:.4f}")
            print(f"[TKPA-FL] A row sums: {self._A.sum(axis=1).round(4)}")

        # ---- Personalised aggregation V_i = sum_j A_ij * U_j ----
        available_pids = [
            pid for pid in self._round_weights
            if pid >= 0 and pid < self.n_clients
        ]

        if self._A is not None and available_pids:
            for i in range(self.n_clients):
                responding = [j for j in available_pids]
                if not responding:
                    continue

                row = self._A[i, responding]
                row_sum = row.sum()
                if row_sum == 0:
                    continue
                row_norm = row / row_sum

                base_weights = self._round_weights[responding[0]]
                agg = [np.zeros_like(p) for p in base_weights]
                for k, j in enumerate(responding):
                    for p_idx, param in enumerate(self._round_weights[j]):
                        agg[p_idx] += row_norm[k] * param

                self._personalized[i] = agg

        # ---- Standard FedAvg model for Flower compatibility / logging ----
        total_n = sum(self._round_n.get(pid, 0) for pid in available_pids)
        if total_n == 0:
            return None, {}

        fedavg_weights: list[np.ndarray] | None = None
        for pid in available_pids:
            frac = self._round_n.get(pid, 0) / total_n
            for p_idx, param in enumerate(self._round_weights[pid]):
                if fedavg_weights is None:
                    fedavg_weights = [np.zeros_like(p) for p in self._round_weights[pid]]
                fedavg_weights[p_idx] += frac * param

        self.initial_parameters = ndarrays_to_parameters(fedavg_weights)
        return ndarrays_to_parameters(fedavg_weights), {}

    # ------------------------------------------------------------------ #
    # configure_evaluate
    # ------------------------------------------------------------------ #

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                            client_manager):
        """Send each hospital its personalised model for evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        clients = client_manager.sample(
            num_clients=self.n_clients, min_num_clients=self.n_clients
        )

        eval_configs = []
        for client in clients:
            pid = self._cid_to_pid.get(client.cid)
            if pid is not None and pid in self._personalized:
                params_to_send = ndarrays_to_parameters(self._personalized[pid])
            else:
                params_to_send = parameters
            eval_configs.append((client, EvaluateIns(params_to_send, {})))

        return eval_configs

    # ------------------------------------------------------------------ #
    # aggregate_evaluate
    # ------------------------------------------------------------------ #

    def aggregate_evaluate(self, server_round: int,
                            results: list[tuple[ClientProxy, EvaluateRes]],
                            failures):
        """Aggregate personalised evaluation metrics; log weighted macro-F1."""
        if not results:
            return None, {}

        total = sum(res.num_examples for _, res in results)
        f1_agg = (
            sum(res.num_examples * res.metrics.get("macro_f1", 0.0)
                for _, res in results) / total
            if total else 0.0
        )

        append_fed_metric(f1_agg)
        return f1_agg, {"macro_f1": round(f1_agg, 4)}


# ============================================================================
# ServerApp factories
# ============================================================================

def topology_server_fn(context: Context):
    """ServerApp factory for TKPA-FL topology-aware strategy."""
    cfg = read_run_config()
    init_weights = get_weights(init_model_on(load_client_graph(0), cfg))
    strategy = TopologyAwareStrategy(
        n_clients=cfg["n_clients"],
        initial_parameters=ndarrays_to_parameters(init_weights),
        kappa=float(cfg.get("kappa", 100.0)),
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg["rounds"]))


# Flower app handles (selected by run.py via the strategy= argument)
app = ServerApp(server_fn=server_fn)
topology_app = ServerApp(server_fn=topology_server_fn)
