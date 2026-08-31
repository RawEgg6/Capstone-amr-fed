# Phase 5 – Topology-Aware Federated Aggregation

## Summary

Implemented a topology-aware extension of the existing FedAvg setup for federated learning across hospitals.

Each hospital/client now calculates a compact topology fingerprint from its local graph using **homophily** and **hubness**. These values are shared with the server along with the model update, without sending patient-level graph data.

On the server side, `TopologyAwareFedAvg` was implemented to adjust the normal FedAvg aggregation weights based on both the client's training data size and its topology similarity to the overall client population.

### Main Changes

- Added topology information to the client updates.
- Stored each client's topology fingerprint using homophily and hubness.
- Implemented `TopologyAwareFedAvg` for topology-based aggregation.
- Kept patient-level graph information local to each hospital.
- Added handling for topology-aware aggregation while retaining FedAvg as the baseline.

### Validation

- All **37 tests passed**.
- Ran the topology-split federated experiment using **3 seeds (42, 43, 44)**.
- Verified the resulting FedAvg performance and worst-hospital performance across the runs.