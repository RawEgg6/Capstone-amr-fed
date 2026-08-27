"""Small topology fingerprints used by topology-aware federated aggregation.

The fingerprint summarizes the structural characteristics of a hospital's
private graph without sending the graph itself to the server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .partition import _patient_homophily, _patient_hubness


@dataclass(frozen=True)
class TopologyFingerprint:
    """Compact structural summary of one hospital."""

    homophily: float
    hubness: float

    def as_dict(self) -> dict[str, float]:
        return {
            "homophily": float(self.homophily),
            "hubness": float(self.hubness),
        }


def compute_topology_fingerprint(df: pd.DataFrame) -> TopologyFingerprint:
    """Compute a small topology summary for one hospital's patient subset.

    The server receives only these summary values, not the underlying
    patient-level graph or data.
    """
    hom = _patient_homophily(df)
    hub = _patient_hubness(df)

    return TopologyFingerprint(
        homophily=float(np.nanmean(hom.to_numpy(dtype=float))),
        hubness=float(np.nanmean(hub.to_numpy(dtype=float))),
    )