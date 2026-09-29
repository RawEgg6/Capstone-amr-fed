"""v0 topology fingerprint + client similarity for topology-aware aggregation.

Standalone and dependency-light (numpy + pandas only; no torch, no Flower). This is math
only — nothing here is wired into the federated pipeline yet.

**Why this exists (see docs/2026-09-28-topology-module-research.md).**
Our simulated hospitals are *patient subsets of one dataset*: they share the same
organism-antibiotic backbone. Classic structural statistics (degree distribution, graph
density, motif counts) are therefore nearly identical across hospitals and carry almost
no signal about how clients differ. This module fingerprints the signals that *do* move:
local sizes/degrees, the resistance rate, the structural axes **homophily** (is
resistance clustered or scattered?) and **hubness** (how broad is the drug repertoire?),
and the leakage-safe **patient-history** signal.

A fingerprint is a short vector per hospital. Only such vectors would ever leave a
client. ``aggregation_weights`` turns a set of fingerprints into server-side mixing
weights, with a temperature dial whose limit is uniform FedAvg.

**A note on the similarity math.** Features are z-scored across clients before comparison
(they have wildly different scales). After z-scoring, the population centroid is exactly
the origin, where cosine similarity is undefined — so centrality is measured as
standardised Euclidean distance to the centroid (small = typical, large = odd-one-out).
Pairwise cosine is still available for inspection and future clustering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import config
from .data_loader import PK, ORG, ABX
from .partition import _patient_homophily, _patient_hubness

# Stable feature order — once fixed, changing it invalidates saved fingerprints/runs.
FEATURE_NAMES: tuple[str, ...] = (
    "log_patients",          # size
    "log_organisms",         # local bug variety
    "log_antibiotics",       # local drug variety
    "log_tested_edges",      # local bug-drug pairs actually observed
    "tested_density",        # how much of the local menu got used
    "org_degree_mean",       # how broad local bugs are
    "org_degree_std",
    "abx_degree_mean",       # how broad local drugs are
    "abx_degree_std",
    "resistance_rate",       # local label balance (train frame)
    "homophily_mean",        # resistance clustered vs scattered
    "hubness_mean",          # drug-repertoire breadth
    "history_prior_rate",    # leakage-safe prior-resistance summary
    "n_communities",         # local community richness (default off)
)

_HISTORY_IDX = FEATURE_NAMES.index("history_prior_rate")


@dataclass(frozen=True)
class TopologyFingerprint:
    """One hospital's compact structural/statistical summary."""

    features: np.ndarray
    names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        arr = np.asarray(self.features, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(self.names):
            raise ValueError(
                f"features must be 1-D of length {len(self.names)}, got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"fingerprint contains non-finite values: {arr}")
        object.__setattr__(self, "features", arr)

    def as_dict(self) -> dict[str, float]:
        """JSON-friendly form (this is what would travel client -> server)."""
        return {n: float(v) for n, v in zip(self.names, self.features)}

    def tolist(self) -> list[float]:
        return self.features.tolist()

    def __len__(self) -> int:
        return len(self.features)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.features, dtype=dtype)


def _n_louvain_communities(df: pd.DataFrame) -> float:
    """#Louvain communities on the local organism-antibiotic weighted graph.

    Returns 0.0 when networkx is unavailable, keeping v0 dependency-light."""
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        return 0.0
    w = df.groupby([ORG, ABX]).size().reset_index(name="w")
    g = nx.Graph()
    g.add_weighted_edges_from(w[[ORG, ABX, "w"]].itertuples(index=False, name=None))
    if g.number_of_edges() == 0:
        return 0.0
    return float(len(louvain_communities(g, weight="weight", seed=config.SEED)))


def compute_fingerprint(df: pd.DataFrame, *, include_history: bool = True,
                        include_communities: bool = False) -> TopologyFingerprint:
    """Compute one hospital's fingerprint from its per-test frame (data_loader output).

    Pure pandas; deterministic. ``include_history`` adds the leakage-safe
    patient-history prior-resistance summary (the Phase-1 winning signal);
    ``include_communities`` adds a Louvain community count (needs networkx, off in v0).
    """
    if df is None or len(df) == 0:
        raise ValueError("compute_fingerprint needs a non-empty frame")

    n_pat = int(df[PK].nunique())
    n_org = int(df[ORG].nunique())
    n_abx = int(df[ABX].nunique())
    pairs = df[[ORG, ABX]].drop_duplicates()
    n_tested = int(len(pairs))
    density = float(n_tested / (n_org * n_abx)) if n_org and n_abx else 0.0

    org_deg = pairs.groupby(ORG).size().to_numpy(dtype=float)
    abx_deg = pairs.groupby(ABX).size().to_numpy(dtype=float)

    resistance_rate = float(df["label"].mean())
    homophily = float(_patient_homophily(df).mean())
    hubness = float(_patient_hubness(df).mean())

    if include_history:
        from .graph_build import _history_raw
        raw = _history_raw(df, global_rate=resistance_rate)
        history_rate = float(np.nanmean(raw[:, 1]))
    else:
        history_rate = 0.0

    n_comm = _n_louvain_communities(df) if include_communities else 0.0

    feats = np.array([
        np.log1p(n_pat), np.log1p(n_org), np.log1p(n_abx), np.log1p(n_tested),
        density,
        float(org_deg.mean()) if org_deg.size else 0.0,
        float(org_deg.std(ddof=0)) if org_deg.size else 0.0,
        float(abx_deg.mean()) if abx_deg.size else 0.0,
        float(abx_deg.std(ddof=0)) if abx_deg.size else 0.0,
        resistance_rate, homophily, hubness, history_rate, n_comm,
    ], dtype=np.float64)
    return TopologyFingerprint(features=feats)


def fingerprint_matrix(fingerprints: Sequence[TopologyFingerprint]) -> np.ndarray:
    """Stack fingerprints into an (n_clients, n_features) matrix."""
    if len(fingerprints) == 0:
        raise ValueError("need at least one fingerprint")
    return np.vstack([np.asarray(f.features, dtype=np.float64) for f in fingerprints])


def zscore(mat: np.ndarray) -> np.ndarray:
    """Column-wise z-score. Zero-variance columns become 0 (never NaN)."""
    f = np.asarray(mat, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {f.shape}")
    mean = f.mean(axis=0)
    std = f.std(axis=0)
    out = np.zeros_like(f)
    live = std > 1e-12
    if live.any():
        out[:, live] = (f[:, live] - mean[live]) / std[live]
    return out


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, shape (n, n). Zero-norm rows yield 0 (never NaN)."""
    f = np.asarray(mat, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {f.shape}")
    norms = np.linalg.norm(f, axis=1)
    unit = np.zeros_like(f)
    live = norms > 1e-12
    if live.any():
        unit[live] = f[live] / norms[live, None]
    return np.clip(unit @ unit.T, -1.0, 1.0)


def distance_to_centroid(mat: np.ndarray) -> np.ndarray:
    """Standardised distance of each client to the population centroid.

    z-score each feature column, then take the row norm. The centroid of standardised
    features is the origin, so this is ``||z_i||``: small = a typical hospital, large =
    an odd one out. This is what the aggregation weights are built on (cosine to the
    mean is undefined here, because the mean vector *is* the origin).
    """
    return np.linalg.norm(zscore(mat), axis=1)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def aggregation_weights(fingerprints, temperature: float = 1.0,
                        mode: str = "consensus") -> np.ndarray:
    """Turn client fingerprints into server aggregation weights (sum to 1).

    Weights are built from standardised distance-to-centroid (see
    ``distance_to_centroid``): softmax(-dist / T) for ``mode="consensus"``
    (up-weight typical clients, FedGTA-style) or softmax(+dist / T) for
    ``mode="distinctiveness"`` (up-weight the odd hospital out, to protect rare
    regimes — our project's failure mode).

    ``temperature=inf`` returns uniform 1/n — exactly plain FedAvg (sanity anchor).
    A single fingerprint returns weight 1.
    """
    if mode not in ("consensus", "distinctiveness"):
        raise ValueError(f"mode must be 'consensus' or 'distinctiveness', got {mode!r}")
    mat = fingerprints.features[None, :] if isinstance(fingerprints, TopologyFingerprint) \
        else fingerprint_matrix(fingerprints)
    n = mat.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if temperature is None or np.isinf(temperature):
        return np.full(n, 1.0 / n, dtype=np.float64)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 (or inf), got {temperature}")
    dist = distance_to_centroid(mat)
    logits = -dist / temperature if mode == "consensus" else dist / temperature
    return _softmax(logits)


# ----------------------------------------------------------------------------
def _self_check() -> None:
    """Printable smoke demo: one tiny fingerprint + example weights."""
    df = pd.DataFrame({
        PK:  ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
        ORG: ["oA", "oB", "oA", "oB", "oA", "oB", "oA", "oB"],
        ABX: ["a0", "a1", "a0", "a1", "a0", "a1", "a0", "a1"],
        "label": [1, 0, 1, 0, 1, 1, 0, 0],
        "order_time_jittered_utc": ["2020-01-01"] * 8,
    })
    fp = compute_fingerprint(df)
    print("fingerprint (one tiny hospital):")
    for k, v in fp.as_dict().items():
        print(f"  {k:>18}: {v:+.4f}")

    rng = np.random.default_rng(0)
    fps = [fp] + [TopologyFingerprint(rng.normal(size=len(FEATURE_NAMES))) for _ in range(3)]
    w_c = aggregation_weights(fps, temperature=0.5, mode="consensus")
    w_d = aggregation_weights(fps, temperature=0.5, mode="distinctiveness")
    w_inf = aggregation_weights(fps, temperature=np.inf)
    print("\nconsensus weights      :", np.round(w_c, 4))
    print("distinctiveness weights:", np.round(w_d, 4))
    print("temperature=inf        :", np.round(w_inf, 4), "(== uniform FedAvg)")
    assert np.allclose(w_c.sum(), 1.0) and np.allclose(w_inf, 0.25)
    print("OK: topology v0 self-check passed.")


if __name__ == "__main__":
    _self_check()
