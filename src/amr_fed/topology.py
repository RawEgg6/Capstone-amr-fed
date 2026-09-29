"""TKPA-FL topology utilities.

Topology-Kernel Personalized Aggregation for Federated Learning.

Each hospital is described by its training-triple distribution:
    m_i(c)  =  count of training triples with cell c = (organism, antibiotic, label)

From these profiles the server computes a pairwise Jensen-Shannon similarity
matrix S and an aggregation matrix A so that each hospital receives a
personalized model:
    V_i = Σ_j A_ij * U_j

No patient-level data leaves the client. Only the (org, abx, label) -> count
mapping is serialised and transmitted.

All computations are pure NumPy / stdlib. No torch dependency.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# --------------------------------------------------------------------------- #
# 1. Hospital profile construction
# --------------------------------------------------------------------------- #

def build_hospital_profile(data) -> dict:
    """Build a sparse (organism_name, antibiotic_name, label) -> count mapping.

    Only training triples (train_mask) are used. No val/test labels are touched.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        Client graph produced by task.build_and_save_clients().  Must carry
        data.node_names (dict with "organism" and "antibiotic" numpy str
        arrays), data.triple_index [3, N], data.triple_label [N], and
        data.train_mask [N] boolean tensor.

    Returns
    -------
    dict mapping (org_name, abx_name, label_int) -> count
    """
    org_names = data.node_names["organism"]    # np.ndarray of str, shape [n_org]
    abx_names = data.node_names["antibiotic"]  # np.ndarray of str, shape [n_abx]

    tri  = data.triple_index                   # [3, N]: patient, org, abx
    y    = data.triple_label                   # [N]
    mask = data.train_mask                     # [N] bool

    # Select only train triples
    org_idx = tri[1, mask].cpu().numpy()   # organism node ids (local)
    abx_idx = tri[2, mask].cpu().numpy()   # antibiotic node ids (local)
    labels  = y[mask].cpu().long().numpy() # 0 or 1

    profile = defaultdict(int)
    for o, a, lbl in zip(org_idx, abx_idx, labels):
        key = (str(org_names[o]), str(abx_names[a]), int(lbl))
        profile[key] += 1

    return dict(profile)


# --------------------------------------------------------------------------- #
# 2. Serialization
# --------------------------------------------------------------------------- #

def serialize_profile(profile: dict) -> bytes:
    """Serialize profile to bytes (JSON-encoded, zlib-compressed).

    The profile is stored as a list of [org_name, abx_name, label, count]
    records, which is compact and trivially reconstructable.
    """
    records = [
        [org, abx, lbl, cnt]
        for (org, abx, lbl), cnt in profile.items()
    ]
    raw = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return zlib.compress(raw)


def deserialize_profile(data_bytes: bytes) -> dict:
    """Deserialize bytes back to (org, abx, label) -> count."""
    raw = zlib.decompress(data_bytes)
    records = json.loads(raw.decode("utf-8"))
    return {(str(r[0]), str(r[1]), int(r[2])): int(r[3]) for r in records}


# --------------------------------------------------------------------------- #
# 3. Similarity matrix computation
# --------------------------------------------------------------------------- #

def build_similarity(profiles: list, kappa: float = 100.0) -> np.ndarray:
    """Compute pairwise JSD-based similarity matrix S between hospitals.

    Algorithm
    ---------
    1. Union all cells c across all hospitals.
    2. Compute raw counts m_i(c) for every hospital i and cell c.
    3. Compute pooled distribution Qbar(c) = sum_i m_i(c) / sum_i n_i.
    4. Shrink each hospital toward the pooled distribution with kappa:
           Qt_i(c) = (m_i(c) + kappa * Qbar(c)) / (n_i + kappa)
       If n_i == 0, Qt_i = Qbar.
    5. Compute pairwise JSD (base-2) between Qt_i and Qt_j.
    6. Convert to similarity:
           tau  = median(d_ij for i < j)
           S_ij = exp(-min(d_ij / tau, 3))  for i != j
           S_ii = 1
       If tau == 0 -> S = ones(H, H).

    Parameters
    ----------
    profiles : list of hospital profiles (one per hospital, in partition order).
    kappa    : Empirical-Bayes shrinkage weight (default 100).

    Returns
    -------
    S : np.ndarray of shape [H, H], float64, values in (0, 1].
    """
    H = len(profiles)

    # --- 3a. Union of all cells ---
    all_cells = sorted({cell for p in profiles for cell in p})
    C = len(all_cells)
    cell_idx = {c: i for i, c in enumerate(all_cells)}

    # --- 3b. Raw count matrix M[H, C] ---
    M = np.zeros((H, C), dtype=np.float64)
    for i, prof in enumerate(profiles):
        for cell, cnt in prof.items():
            M[i, cell_idx[cell]] = cnt

    n_i = M.sum(axis=1)  # [H] total training triples per hospital

    # --- 3c. Pooled distribution Qbar [C] ---
    total = n_i.sum()
    if total == 0:
        return np.ones((H, H), dtype=np.float64)
    Qbar = M.sum(axis=0) / total  # [C]

    # --- 3d. Shrunk distributions Qt [H, C] ---
    Qt = np.zeros((H, C), dtype=np.float64)
    for i in range(H):
        denom = n_i[i] + kappa
        Qt[i] = (M[i] + kappa * Qbar) / denom

    # --- 3e. Pairwise JSD ---
    D = np.zeros((H, H), dtype=np.float64)
    for i in range(H):
        for j in range(i + 1, H):
            d = _jsd(Qt[i], Qt[j])
            D[i, j] = d
            D[j, i] = d

    # --- 3f. Distance -> similarity ---
    upper = D[np.triu_indices(H, k=1)]
    tau = float(np.median(upper)) if len(upper) > 0 else 0.0

    if tau == 0.0:
        return np.ones((H, H), dtype=np.float64)

    S = np.exp(-np.minimum(D / tau, 3.0))
    np.fill_diagonal(S, 1.0)
    return S


def _kl_safe(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in bits (base-2). Handles 0*log(0)=0."""
    mask = p > 0
    p_m = p[mask]
    q_m = q[mask]
    # Clip q to avoid log(0) where q=0 but p>0
    q_m = np.where(q_m > 0, q_m, np.finfo(float).tiny)
    return float(np.sum(p_m * np.log2(p_m / q_m)))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence JSD(p || q) in bits, base-2. Range [0, 1]."""
    M = 0.5 * (p + q)
    return 0.5 * _kl_safe(p, M) + 0.5 * _kl_safe(q, M)


# --------------------------------------------------------------------------- #
# 4. Aggregation matrix
# --------------------------------------------------------------------------- #

def build_aggregation_matrix(S: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Compute row-normalised aggregation matrix A from similarity S and sizes n.

    A_ij = n_j * S_ij / sum_k (n_k * S_ik)

    Every row sums to 1.  n is the training-triple count per hospital.

    Parameters
    ----------
    S : [H, H] similarity matrix (S_ii = 1, S_ij in (0,1]).
    n : [H] array of training triple counts.

    Returns
    -------
    A : [H, H] row-stochastic aggregation matrix.
    """
    W = S * n[np.newaxis, :]          # W_ij = S_ij * n_j, broadcast n as column
    row_sums = W.sum(axis=1, keepdims=True)
    # Guard against all-zero row (should not happen with S_ii=1, n_i>0)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    A = W / row_sums
    return A
