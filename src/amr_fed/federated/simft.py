"""SimFT: personalized aggregation weights from how well each hospital's model works on MY data
(function-space similarity), torch-free so it can be unit-tested.

Why function space: parameter-space cosine similarity is a weak client-similarity signal (curse of
dimensionality; parameter similarity rises as training continues, so thresholds depend on training
length; models with small parameter gaps can behave very differently). FedFomo (Zhang et al., ICLR 2021,
arXiv 2012.08565) instead scores each received model on the receiving client's own validation data.
We do the same: ``L[i, j]`` = loss of hospital j's locally trained model on hospital i's validation
rows (half A of its validation patients; half B is kept for selecting round / cutoff / epoch).

Weights for hospital i:  d_ij = L[i, j] - L[i, i]  (negative when j's model is better for i than its own),
scale-free  z_ij = d_ij / (tau * s_i),  s_i = mean_{j != i} |d_ij|,
W[i] = self_floor * e_i + (1 - self_floor) * softmax_j(-z_ij).
A hospital therefore always keeps at least ``self_floor`` of its own model (similarity-driven weights can
otherwise collapse into self-clusters), and the diagonal of the softmax is the reference point.
``flat=True`` is the ablation: the same self-floor structure with the neighbours weighted equally, so any
gain over it is attributable to the similarity signal itself.
"""
from __future__ import annotations

import numpy as np


def simft_weights(L, tau: float = 1.0, self_floor: float = 0.25, flat: bool = False) -> np.ndarray:
    """Row-stochastic (n, n) mixing matrix from the cross-evaluation loss matrix ``L`` (see module doc)."""
    L = np.asarray(L, dtype=np.float64)
    n = L.shape[0]
    if L.shape != (n, n):
        raise ValueError(f"L must be square, got {L.shape}")
    if n == 1:
        return np.ones((1, 1))
    if not 0.0 <= self_floor <= 1.0:
        raise ValueError(f"self_floor must be in [0, 1], got {self_floor}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    eye = np.eye(n)
    uniform = np.full((n, n), 1.0 / n)
    W = np.empty((n, n))
    for i in range(n):
        row = L[i]
        if flat or not np.isfinite(row).all():          # no usable scores for this hospital: neighbours equal
            soft = uniform[i]
        else:
            d = row - row[i]
            others = np.abs(np.delete(d, i))
            s = others.mean() if others.size and others.mean() > 1e-12 else 1.0
            z = -d / (tau * s)
            z = z - z.max()
            e = np.exp(z)
            soft = e / e.sum()
        W[i] = self_floor * eye[i] + (1.0 - self_floor) * soft
    return W
