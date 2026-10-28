"""Offline tests for SimFT weights and report helpers (numpy only):  python tests/test_simft.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from amr_fed.federated import analysis as A
from amr_fed.federated import simft_runs as SR
from amr_fed.federated.simft import simft_weights


def _L():
    # hospital 0's own model is best on its data; hospital 1's model is a good fit for hospital 0 (L=0.55),
    # hospital 2's model is bad for it (L=0.95). Rows are 'my data', columns are 'whose model'.
    return np.array([[0.50, 0.55, 0.95],
                     [0.60, 0.50, 0.58],
                     [0.90, 0.62, 0.50]])


def test_weights_properties() -> None:
    L = _L()
    for tau in (0.25, 0.5, 1.0, 4.0):
        W = simft_weights(L, tau=tau, self_floor=0.25)
        assert W.shape == (3, 3) and np.allclose(W.sum(axis=1), 1.0) and (W >= 0).all()
        assert (np.diag(W) >= 0.25 - 1e-12).all(), "self-floor must hold"
        assert W[0, 1] > W[0, 2], "the neighbour whose model fits my data better gets more weight"
        assert W[2, 1] > W[2, 0]
    W = simft_weights(L, tau=1.0, self_floor=0.25, flat=True)
    assert np.allclose(W[0, 1], W[0, 2]) and np.allclose(W[0, 0], 0.25 + 0.75 / 3), "flat ablation: equal neighbours"
    # large tau -> close to flat; small tau -> concentrates
    big, small = simft_weights(L, tau=1e3), simft_weights(L, tau=0.05)
    assert abs(big[0, 1] - big[0, 2]) < 1e-3 and abs(small[0, 1] - small[0, 2]) > abs(big[0, 1] - big[0, 2])
    # scale invariance: multiplying every loss by a constant leaves W unchanged
    assert np.allclose(simft_weights(L, tau=0.5), simft_weights(L * 10, tau=0.5))
    # NaN row (no validation rows) falls back to equal neighbours; others unaffected
    Ln = L.copy(); Ln[1] = np.nan
    Wn = simft_weights(Ln, tau=0.5)
    assert np.allclose(Wn.sum(axis=1), 1.0) and np.allclose(Wn[1, 0], Wn[1, 2]) and np.allclose(Wn[0], simft_weights(L, tau=0.5)[0])
    assert np.allclose(simft_weights(np.array([[0.4]])), 1.0)
    for bad in (dict(tau=0.0), dict(self_floor=1.5)):
        try:
            simft_weights(L, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("OK: SimFT weights are row-stochastic, keep the self-floor, favour better-fitting neighbours, scale-free, flat ablation works")


def _run(local, fa, ft, sf, sff, n=4):
    base = np.full(n, 0.0)
    mk = lambda v: list(v + base)
    nt = [{"test": 100 * (i + 1)} for i in range(n)]
    return {"val_local_per_client": mk(local), "phc_per_client": mk(fa), "finetune": {"per_client": mk(ft)},
            "n_triples": nt, "phc_val_score": 0.7, "simft_rounds": [{"W": np.eye(n).tolist(), "L": np.zeros((n, n)).tolist()}]}


def test_choose_and_report() -> None:
    rng = np.random.default_rng(1)
    mkarm = lambda f: {"seeds": [42, 43, 44, 45, 46], "runs": [f(i) for i in range(5)]}
    res = {"fedavg": mkarm(lambda i: _run(0.700 + 0.001 * i, 0.702, 0.708 + 0.001 * i, 0, 0)),
           "simft_t0.5": mkarm(lambda i: dict(_run(0, 0.705, 0.712 + rng.normal(0, .001), 0, 0), phc_val_score=0.71)),
           "simft_t1": mkarm(lambda i: dict(_run(0, 0.704, 0.711 + rng.normal(0, .001), 0, 0), phc_val_score=0.69)),
           "simft_flat": mkarm(lambda i: _run(0, 0.703, 0.709, 0, 0)), "_meta": {}}
    tau, scores = SR.choose_tau(res)
    assert tau == 0.5 and scores[0.5] > scores[1.0]
    res["_meta"]["chosen_tau"] = tau
    txt = A.simft_report(res)
    assert "PRE-REGISTERED VERDICT" in txt and "SimFT+fine-tune" in txt and "mean over hospitals" in txt
    v = A.simft_values(res, "simft")
    assert abs(v["mean"].mean() - 0.712) < 0.003
    assert "self-weight" in A.simft_weights_report(res)
    for setting in ("louvain", "org10", "ward10"):
        assert [k for k, _, _ in SR.arm_specs(setting)] == ["simft_t0.5", "simft_t1", "simft_flat"]
    assert dict((k, e) for k, _, e in SR.arm_specs("ward10"))["simft_flat"]["train_frac"] == 0.10
    assert "train_frac" not in dict((k, e) for k, _, e in SR.arm_specs("louvain"))["simft_flat"]
    try:
        SR.arm_specs("conflict")
        raise AssertionError("the conflict setting was dropped")
    except ValueError:
        pass
    print("OK: temperature chosen on validation, SimFT report/verdict run, arm lists as pre-registered")


if __name__ == "__main__":
    test_weights_properties()
    test_choose_and_report()
    print("OK: SimFT tests passed")
