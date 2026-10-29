"""Offline test: does a hospital fingerprint predict cross-hospital transfer? (no training, CPU, ~2 min)

Ground truth = SimFT's cross-evaluation losses (results/amr_simft_<setting>_all.json, notebook 19):
G[i, j] = loss of hospital j's model on hospital i's validation rows minus i's own model's loss.
Each candidate fingerprint (amr_fed.label_fingerprints.CANDIDATES) is computed from TRAIN rows only and
scored by Spearman against G. Run:  ARMD_DIR=... python scripts/2026-10-29-fingerprint-offline-test.py
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from amr_fed.data_loader import PK, load_cohort_frame
from amr_fed.federated.regimes import subsample_frame
from amr_fed.label_fingerprints import CANDIDATES, _spearman, score_against, transfer_gap
from amr_fed.partition import dirichlet_ward_mixture, louvain_split, organism_community
from amr_fed.topology import train_frame

SETTINGS = {
    "org10":   (lambda df, s: organism_community(df, n_clients=10, seed=s), 1.0),
    "louvain": (lambda df, s: louvain_split(df, n_clients=3, seed=s), 1.0),
    "ward10":  (lambda df, s: dirichlet_ward_mixture(df, n_clients=10, alpha=0.5, seed=s), 0.10),
}

cache = sys.argv[1] if len(sys.argv) > 1 else None
df = pd.read_pickle(cache) if cache else load_cohort_frame()
rows, ceil = [], {}
for setting, (split, frac) in SETTINGS.items():
    res = json.load(open(ROOT / "results" / f"amr_simft_{setting}_all.json"))
    arm = res["simft_t0.5"]
    Gs = {}
    for seed, run in zip(arm["seeds"], arm["runs"]):
        raw = split(df, seed)
        assign = pd.Series(pd.factorize(raw)[0], index=raw.index)   # same relabelling as run.run_fedavg
        n = int(assign.max()) + 1
        frames = []
        for c in range(n):
            sub = df[df[PK].map(assign) == c]
            if frac < 1.0:
                sub, _ = subsample_frame(sub, seed=seed, cid=c, frac=frac)
            frames.append(train_frame(sub, seed))
        sizes = [int((assign == c).sum()) for c in range(n)]
        if frac == 1.0 and sizes != run["sizes"]:
            print(f"  WARNING {setting} seed {seed}: rebuilt sizes {sizes} != saved {run['sizes']}")
        G1 = transfer_gap(run["simft_rounds"][0]["L"])
        Gm = np.mean([transfer_gap(r["L"]) for r in run["simft_rounds"]], axis=0)
        Gs[seed] = G1
        for name, fn in CANDIDATES.items():
            D = fn(frames)
            for tgt, G in (("round 1", G1), ("all rounds", Gm)):
                s = score_against(D, G, n_perm=500, seed=seed)
                rows.append(dict(setting=setting, seed=seed, target=tgt, fingerprint=name, **s))
        print(f"{setting} seed {seed} done", flush=True)
    # noise ceiling: how well does one seed's G predict another seed's G (only meaningful for a fixed partition)
    if setting == "org10":
        ks = list(Gs)
        off = ~np.eye(len(Gs[ks[0]]), dtype=bool)
        ceil[setting] = np.mean([_spearman(Gs[a][off], Gs[b][off]) for i, a in enumerate(ks) for b in ks[i + 1:]])

out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
summ = (out.groupby(["setting", "target", "fingerprint"])
           .agg(pooled=("pooled", "mean"), pooled_sd=("pooled", "std"), row=("row", "mean"),
                seeds_p05=("p_perm", lambda p: int((p < 0.05).sum())))
           .round(3))
print(summ.to_string())
print("\nnoise ceiling (org10, same partition, seed-to-seed Spearman of round-1 G):", {k: round(v, 3) for k, v in ceil.items()})
out.to_csv(ROOT / "results" / "fingerprint_offline_test.csv", index=False)
