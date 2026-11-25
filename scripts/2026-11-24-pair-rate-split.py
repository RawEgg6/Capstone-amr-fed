"""Split D: hospitals whose resistance RATE differs, by a realistic amount, on every well-tested organism-antibiotic pair.

10 equal hospitals (patients assigned by a hash of the patient id, so no natural structure). Every pair with at least
MIN_TESTS tests gets, per hospital, a target resistant rate = the pair's overall rate moved on the logit scale by
delta[h, organism, drug class] + a small per-drug term. Drugs of the same class share their organism's shift (a hospital
that is high on ciprofloxacin is high on levofloxacin too), as in real data; different classes and different organisms
move independently, so a hospital is high on some things and low on others. Hospitals sit in N_REGIONS regions: the shift is a
regional profile plus a smaller hospital term, so hospitals of one region resemble each other (as in the regional sources
below). Without this every pair of hospitals is about equally different and a similarity method has nothing to find. For each (organism, class) block the
most-vs-least-resistant hospital odds ratio is set inside the range seen in real data:
  Gaza, two hospitals (El Aila & Abu El Aish, BMC Microbiol 2025;25:559, Table 4, E. coli): ceftriaxone 41.2% vs 64.7%
    (OR 2.6), ciprofloxacin 27.5% vs 57.0% (OR 3.5), amikacin 4.6% vs 15.1% (OR 3.7).
  US VA census divisions (Livorsi et al., Antimicrob Resist Infect Control 2016;5:38, E. coli bloodstream, FQ-NS):
    20.0% to 32.7% (OR 1.9).
  Washington State, 9 regions (Fenelon et al., Microorganisms 2024;12:2313, urinary E. coli): adjusted odds ratios vs the
    reference region 0.20-0.78 (ceftriaxone), 0.64-0.81 (ciprofloxacin).
  15 US emergency departments (Acad Emerg Med 2022, E. coli UTI, FQ-R): 10.5% to 29.7% (OR 3.6). Search snippet only, not opened.
Labels are the minimum number of random flips needed to reach each target (susceptible -> resistant when raising,
resistant -> susceptible when lowering). Everything else is untouched.
Run: ARMD_DIR=... python scripts/2026-11-24-pair-rate-split.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

N_HOSP, MIN_TESTS, OR_LOW, OR_HIGH, DRUG_NOISE, SEED = 10, 200, 2.0, 3.6, 0.25, 0
N_REGIONS, WITHIN = 3, 0.35   # hospital c is in region c % 3; within-region spread = 0.35 x the between-region spread
REGION = np.arange(N_HOSP) % N_REGIONS
REAL = {"Gaza ceftriaxone": 2.6, "Gaza ciprofloxacin": 3.5, "VA E. coli FQ": 1.9, "US EDs E. coli FQ": 3.6}

CLASS = {}
for cls, drugs in {
    "fluoroquinolone": ["Ciprofloxacin", "Levofloxacin", "Moxifloxacin"],
    "aminoglycoside": ["Gentamicin", "Amikacin", "Tobramycin", "Streptomycin"],
    "folate pathway": ["Trimethoprim/Sulfamethoxazole"],
    "aminopenicillin": ["Ampicillin", "Amoxicillin/Clavulanic Acid", "Ampicillin/Sulbactam"],
    "anti-staph penicillin": ["Penicillin", "Oxacillin"],
    "antipseudomonal penicillin": ["Piperacillin/Tazobactam", "Piperacillin", "Ticarcillin", "Ticacarcillin/Clavulanic Acid"],
    "early cephalosporin": ["Cefazolin", "Cefoxitin", "Cefuroxime", "Cefotetan", "Cephalexin/Cephalothin", "Cefpodoxime"],
    "late cephalosporin": ["Ceftriaxone", "Ceftazidime", "Cefepime", "Cefotaxime", "Ceftaroline", "Ceftolozane/Tazobactam",
                           "Ceftazidime/Avibactam", "Cefiderocol"],
    "carbapenem": ["Meropenem", "Ertapenem", "Imipenem", "Doripenem", "Imipenem/Ebactam", "Meropenem/Vaborbactam"],
    "tetracycline": ["Tetracycline", "Doxycycline", "Minocycline", "Tigecycline"],
    "macrolide/lincosamide": ["Erythromycin", "Clarithromycin", "Clindamycin"],
}.items():
    CLASS.update({d: cls for d in drugs})


def drug_class(abx: str) -> str:
    return CLASS.get(abx, abx)   # monobactam, glycopeptide, nitrofurantoin, ... are their own class


def hospital_of(df) -> pd.Series:
    from amr_fed.data_loader import PK
    pk = pd.Series(sorted(df[PK].unique()))
    return pd.Series((pd.util.hash_pandas_object(pk, index=False).to_numpy() % N_HOSP).astype(int),
                     index=pk.to_numpy(), name="hospital")


def partition_fn(df, seed):
    return hospital_of(df)   # fixed by patient id: the data differs by hospital, not the assignment


def make_df(df, seed=SEED, mode="flip"):
    """Copy of df with each hospital's rate on every well-tested pair moved to its target.
    mode="flip": the minimum number of random label flips. mode="resample": keep every label real and DROP rows instead (to raise a
    rate keep only a share of the susceptible rows, to lower it keep only a share of the resistant rows), so the rate hits the target
    without any flipped label. Returns (df, pairs, changed) where changed = flipped (flip) or dropped (resample) rows and pairs is a
    DataFrame (organism, drug, class, tests, base rate)."""
    from amr_fed.data_loader import ABX, ORG, PK
    rng = np.random.default_rng(seed)
    out = df.copy()
    pair = out.groupby([ORG, ABX]).ngroup().to_numpy()
    n_all = pair.max() + 1
    y = out["label"].to_numpy().astype(float)
    tests = np.bincount(pair)
    base = np.bincount(pair, weights=y, minlength=n_all) / tests
    keys = out.groupby([ORG, ABX]).size().index
    org = np.array([k[0] for k in keys]); abx = np.array([k[1] for k in keys])
    cls = np.array([drug_class(a) for a in abx])
    keep = tests >= MIN_TESTS
    # class-level shift per (organism, class), range = ln(OR), then a small per-drug term
    block = pd.Series(org + "|" + cls)
    delta = np.zeros((n_all, N_HOSP))
    for b in block[keep].unique():
        idx = np.where((block == b).to_numpy() & keep)[0]
        d = rng.standard_normal(N_REGIONS)[REGION] + WITHIN * rng.standard_normal(N_HOSP)   # regional profile + own term
        d = (d - d.mean()) / (d.max() - d.min()) * np.log(rng.uniform(OR_LOW, OR_HIGH))
        noise = rng.standard_normal((len(idx), N_HOSP)) * DRUG_NOISE * np.log(OR_HIGH) / 3
        delta[idx] = d + noise - noise.mean(axis=1, keepdims=True)
    b = np.clip(base, 0.005, 0.995)
    target = 1 / (1 + np.exp(-(np.log(b / (1 - b))[:, None] + delta)))
    hosp = out[PK].map(hospital_of(out)).to_numpy()
    cell = pair * N_HOSP + hosp
    r = np.bincount(cell, weights=y, minlength=n_all * N_HOSP) / np.maximum(np.bincount(cell, minlength=n_all * N_HOSP), 1)
    t_row, r_row = target.reshape(-1)[cell], r[cell]
    u = rng.random(len(out))
    if mode == "resample":
        r_c, t_c = np.clip(r_row, 1e-6, 1 - 1e-6), np.clip(t_row, 1e-6, 1 - 1e-6)
        keep_sus = np.where(t_c > r_c, r_c * (1 - t_c) / (t_c * (1 - r_c)), 1.0)    # share of susceptible rows kept (raise)
        keep_res = np.where(t_c < r_c, t_c * (1 - r_c) / (r_c * (1 - t_c)), 1.0)    # share of resistant rows kept (lower)
        drop = keep[pair] & (((y == 0) & (u >= keep_sus)) | ((y == 1) & (u >= keep_res)))
        out = out[~drop].copy()
        pairs = pd.DataFrame({"organism": org, "drug": abx, "class": cls, "tests": tests, "base": base})[keep].reset_index()
        return out, pairs, int(drop.sum())
    up = keep[pair] & (t_row > r_row) & (y == 0) & (u < (t_row - r_row) / np.maximum(1 - r_row, 1e-9))
    down = keep[pair] & (t_row < r_row) & (y == 1) & (u < (r_row - t_row) / np.maximum(r_row, 1e-9))
    y_new = y.copy()
    y_new[up] = 1; y_new[down] = 0
    out["label"] = y_new.astype("int8")
    pairs = pd.DataFrame({"organism": org, "drug": abx, "class": cls, "tests": tests, "base": base})[keep].reset_index()
    return out, pairs, int(up.sum() + down.sum())


def rate_grid(df, pairs):
    """Per-hospital resistant rate for every kept pair: DataFrame (pair x hospital)."""
    from amr_fed.data_loader import ABX, ORG, PK
    hosp = df[PK].map(hospital_of(df))
    g = df.groupby([ORG, ABX, hosp])["label"].agg(["mean", "size"]).unstack()
    idx = pd.MultiIndex.from_frame(pairs[["organism", "drug"]])
    return g["mean"].reindex(idx), g["size"].reindex(idx)


def describe(df, df2, pairs, flipped):
    """Print how many labels changed, the odds ratios against the real references, and the class check."""
    print(f"{len(df):,} tests | pairs with >= {MIN_TESTS} tests: {len(pairs)} "
          f"({100 * pairs['tests'].sum() / len(df):.1f}% of tests) | labels changed: {flipped:,} ({100 * flipped / len(df):.2f}%)\n")
    rate, size = rate_grid(df2, pairs)
    ok = rate.where(size >= 20)           # ignore hospital cells under 20 tests
    lo, hi = ok.min(axis=1).to_numpy(), ok.max(axis=1).to_numpy()
    odds = (hi / np.clip(1 - hi, 1e-9, None)) / (lo / np.clip(1 - lo, 1e-9, None))
    big = (pairs["tests"].to_numpy() >= 1000) & (lo > 0.02) & (hi < 0.98)
    q = np.round(np.quantile(odds[big], [0.1, 0.25, 0.5, 0.75, 0.9]), 2).tolist()
    print(f"most-vs-least-resistant hospital odds ratio, pairs with >= 1000 tests and rates 2-98% (n={int(big.sum())}): "
          f"p10/25/50/75/90 = {q}\nreal: {REAL}\n")
    ex = pairs.sort_values("tests", ascending=False).head(12)
    t = pd.DataFrame({"pair": [f"{o.title()[:22]} / {a[:20]}" for o, a in zip(ex.organism, ex.drug)],
                      "tests": ex.tests.to_numpy(), "overall": [f"{100 * v:.0f}%" for v in ex.base],
                      "hospital range": [f"{100 * lo[i]:.0f}% - {100 * hi[i]:.0f}%" for i in ex.index],
                      "odds ratio": odds[ex.index].round(2)})
    print(t.to_string(index=False))
    ec = pairs[(pairs.organism == "ESCHERICHIA COLI") & (pairs["class"] == "fluoroquinolone")]
    print("\nclass check, E. coli fluoroquinolones: rate per hospital 0-9 (%) and correlation across hospitals")
    g = (100 * rate.loc[[(o, d) for o, d in zip(ec.organism, ec.drug)]]).round(0).astype(int)
    g.index = [d for _, d in g.index]
    print(g.to_string())
    print("correlation:", np.corrcoef(g.to_numpy()).round(2).tolist())


def structure_check(df2, pairs, seed=42):
    """Are some hospitals really more alike than others? Mean distance between hospitals of the SAME region against
    hospitals of DIFFERENT regions (ratio < 1 = same-region hospitals are closer), for the true rate deviations and the two
    fingerprints. Nothing is trained."""
    from amr_fed.data_loader import PK
    from amr_fed.label_fingerprints import cross_antibiogram_loss, topology_v3_distance
    from amr_fed.topology import train_frame
    h = df2[PK].map(hospital_of(df2))
    frames = [train_frame(df2[h == c], seed) for c in range(N_HOSP)]
    rate, _ = rate_grid(df2, pairs)
    R = rate.to_numpy()[pairs["tests"].to_numpy() >= 1000]
    R = np.nan_to_num(R - np.nanmean(R, axis=1, keepdims=True))
    D_true = np.sqrt(((R.T[:, None, :] - R.T[None, :, :]) ** 2).mean(-1))
    same = (REGION[:, None] == REGION[None, :]) & ~np.eye(N_HOSP, dtype=bool)
    diff = REGION[:, None] != REGION[None, :]
    rows = []
    for name, D in (("true rate deviations", D_true), ("label fingerprint (cross-antibiogram)", cross_antibiogram_loss(frames)),
                    ("topology fingerprint (graph shape)", topology_v3_distance(frames))):
        D = np.asarray(D, float)
        rows.append((name, round(D[same].mean(), 3), round(D[diff].mean(), 3), round(D[same].mean() / D[diff].mean(), 2)))
    print(pd.DataFrame(rows, columns=["distance between hospitals", "same region", "different region", "ratio (<1 = sees regions)"]).to_string(index=False))


def main():
    from amr_fed.data_loader import load_cohort_frame
    df = load_cohort_frame()
    df2, pairs, flipped = make_df(df)
    describe(df, df2, pairs, flipped)
    structure_check(df2, pairs)


if __name__ == "__main__":
    main()
