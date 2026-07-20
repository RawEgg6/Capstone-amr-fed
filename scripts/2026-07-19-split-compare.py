import sys; sys.path.insert(0, "src")
from amr_fed import config
import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)

D = Path(config.DATA_DIR)
CK, PK = config.KEYS["culture"], config.KEYS["patient"]
ORG, ABX, WP = config.COLUMNS["organism"], config.COLUMNS["antibiotic"], config.COLUMNS["was_positive"]
CD, LAB = config.COLUMNS["culture_description"], config.LABEL_COLUMN
MIN = 5000

cohort_cols = pd.read_csv(D / config.ARMD_TABLES["cohort"], nrows=0).columns
use = [PK, CK, ORG, ABX, WP, LAB, CD] + (["ordering_mode"] if "ordering_mode" in cohort_cols else [])
coh = pd.read_csv(D / config.ARMD_TABLES["cohort"], usecols=use)
work = coh[coh[LAB].isin(["Susceptible", "Resistant", "Intermediate"])].copy()
work["is_R"] = work[LAB].isin(["Resistant", "Intermediate"]).astype(int)
print("outcome rows:", len(work), "| overall R rate:", round(work["is_R"].mean(), 3))

splits = {}
wc = [config.WARD_FLAG_COLUMNS[w] for w in config.WARD_PRIORITY]
ward = pd.read_csv(D / config.ARMD_TABLES["ward"], usecols=[CK] + wc)
conds = [ward[config.WARD_FLAG_COLUMNS[w]] == 1 for w in config.WARD_PRIORITY]
ward["c_ward"] = np.select(conds, config.WARD_PRIORITY, default="NONE")
work = work.merge(ward[[CK, "c_ward"]], on=CK, how="left"); splits["ward"] = "c_ward"

top = work[CD].value_counts().head(3).index
work["c_spec"] = np.where(work[CD].isin(top), work[CD], "Other"); splits["specimen"] = "c_spec"

if "ordering_mode" in work.columns:
    work["c_om"] = work["ordering_mode"].fillna("Unknown"); splits["ordering_mode"] = "c_om"

demo = pd.read_csv(D / config.ARMD_TABLES["demographics"], usecols=[PK, config.COLUMNS["age"]]).drop_duplicates(PK)
work = work.merge(demo, on=PK, how="left")
work["c_age"] = work[config.COLUMNS["age"]].fillna("Unknown"); splits["age"] = "c_age"

acol = [c for c in pd.read_csv(D / config.ARMD_TABLES["adi"], nrows=0).columns if "adi_score" in c.lower()][:1]
if acol:
    adi = pd.read_csv(D / config.ARMD_TABLES["adi"], usecols=[PK] + acol, low_memory=False).drop_duplicates(PK)
    adi[acol[0]] = pd.to_numeric(adi[acol[0]], errors="coerce")
    work = work.merge(adi, on=PK, how="left")
    work["c_adi"] = pd.qcut(work[acol[0]], 4, duplicates="drop").astype("object").fillna("Unknown")
    splits["adi"] = "c_adi"

def score(col):
    g = work.groupby(col)["is_R"].agg(n="size", rate="mean"); g = g[g.index.astype(str) != "NONE"]
    w = g["n"] / g["n"].sum(); wm = (w * g["rate"]).sum()
    wstd = float(np.sqrt((w * (g["rate"] - wm) ** 2).sum()))
    overlap = float((work.groupby(PK)[col].nunique() > 1).mean())
    return {"clients": len(g), "usable>=5k": int((g["n"] >= MIN).sum()), "min": int(g["n"].min()),
            "max": int(g["n"].max()), "ratio": round(g["n"].max()/max(g["n"].min(),1), 1),
            "R_spread": round(g["rate"].max()-g["rate"].min(), 3), "non_iid_wstd": round(wstd, 4),
            "patients_in_>1_client": round(overlap, 3)}

print(pd.DataFrame({k: score(v) for k, v in splits.items()}).T.sort_values("non_iid_wstd", ascending=False))
