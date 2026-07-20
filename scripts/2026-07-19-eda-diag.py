import sys; sys.path.insert(0, "src")
from amr_fed import config
import pandas as pd, numpy as np
from pathlib import Path

D = Path(config.DATA_DIR)
CK, PK = config.KEYS["culture"], config.KEYS["patient"]
ORG, ABX, WP = config.COLUMNS["organism"], config.COLUMNS["antibiotic"], config.COLUMNS["was_positive"]

# 1) WARD flags on full data — why did the collapse give one client?
wcols = [config.WARD_FLAG_COLUMNS[w] for w in config.WARD_PRIORITY]
ward = pd.read_csv(D / config.ARMD_TABLES["ward"], usecols=[CK] + wcols)
print("ward rows:", len(ward))
for c in wcols:
    print(" ", c, ward[c].value_counts(dropna=False).to_dict())
flags = ward[wcols].apply(pd.to_numeric, errors="coerce").fillna(0)
print("cultures by #flags set:", flags.sum(axis=1).value_counts().sort_index().to_dict())
conds = [ward[config.WARD_FLAG_COLUMNS[w]] == 1 for w in config.WARD_PRIORITY]
ward["ward"] = np.select(conds, config.WARD_PRIORITY, default="NONE")
print("collapsed ward sizes:", ward["ward"].value_counts(dropna=False).to_dict())

# 2) Null / negative-culture impact
coh = pd.read_csv(D / config.ARMD_TABLES["cohort"], usecols=[PK, CK, ORG, ABX, WP, config.LABEL_COLUMN])
print("\ncohort rows:", len(coh))
print("was_positive:", coh[WP].value_counts(dropna=False).to_dict())
print("organism==Null:", int((coh[ORG].astype(str).str.lower() == "null").sum()))
print("antibiotic==Null:", int((coh[ABX].astype(str).str.lower() == "null").sum()))
pos = coh[coh[WP] == 1]
print("after was_positive==1 -> rows:", len(pos), "| organisms:", pos[ORG].nunique(), "| antibiotics:", pos[ABX].nunique())
print("label balance (positive only):", pos[config.LABEL_COLUMN].value_counts(dropna=False).to_dict())

# 3) patient sparsity on full data
deg = coh.groupby(PK)[CK].nunique()
print("\npatients:", deg.size, "| mean:", round(deg.mean(),2), "| median:", int(deg.median()),
      "| pct with exactly 1 culture:", round((deg==1).mean(),3), "| max:", int(deg.max()))
