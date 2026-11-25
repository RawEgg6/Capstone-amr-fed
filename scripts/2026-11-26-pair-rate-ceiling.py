"""Offline ceiling (no federation): how much does knowing each hospital's own pair rate add on top of patient history + the pooled pair rate? Gradient-boosted trees, out-of-fold encoding. Run from the repo root with ARMD_DIR set."""
import importlib.util, sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as G
spec = importlib.util.spec_from_file_location('pr', 'scripts/2026-11-24-pair-rate-split.py')
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
from amr_fed.data_loader import load_cohort_frame, PK, ORG, ABX
from amr_fed.graph_build import _history_raw
df = load_cohort_frame(); df2 = pr.make_df(df)[0]
lg = lambda p: np.log(np.clip(p, 1e-4, 1-1e-4) / (1 - np.clip(p, 1e-4, 1-1e-4)))
def run(d, label):
    h = d[PK].map(pr.hospital_of(d)).to_numpy(); y = d['label'].to_numpy().astype(float)
    test = (pd.util.hash_array(pd.Series(d[PK].to_numpy()).astype(str).add('|test').to_numpy()) % 10) < 3
    tr = ~test
    key = d.groupby([ORG, ABX]).ngroup().to_numpy(); kh = key * 10 + h
    gl = pd.Series(y[tr]).groupby(key[tr]).mean(); g = pd.Series(key).map(gl).fillna(y[tr].mean()).to_numpy()
    fold = (pd.util.hash_array(pd.Series(d[PK].to_numpy()).astype(str).add('|fold').to_numpy()) % 5).astype(int)
    tot = pd.DataFrame({'k': kh[tr], 'y': y[tr]}).groupby('k')['y'].agg(['sum', 'size'])
    S = pd.Series(kh).map(tot['sum']).fillna(0).to_numpy().copy(); N = pd.Series(kh).map(tot["size"]).fillna(0).to_numpy().copy()
    for f in range(5):                      # out-of-fold: a train row's own patient fold is removed from its encoding
        m_ = tr & (fold == f)
        sf = pd.DataFrame({'k': kh[m_], 'y': y[m_]}).groupby('k')['y'].agg(['sum', 'size'])
        S[m_] -= pd.Series(kh[m_]).map(sf['sum']).fillna(0).to_numpy(); N[m_] -= pd.Series(kh[m_]).map(sf['size']).fillna(0).to_numpy()
    own = (S + 20 * g) / (N + 20)
    H = _history_raw(d, float(y[tr].mean()))
    X1 = np.column_stack([lg(g), H]); X2 = np.column_stack([X1, lg(own)])
    sel = np.isin(key[test], np.where(np.bincount(key) >= pr.MIN_TESTS)[0])
    out = {}
    for name, X in (("history + pooled pair rate", X1), ("  + own-hospital pair rate", X2)):
        m = G(max_iter=150, learning_rate=0.1).fit(X[tr], y[tr]); p = m.predict_proba(X[test])[:, 1]
        out[name] = roc_auc_score(y[test][sel], p[sel])
        print(f"{label:<12}{name:<30}AUROC {out[name]:.4f}", flush=True)
    print(f"{label:<12}gain from knowing own-hospital pair rate: {out['  + own-hospital pair rate'] - out['history + pooled pair rate']:+.4f}", flush=True)
run(df, 'original'); run(df2, 'shifted')
