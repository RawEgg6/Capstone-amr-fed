# Pre-registration — antibiogram-conflict split (Experiment #1: replicate a documented FedAvg failure condition)

**Recorded 2026-10-18, BEFORE any training run on this split.** Purpose: prevent
"search splits until FedAvg loses" cherry-picking. The split design, hypothesis, and
pass/fail criteria below are fixed now; results will be reported against them,
whatever they are. Any subsequent iteration must be appended here with its date.

## 1. Motivation — literature-motivated condition, not a random hunt

Literature search (2026-10-18) identified when plain averaging demonstrably fails:

| Condition | Source | Our status |
|---|---|---|
| **Concept shift** — clients disagree in P(y\|x) | *On the Unreasonable Effectiveness of FedAvg* (`arXiv:2206.04723`): FedAvg is fine when clients share the labeling rule (explains our wins) and breaks when the rule differs | **Never reproduced → this experiment** |
| **Mixed label × quantity skew** | FNR-FL, IJCAI 2024 (`ijcai.org/proceedings/2024/0457.pdf`) | Partially (louvain) |
| Client drift × many local epochs | Li et al., ICLR 2020 (`arXiv:1907.02189`) | Experiment #2 (separate) |
| Many clients | `arXiv:2504.08198` | Experiment #3 (separate) |
| ~~Marginal label skew~~ | our own label-Dirichlet dead end (β=0.5 no-op; β=0.1 degenerate) | Proven not to bite our feature-driven model |
| ~~Pure quantity skew~~ | `arXiv:2503.17070` — no effect alone | Do not pursue alone |

**Clinical framing:** hospitals genuinely differ in local antibiograms for the *same*
bug–drug pair — the reason CLSI-style local antibiograms and institution-specific
empirical guidelines exist. The split simulates that divergence, combined with
imbalance (one big community-style hospital vs a small high-resistance tail).

## 2. The split (`antibiogram_conflict`, `src/amr_fed/partition.py`)

1. Score each patient by mean label over tests on **shared pairs** — pairs tested in
   ≥ `min_pair_patients` (default 25) distinct patients. Shared pairs are the
   battleground every hospital holds; disagreement there is what makes averaging
   conflict. (Fallback: overall rate if no pair qualifies — warns.)
2. Rank patients ascending by that score (tiebreak: overall rate, then patient id).
3. Cut contiguous blocks with **shares** — default geometric taper
   (`0.55^i` normalized): big susceptible-leaning majority → small resistant-heavy
   minority tail. Shares are **floored at 3%** — no degenerate 1-patient hospitals
   (the flaw that killed label-Dirichlet β=0.1).
4. Hospital 0 = most susceptible on shared pairs; hospital k−1 = most resistant.

**Why this differs from label-Dirichlet:** (a) scoring conditioned on the shared pairs
themselves, guaranteeing divergence on exactly the pairs all hospitals hold; (b)
contiguous rank-extremes instead of Dirichlet draws; (c) explicit size guard; (d)
sizes are deliberately imbalanced (mixed label × quantity skew).

Mechanism being tested: each hospital's local model is self-consistent, but the
learned function for the same bug–drug pair differs across hospitals; size-weighted
FedAvg follows the big hospital, and the small resistant tail's ranking collapses
(worst-hospital macro-F1 drops). The topology fingerprint sees the tail as odd
(extreme `history_prior_rate`, small size, distinct homophily) → distinctiveness
weighting up-weights it → worst-hospital rescue.

## 3. Hypothesis (directional, written before results)

- **H1 (condition reproduces):** FedAvg's worst-hospital macro-F1 falls **below
  local-only's worst-hospital** by ≥ 0.01 — i.e. averaging actively *hurts* the
  minority hospital (the "Path A" shape from `docs/2026-08-11-splits-and-results.md`).
- **H2 (our method wins):** topology-aware (distinctiveness, T=1.0) beats FedAvg by
  **≥ 0.005 on worst-hospital OR mean macro-F1** (the pre-registered Path-B gate),
  at least partly by *raising* the tail hospital rather than sinking the others.

## 4. Protocol (fixed)

- **Split:** `antibiogram_conflict(df, n_clients=5)` (default shares, min 25).
- **Validities checked before training:** 1:1 patient assignment; every hospital ≥ 3%
  of tests; resistance-rate spread (max−min across hospitals) ≥ 0.15; shared-pair
  rates diverge (print top-pair per-hospital rates in the smoke cell).
- **Training:** notebook-07 v2 protocol for comparability — `rounds=8`,
  `local_epochs=4`, `local_only_epochs=32`, `hidden=128`, matched budgets,
  pooled + local-only + FedAvg computed by `run_multiseed` as usual.
- **Arms:** FedAvg vs `strategy='topology'` (`distinctiveness`, `T=1.0` — the arm
  already validated in Phase 5; no tuning on this split).
- **Seeds: 42, 43, 44, 45, 46 (5)** — Phase 5's +0.005 effect was fragile at 3.

## 5. Pass/fail criteria (fixed now)

| # | Criterion | Meaning |
|---|---|---|
| C0 | validity checks above | split itself is sound |
| C1 | `fedavg_worst ≤ local_worst − 0.01` (mean over seeds) | **H1:** FedAvg visibly fails the minority (condition reproduced) |
| C2 | `topo − fedavg ≥ +0.005` on worst-hospital or mean | **H2:** Path-B gate passed |
| C3 | decomposition: tail hospital's F1 under topo > under FedAvg | the gain comes from *raising* the tail, not sinking others |

**Falsification rules (binding):**
- If C1 fails → the condition was not reproduced; report that (do NOT re-tune the
  split into a pass and present it as the original design).
- If C1 passes but C2 fails → genuine finding: condition breaks FedAvg but our
  fingerprint weighting doesn't rescue it — analyze why (fingerprint may not see the
  conflict: `resistance_rate` is excluded, `history_prior_rate` must carry it).
- Any split/protocol change after seeing results = **iteration**, appended below with
  date and rationale, and both runs reported.

## 6. Run recipe (Colab, GPU/high-RAM; loads cohort in cell 4 of notebook 07)

```python
# SMOKE (no training): partition validity + per-hospital antibiograms
from amr_fed.data_loader import PK
from amr_fed.partition import antibiogram_conflict, partition_summary, _patient_shared_rate
assign = antibiogram_conflict(df, n_clients=5)
g, niid = partition_summary(df, assign)
print(g, "\nnon-IID(wstd):", round(niid, 4))
assert assign.index.is_unique and len(assign) == df[PK].nunique()
assert (g["n_tests"] / g["n_tests"].sum() >= 0.03).all(), "hospital below 3% size floor"
assert g["resist_rate"].max() - g["resist_rate"].min() >= 0.15, "rate spread too small"
```

```python
# FedAvg arm (5 seeds, v2 protocol)
from amr_fed.federated.run import run_multiseed
from amr_fed.partition import antibiogram_conflict
ab_fed = run_multiseed(seeds=(42, 43, 44, 45, 46), df=df, label="antibiogram/FedAvg",
                       partition_fn=lambda d, s: antibiogram_conflict(d, n_clients=5, seed=s),
                       n_clients=5, rounds=8, local_epochs=4,
                       local_only_epochs=32, hidden=128)
```

```python
# Topology arm (identical protocol)
from amr_fed.federated.run import run_multiseed
from amr_fed.partition import antibiogram_conflict
ab_topo = run_multiseed(seeds=(42, 43, 44, 45, 46), df=df, label="antibiogram/TOPO",
                        partition_fn=lambda d, s: antibiogram_conflict(d, n_clients=5, seed=s),
                        n_clients=5, rounds=8, local_epochs=4,
                        local_only_epochs=32, hidden=128, strategy="topology",
                        topo_mode="distinctiveness", topo_temperature=1.0)
```

```python
# VERDICT: pre-registered C1/C2/C3 against the criteria fixed in this doc
import json
import numpy as np

def _tail_f1(res, key):
    """Per-seed F1 of the tail hospital (highest client id = resistant-heavy block)."""
    vals = []
    for r in res["runs"]:
        pc = r.get(key)
        if not pc:
            continue
        vals.append(pc[max(pc, key=lambda x: int(x))] if isinstance(pc, dict) else pc[-1])
    return vals

d_mean = round(ab_topo["fedavg_best"][0] - ab_fed["fedavg_best"][0], 4)
d_worst = round(ab_topo["worst_fed"][0] - ab_fed["worst_fed"][0], 4)
c1 = ab_fed["worst_fed"][0] <= ab_fed["worst_local"][0] - 0.01
c2 = (d_worst >= 0.005) or (d_mean >= 0.005)

print(f"fedavg  best {ab_fed['fedavg_best']} | worst {ab_fed['worst_fed']} "
      f"| worst_local {ab_fed['worst_local']}")
print(f"topo    best {ab_topo['fedavg_best']} | worst {ab_topo['worst_fed']}")
print(f"C1 FedAvg hurts minority (worst_fed <= worst_local - 0.01): "
      f"{'PASS' if c1 else 'FAIL'}")
print(f"C2 Path-B gate (delta_mean {d_mean} | delta_worst {d_worst}): "
      f"{'PASS' if c2 else 'FAIL'}")

tail_f = _tail_f1(ab_fed, "fedavg_per_client")
tail_t = _tail_f1(ab_topo, "fedavg_per_client")
tail_l = _tail_f1(ab_fed, "local_only_per_client")
tf = tt = tl = None
if tail_f and tail_t:
    tf, tt = round(float(np.mean(tail_f)), 4), round(float(np.mean(tail_t)), 4)
    tl = round(float(np.mean(tail_l)), 4) if tail_l else None
    print(f"C3 tail hospital F1: local {tl} | fedavg {tf} | topo {tt} -> "
          f"{'PASS' if tt > tf else 'FAIL'} (topo must RAISE the tail)")

payload = {"C1": bool(c1), "C2": bool(c2), "delta_mean": d_mean, "delta_worst": d_worst,
           "tail_local": tl, "tail_fedavg": tf, "tail_topo": tt,
           "tail_fedavg_per_seed": tail_f, "tail_topo_per_seed": tail_t}
path = "/content/drive/MyDrive/amr_antibiogram_verdict.json"
with open(path, "w") as f:
    json.dump(payload, f, default=str, indent=1)
print("saved", path)
print("=== COPY EVERYTHING BELOW THIS LINE BACK TO CHAT ===")
print(json.dumps(payload, default=str))
print("=== END ===")
```

## 7. Honesty statement

This experiment is motivated by a specific, cited failure condition — not by
searching until FedAvg loses. Criteria above were fixed before the first run. The
relationship to the label-Dirichlet dead end is disclosed: this is the *mixed,
guarded, shared-pair-conditioned* form of label conflict, which prior literature
(FNR-FL) says behaves differently from either lever alone. All runs — pass or fail —
get recorded in this doc's results section.

## 8. Results (filled in after the run)

*(pending — Colab run not yet performed as of 2026-10-18)*
