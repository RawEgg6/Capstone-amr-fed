# Notebook 17 results — final protocol on the organism split (10 hospitals)

**Run:** local (Mac CPU), `notebooks/17_org10_final.ipynb` logic via `final_protocol.run_all`; seeds 42–46 (one fixed partition: seeds vary training only);
rounds 8, local epochs 4, local-only epochs 32, hidden 128. **Raw:** `results/amr_org10_final_all.json` (local; not in git).
**Pre-registration:** `2026-10-28-final-protocol-preregistration.md`. Primary metric `phc_f1`: test macro-F1, size-weighted, every hospital at its own validation-chosen
cutoff, round chosen on validation. v3 temperature chosen on validation: **T = 0.25** (mean val F1 0.7293 / 0.7271 / 0.7262 at T = 0.25 / 0.5 / 1.0).
Hospital sizes (patients): H0 37,952; H4 3,322; H6 4,886; H7 3,543; the others 2,884–2,885.

## 1. Summary (mean ± std over 5 seeds)

| Method | Per-hospital cutoffs (primary) | Shared cutoff | Worst hospital | AUROC |
|---|---|---|---|---|
| Local-only | 0.7219 ± 0.0039 | 0.7219 ± 0.0039 | 0.6959 ± 0.0058 | 0.8417 ± 0.0029 |
| FedAvg | 0.7257 ± 0.0027 | 0.7226 ± 0.0034 | 0.7017 ± 0.0024 | 0.8470 ± 0.0024 |
| **FedAvg + fine-tune** | **0.7308 ± 0.0029** | — | **0.7068 ± 0.0027** | — |
| v3 (T = 0.25) | 0.7268 ± 0.0031 | 0.7219 ± 0.0039 | 0.7027 ± 0.0036 | 0.8480 ± 0.0029 |
| v3-shuffled (T = 0.25) | 0.7265 ± 0.0021 | 0.7211 ± 0.0027 | 0.7022 ± 0.0023 | 0.8482 ± 0.0017 |
| Uniform | 0.7228 ± 0.0025 | 0.7186 ± 0.0028 | 0.6988 ± 0.0020 | 0.8443 ± 0.0020 |
| Topo v1 | 0.7239 ± 0.0028 | 0.7208 ± 0.0030 | 0.6999 ± 0.0025 | 0.8449 ± 0.0030 |
| FedProx | 0.7246 ± 0.0029 | 0.7220 ± 0.0033 | 0.7006 ± 0.0028 | 0.8466 ± 0.0025 |
| FedMedian | 0.7228 ± 0.0023 | 0.7184 ± 0.0026 | 0.6986 ± 0.0020 | 0.8443 ± 0.0018 |

The selected round is 8 for every federated arm and seed.

## 2. Pre-registered verdict: v3 does NOT win

| Primary comparison (paired by seed, `phc_f1`) | Mean difference | 95% interval | p | Seeds > 0 | Rule |
|---|---|---|---|---|---|
| v3 − local-only | +0.0049 | [+0.0025, +0.0073] | 0.01 | 5/5 | passes |
| v3 − FedAvg + fine-tune | −0.0039 | [−0.0061, −0.0018] | 0.01 | 0/5 | **fails (fine-tune is better)** |
| v3 − v3-shuffled | +0.0003 | [−0.0030, +0.0036] | 0.81 | 3/5 | **fails (fingerprint carries no signal)** |

Worst-hospital F1 gives the same picture (v3 − local +0.0068, v3 − fine-tune −0.0041, v3 − shuffled +0.0005). This repeats the Louvain verdict
(`2026-10-28-final-protocol-louvain-results.md`) on a split where hospitals differ much more.

## 3. Secondary comparisons (paired by seed, `phc_f1`)

| Comparison | Mean difference | 95% interval | p | Seeds > 0 |
|---|---|---|---|---|
| FedAvg + fine-tune − local-only | +0.0088 | [+0.0059, +0.0118] | <0.01 | 5/5 |
| FedAvg − local-only | +0.0038 | [+0.0013, +0.0062] | 0.01 | 5/5 |
| v3 − FedAvg | +0.0011 | [−0.0000, +0.0023] | 0.05 | 4/5 |
| Uniform − FedAvg | −0.0029 | [−0.0040, −0.0018] | <0.01 | 0/5 |
| FedAvg − FedMedian | +0.0029 | [+0.0017, +0.0040] | <0.01 | 5/5 |
| FedAvg − FedProx | +0.0012 | [+0.0000, +0.0023] | 0.05 | 4/5 |
| Topo v1 − Uniform | +0.0011 | [+0.0003, +0.0019] | 0.02 | 5/5 |

## 4. Per-hospital test F1 (mean over seeds)

| Hospital | Patients | Local-only | FedAvg | FedAvg + FT | v3 | v3-shuffled | Uniform | Topo v1 | FedMedian | FedProx |
|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 37,952 | 0.6959 | 0.7017 | 0.7068 | 0.7027 | 0.7022 | 0.6988 | 0.6999 | 0.6986 | 0.7006 |
| H1 | 2,885 | 0.7819 | 0.7873 | 0.7897 | 0.7818 | 0.7828 | 0.7811 | 0.7842 | 0.7805 | 0.7876 |
| H2 | 2,884 | 0.7388 | 0.7335 | 0.7412 | 0.7342 | 0.7351 | 0.7300 | 0.7302 | 0.7313 | 0.7343 |
| H3 | 2,884 | 0.7450 | 0.7465 | 0.7567 | 0.7526 | 0.7517 | 0.7454 | 0.7417 | 0.7471 | 0.7461 |
| H4 | 3,322 | 0.7673 | 0.7635 | 0.7729 | 0.7675 | 0.7671 | 0.7639 | 0.7665 | 0.7627 | 0.7634 |
| H5 | 2,884 | 0.7558 | 0.7431 | 0.7530 | 0.7498 | 0.7490 | 0.7423 | 0.7411 | 0.7423 | 0.7413 |
| H6 | 4,886 | 0.7798 | 0.7867 | 0.7882 | 0.7849 | 0.7859 | 0.7828 | 0.7874 | 0.7827 | 0.7856 |
| H7 | 3,543 | 0.7591 | 0.7677 | 0.7708 | 0.7672 | 0.7659 | 0.7604 | 0.7695 | 0.7642 | 0.7651 |
| H8 | 2,884 | 0.7786 | 0.7736 | 0.7735 | 0.7776 | 0.7770 | 0.7742 | 0.7713 | 0.7724 | 0.7688 |
| H9 | 2,884 | 0.7750 | 0.7625 | 0.7689 | 0.7657 | 0.7658 | 0.7634 | 0.7535 | 0.7624 | 0.7604 |

Hospitals beating local-only (of 10): FedAvg + fine-tune 7, FedAvg 5, v3 5, v3-shuffled 5, FedProx 5, Uniform 4, Topo v1 4, FedMedian 4. Hospitals H2, H5, H9 still do worse
than local-only under every method except fine-tuning, which narrows but does not remove the gap.

## 5. Weights

- **Topo v1** (fixed, all-rows fingerprints): 0.732 on H4 (3,322 patients), 0.194 on H0, 0.004–0.025 on the other eight — the same concentrated weights as in notebook 15.
- **v3 (T = 0.25):** self-weights 0.49–0.996 depending on the hospital (H4 keeps 0.98–0.996 of its own model: it is the most distinct hospital), mean off-diagonal weight 0.031–0.039.
  v3-shuffled has the same self-weights on different hospitals.

## 6. What the numbers say

- **The fingerprint similarity carries no usable signal here either** (v3 − shuffled +0.0003, p = 0.81), on the split where hospitals differ most. Combined with Louvain, topology-fingerprint
  weighting is not supported on either split.
- **FedAvg + fine-tune is again the best arm:** +0.0088 over local-only (5/5 seeds, 7 of 10 hospitals) and +0.0039 over v3 (0/5 seeds).
- **With per-hospital cutoffs federation beats local-only on this split** (FedAvg +0.0038, 5/5 seeds), unlike the shared-cutoff result in `2026-10-28-validation-selected-results.md`
  (a tie): a shared cutoff was costing the federated arms about 0.003.
- **Size-weighted FedAvg is the best non-personalized aggregator:** equal weighting (Uniform) is 0.0029 worse, FedMedian 0.0029 worse, FedProx 0.0012 worse.
- **Topo v1 > Uniform by +0.0011 (5/5 seeds)** should not be read as a fingerprint effect: its weights give the largest hospital 0.19 against Uniform's 0.10, which is a partial return to size-weighting
  (Uniform − FedAvg is −0.0029), and v3 with real fingerprints does not beat its shuffled control.

## 7. Caveats

- **Budget confound for fine-tuning:** fine-tuning gets 8 extra local epochs (40 vs 32) and everything is still improving at the budget; part of the fine-tune advantage may be extra training
  (see `2026-10-28-low-data-and-new-site-results.md`, caveat 1). The budget-matched control was not run.
- One fixed partition (five seeds vary training only): the intervals do not cover other ten-hospital splits.
- The size-weighted score is dominated by H0 (57% of patients); worst-hospital and per-hospital tables are given beside it.
- Selected round is 8 everywhere: all federated arms were still improving at the training budget.
