# SimFT results (notebook 19, run locally on a Mac CPU)

Pre-registration: `2026-10-28-simft-preregistration.md`. Win rule: SimFT+fine-tune must beat
(a) FedAvg+fine-tune, (b) local-only and (c) SimFT-flat, each with a paired 95% CI above zero, on
the primary metric (unweighted mean over hospitals of test macro-F1). 5 seeds (42-46) per setting.
Chosen temperature: tau = 0.5 in all three settings (validation half B).

## Verdict: SimFT does NOT win in any of the three settings

| Setting | vs FedAvg+FT | vs local-only | vs SimFT-flat |
|---|---|---|---|
| Louvain (3 hospitals) | -0.0001 [-0.0012, +0.0010] no | +0.0045 [+0.0011, +0.0078] yes | +0.0004 [-0.0010, +0.0018] no |
| ward-mix @10% data (10 hospitals) | -0.0004 [-0.0053, +0.0044] no | +0.0168 [+0.0065, +0.0271] yes | +0.0006 [-0.0026, +0.0039] no |
| organism-10 | +0.0022 [-0.0003, +0.0047] no (p=0.07) | +0.0067 [+0.0034, +0.0099] yes | +0.0004 [-0.0007, +0.0016] no |

## Means (test macro-F1, mean over hospitals)

| Setting | Local | FedAvg | FedAvg+FT | SimFT+FT | SimFT-flat+FT |
|---|---|---|---|---|---|
| Louvain | 0.7244 | 0.7201 | 0.7290 | 0.7289 | 0.7284 |
| ward10 @10% | 0.6762 | 0.6894 | 0.6934 | 0.6930 | 0.6924 |
| org10 | 0.7577 | 0.7566 | 0.7622 | 0.7644 | 0.7640 |

## What this means
- The primary metric fails the rule in every setting. In org10 SimFT is nominally best, and the
  size-weighted and worst-hospital differences vs FedAvg+FT have CIs above zero (+0.0025, +0.0033).
  These are secondary metrics, not the pre-registered primary, so they are reported but do not
  change the verdict. The gain is about 0.002 F1.
- The similarity weighting adds nothing measurable. SimFT-flat (same self-floor, uniform neighbours)
  is statistically indistinguishable from SimFT in all three settings. In org10 SimFT-flat+FT also
  edges FedAvg+FT (+0.0018, p=0.05), so whatever small gain exists comes from the self-floor
  (keeping more of each hospital's own model), not from the function-space similarity.
- SimFT beats local-only everywhere, but so does plain FedAvg+fine-tune by a similar margin. That
  is not evidence for SimFT.
- Without the final fine-tune, SimFT is no better than FedAvg (Louvain +0.0035 mean-over-hospitals,
  org10 -0.0001, ward10 -0.0042).

## Caveats
- Same epoch confound as before: the +FT arms get 8 extra local epochs; no budget-matched control.
- Ward10 baselines (FedAvg, FedAvg+FT, local) were produced on a Tesla T4 in notebook 18; SimFT arms
  ran on a Mac CPU. Same seeds and code, but numerics differ across hardware.
- Cross-evaluation exposes every hospital's model to the server and to the other hospitals'
  validation data (a privacy cost not present in FedAvg).
- Ray segfaulted once (exit 139, native Ray code); the run resumed from per-seed checkpoints.
