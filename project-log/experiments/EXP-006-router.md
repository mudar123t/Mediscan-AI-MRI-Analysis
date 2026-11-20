# EXP-006 / EXP-007 — Segmentation-quality-aware routing (R1) and conformal referral (R7), in-distribution (M1.3)

- **Date:** 2026-11-20. Analysis only, no training, CPU.
  - Preliminary run with det/mc/tta.
  - Final run with det/mc/tta/ens after the seed run (log `runs/eval_exp006_final.log`).
- **Branch / code:** `exp/m1-uncertainty`.
  - `src/mediscan/router.py`, `src/mediscan/policy_eval.py`, `src/mediscan/metrics/selective.py`, `scripts/eval_exp006.py`.
  - Test: `tests/test_router_selective.py`.
- **Inputs:**
  - EXP-003 logits: ResNet-18 full view on GT views (no mask needed); crop 32 px and masked ROI on **predicted** masks.
  - EXP-005 scores.
- **Split:** per fold, everything is fitted on that fold's val slices (fold (k+1) mod 5): temperatures, τ1, combined regression, conformal thresholds. It is then applied to test fold k. The router's score is chosen on pooled val.
- **Results:** [EXP-006-router/results.json](EXP-006-router/results.json), [test_routed.csv](EXP-006-router/test_routed.csv), [risk_coverage.png](EXP-006-router/risk_coverage.png). Pooled test set: 3,064 slices, 233 patients. CIs from 2,000 paired patient-bootstrap resamples.

## Hypotheses (plan §3.3)
- **H-B1:** the router ≥ the best single view at 100 % coverage.
- **H-B2:** the router has lower AURC/AUGRC than the MSP gate.
- **EXP-007:** split conformal coverage ≈ 1 − α.

## Design
- **Router:** if u ≤ τ1, use the crop-32 view, else the full view.
  - u is the segmentation-uncertainty score.
  - τ1 is fitted on val to minimise AURC. Candidates are the u quantiles and ±∞; ties go to the crop.
  - The referral confidence is the routed MSP after temperature scaling.
- **router+combined:** the same prediction, ranked by a logistic regression on [−log(1−MSP), u, used-crop], fitted on val.
- **Baselines:**
  - full+MSP, crop32+MSP and masked+MSP: each fixed view, with MSP as the gate (Sharma-style).
  - crop32+seg-unc: crop-32 predictions, referred by u.
  - oracle-router: uses the crop unless the crop is wrong and the full view is right. This is an upper bound and uses labels.
- **Score selection.** The score is chosen on pooled val by AUROC for **"crop 32 wrong AND full right"**.
  - That target is the event routing can fix.
  - *This criterion was set after the EXP-004/005 analyses and before EXP-006 was run.* The first draft used "any crop error", which EXP-005 showed is the wrong target.
- **Choice:** `ens:disagree`, val AUROC 0.625; 96 positive val slices out of 3,064.
  - Next: `ens:area_cv` 0.623 and `tta:disagree` 0.593.
  - The preliminary run without the ensemble chose `tta:disagree`; its results were nearly identical: router − crop32 −0.006, router − full AURC −0.011.
- **EXP-007:**
  - Split conformal prediction (LAC) on the val fold, with α ∈ {0.05, 0.1}.
  - A slice is *referred* if its set size ≠ 1, i.e. an empty or ambiguous set.

## Results [V: results.json]
- **τ1 per fold:** [−∞, 0.018, 0.0103, 0.254, 0.447].
- **Crop share on val:** [0.00, 0.18, 0.02, 0.90, 0.94]. The fitted threshold is **unstable across folds**, because the val objective is flat.
- **Test routing:**
  - 40.5 % of slices go to the crop.
  - Median Dice of the slices routed to the crop: 0.894. Median Dice of those routed to the full view: 0.874.
  - So the router barely separates good from bad masks in-distribution.

| Policy | Bal. acc @100 % [95 % CI] | AURC [95 % CI] | AUGRC | E-AURC | acc @90 % coverage |
|---|---|---|---|---|---|
| full + MSP | 0.928 [0.903, 0.948] | 0.0203 [0.0058, 0.0491] | 0.0118 | 0.018 | 0.968 |
| masked + MSP | 0.849 [0.820, 0.875] | 0.0475 | 0.0330 | 0.037 | 0.896 |
| **crop32 + MSP** | **0.941 [0.916, 0.961]** | 0.0097 [0.0037, 0.0198] | 0.0070 | 0.008 | **0.979** |
| crop32 + seg-unc | 0.941 | 0.0490 | 0.0247 | 0.048 | 0.954 |
| **router** | 0.935 [0.912, 0.953] | **0.0089 [0.0039, 0.0171]** | 0.0069 | 0.007 | 0.975 |
| router + combined | 0.935 | 0.0095 | 0.0070 | 0.008 | 0.977 |
| oracle router (labels) | 0.974 [0.954, 0.988] | 0.0068 | 0.0043 | 0.007 | 0.988 |

Paired differences:

| Difference | Estimate [95 % CI] |
|---|---|
| router − crop32: bal. acc | −0.006 [−0.019, +0.008] |
| router − crop32: AURC | −0.0009 [−0.0042, +0.0019] |
| router − crop32: AUGRC | −0.0001 [−0.0021, +0.0019] |
| router − full: bal. acc | +0.007 [−0.002, +0.017] |
| router − full: AURC | **−0.0115 [−0.0327, −0.0006]** |
| router − full: AUGRC | **−0.0050 [−0.0122, −0.0007]** |
| router − full: acc @90 % | **+0.008 [+0.002, +0.017]** |

### EXP-007 conformal referral (pooled test)
| Classifier | α | Coverage | Per class (men. / gli. / pit.) | Mean set size | Referral rate | Empty sets | Acc on singletons |
|---|---|---|---|---|---|---|---|
| router | 0.05 | 0.936 | 0.856 / 0.957 / 0.967 | 0.98 | 0.019 | 0.018 | 0.953 |
| crop32 | 0.05 | 0.950 | 0.894 / 0.972 / 0.958 | 1.00 | 0.018 | 0.007 | 0.956 |
| full | 0.05 | 0.930 | 0.849 / 0.950 / 0.963 | 0.99 | 0.031 | 0.019 | 0.948 |
| router | 0.10 | 0.887 | 0.754 / 0.920 / 0.938 | 0.91 | 0.086 | 0.086 | 0.971 |
| crop32 | 0.10 | 0.902 | 0.811 / 0.927 / 0.932 | 0.93 | 0.074 | 0.074 | 0.974 |
| full | 0.10 | 0.882 | 0.849 / 0.912 / 0.934 | 0.91 | 0.086 | 0.086 | 0.965 |

## Conclusions
- **H-B1 is not supported in-distribution.** The router does not beat the best single view: −0.006 vs crop 32, n.s. It is at least not worse than the full image.
- **H-B2 is supported only against the full-image MSP gate.** AURC −0.0115 and AUGRC −0.0050 (CIs exclude 0). Against the crop32 MSP gate there is no difference.
- **Why:**
  - In-distribution, broken masks are rare: 96 val slices of the "crop wrong, full right" type.
  - The best signal identifies them only weakly (val AUROC 0.625), and τ1 swings between "always full" and "almost always crop" across folds.
  - The oracle (0.974) shows the head-room exists if the event could be detected.
- **EXP-007:**
  - Marginal coverage is close to 1 − α (crop32 is exact; router and full are 0.6–1.8 pp below, within the fold-to-fold variance of split CP with about 600 calibration slices [I]).
  - **Class-conditional coverage fails for meningioma:** 0.75–0.81 at α = 0.1.
  - With 3 well-separated classes, the sets are almost never ambiguous. Referral comes from *empty* sets, and the referral rate is 2–9 %.
  - [I] A class-conditional (Mondrian) CP would address the meningioma gap; this is an option for discussion, not part of the plan.
