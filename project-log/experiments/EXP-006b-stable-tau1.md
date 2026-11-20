# EXP-006b — Stabilised router threshold τ1, and CIs for the robustness summary (ADR-0008 (b) + (d))

- **Status:** analysis specified 2026-11-20 before any result (`f547997`), run 17:31 (`9ced2ce`). Results below.
- **Branch:** `exp/m1-tau1-robustness`, from `main` `fa29301`.
- **Nature:** secondary, post-hoc analysis. It was decided after the EXP-006/008/009 test results were seen ([ADR-0008](../decisions/ADR-0008-m1-gate-decisions.md)). **The frozen-protocol EXP-006/008/009 results remain the primary results** and are not overwritten.

## Pre-specified analysis (fixed before running)
### (b) Stabilised τ1, the "pooled-val" rule
- **Problem (EXP-006):**
  - τ1 is fitted per fold on a single val fold. That fold has only about 19 positives on average [I: 96/5] of the "crop wrong, full right" event, from 96 in total over 5 folds.
  - The fitted threshold swings from "always full" to "almost always crop": val crop share 0.00–0.94.
- **Rule:**
  - For test fold k, τ1 is fitted with the unchanged `router.fit_tau1`: same objective (val AURC of the routed classifier, MSP ranking), same candidate grid (51 quantiles of u, plus ±∞), same tie rule.
  - It is fitted on the **pooled val frames of the fold models j ≠ (k − 1) mod 5**, i.e. 4 val folds and about 2,450 slices.
  - Model j's val fold is (j + 1) mod 5. Excluding j = k − 1 removes the only val frame that contains test fold k, so **no test-fold patient is used**.
  - Each pooled frame uses its own model's temperature-scaled probabilities, with temperatures fitted on that frame.
- **Everything else is as in EXP-006, for model k:**
  - The temperatures, the router+combined logistic regression and the conformal thresholds are fitted on model k's own val frame, using the new τ1.
  - The score stays frozen: `ens:disagree`.
- **Evaluation:** the same policies and reports as the primary analyses.
  - In-distribution test (as EXP-006), with paired patient bootstrap, 2,000 resamples.
  - The 13 synthetic-shift conditions (as EXP-008).
  - The external BDNeuro clean set, per fold model set (as EXP-009, image-level bootstrap).
  - Also reported: τ1 and the val/test crop share per fold, for the old and the new rule.
- **No other rule is tried.** If this rule does not help, that is the result.

### (d) CIs for the robustness summary over the 13 EXP-008 conditions
- **Statistics:** for each policy (full+MSP, crop32+MSP, masked+MSP, router as frozen, router with pooled-val τ1, oracle):
  - **worst-case** (min) and **mean** of balanced accuracy over the 13 conditions (clean + 4 kinds × 3 severities);
  - the same for AURC (worst = max);
  - the paired differences router − crop32 and router − full, for both router variants.
- **CIs:** paired patient bootstrap with 2,000 resamples. The **same patient resample is applied to all 13 conditions**, with slices aligned on `row`. The min and mean are recomputed in each resample.
- **Inputs:** the existing EXP-008 det+ens inference (`runs/exp008_shift/*.csv`). No new inference.

## Results [V: EXP-006b-stable-tau1/results.json, log `runs/eval_exp006b.log`; run 2026-11-20 17:31, CPU]
### (b) Pooled-val τ1 does **not** stabilise the router
| Fold | τ1 frozen (own-val crop share) | τ1 pooled-val (own-val crop share) |
|---|---|---|
| 0 | −∞ (0.00) | 0.0256 (0.27) |
| 1 | 0.0180 (0.18) | 0.0256 (0.37) |
| 2 | 0.0103 (0.02) | 0.0329 (0.29) |
| 3 | 0.2540 (0.90) | 0.2553 (0.90) |
| 4 | 0.4468 (0.94) | **−∞ (0.00)** |

- **Folds 3 and 4 flip.** Their pooled sets share 3 of 4 val frames (fold 3 excludes model 2's frame, fold 4 excludes model 3's), yet τ1 goes from "90 % crop" to "always full". Changing one quarter of the data flips the decision. **The val-AURC objective is nearly flat** between routing many slices to the crop and routing none.
- **In-distribution:**
  - router(pooled) 0.933 vs frozen router 0.935: Δ −0.002 [−0.012, +0.008].
  - AURC 0.0200 vs 0.0089: Δ +0.0111 [−0.0002, +0.0332].
  - vs crop32: bal. acc −0.008 [−0.019, +0.004].
- **Shift (13 conditions):**
  - Worse than the frozen router on mean AURC: +0.0134 [+0.0019, +0.0285].
  - Its gain over the crop under noise/blur remains (blur s3 +0.163 [+0.125, +0.204]).
  - Under gamma it loses more to the crop (−0.054 to −0.091), since more slices go to the full view.
- **External:**
  - Mean bal. acc 0.913, range 0.868–0.932, against 0.922 for the frozen router and 0.931 for crop32.
  - Fold 4, now always full, costs −0.065 [−0.077, −0.053] vs crop32.
- **Conclusion (b):** the pre-specified stabilisation fails. The instability is not caused by too little val data. It comes from the fitting objective: AURC differences between thresholds are tiny compared with their sampling noise.
  - [I] A threshold chosen by a different criterion (e.g. a fixed Dice-proxy or a target crop share) might be stable. It was **not** pre-specified, so it is not tried here; it is a possible future direction.

### (d) Robustness summary of the frozen router, with CIs (paired patient bootstrap, 2,000 resamples, same resample in all 13 conditions)
| Policy | Worst-case bal. acc [95 % CI] | Mean bal. acc | Worst-case AURC | Mean AURC |
|---|---|---|---|---|
| full + MSP | 0.581 [0.543, 0.611] | 0.793 [0.773, 0.811] | 0.242 [0.185, 0.319] | 0.090 [0.067, 0.116] |
| crop32 + MSP | 0.535 [0.501, 0.554] | 0.801 [0.781, 0.819] | 0.324 [0.287, 0.390] | 0.090 [0.076, 0.106] |
| masked + MSP | 0.372 [0.339, 0.407] | 0.709 [0.686, 0.730] | 0.460 [0.392, 0.539] | 0.154 [0.131, 0.181] |
| **router (frozen)** | **0.607 [0.583, 0.630]** | **0.813 [0.794, 0.830]** | **0.192 [0.172, 0.258]** | **0.072 [0.057, 0.089]** |
| router (pooled τ1) | 0.602 [0.570, 0.622] | 0.804 [0.785, 0.822] | 0.218 [0.181, 0.293] | 0.085 [0.064, 0.111] |

| Frozen router minus | Worst-case bal. acc | Mean bal. acc | Worst-case AURC | Mean AURC |
|---|---|---|---|---|
| crop32 + MSP | **+0.072 [+0.051, +0.110]** | +0.012 [−0.001, +0.025] | **−0.131 [−0.182, −0.070]** | **−0.018 [−0.031, −0.006]** |
| full + MSP | **+0.025 [+0.001, +0.063]** | **+0.020 [+0.011, +0.030]** | −0.050 [−0.086, +0.011] | **−0.018 [−0.032, −0.006]** |

- **Conclusion (d):**
  - Across the 13 shift conditions, the frozen router has a significantly better **worst case** than both fixed views. It also has a significantly lower **mean AURC** (referral risk) than both, and a higher mean accuracy than the full view.
  - This turns the EXP-008 point-estimate reading ("never collapses") into a supported claim. **H-B3 support is strengthened for the robustness formulation.**
  - [I] Caveat: a bootstrap of a minimum is biased and its CI is approximate. The 13 conditions were chosen by us (EXP-008 levels fixed before results).
