# EXP-006b — Stabilised router threshold τ1, and CIs for the robustness summary (ADR-0008 (b) + (d))

- **Status:** analysis specified 2026-11-20, **before any result**. This section is committed before the analysis is run.
- **Branch:** `exp/m1-tau1-robustness`, from `main` `fa29301`.
- **Nature:** secondary, post-hoc analysis. It was decided after the EXP-006/008/009 test results were seen ([ADR-0008](../decisions/ADR-0008-m1-gate-decisions.md)). **The frozen-protocol EXP-006/008/009 results remain the primary results** and are not overwritten.

## Pre-specified analysis (fixed before running)
### (b) Stabilised τ1, the "pooled-val" rule
- **Problem (EXP-006):**
  - τ1 is fitted per fold on a single val fold. That fold has only about 15–25 positives of the "crop wrong, full right" event, from 96 in total over 5 folds.
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

## Results
*(to be added after the run)*
