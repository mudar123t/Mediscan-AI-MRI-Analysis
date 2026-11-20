# EXP-010 — Explainability as a diagnostic (R4, M3)

- **Status:** pre-specified 2026-11-20, **before any code or result**. This section is committed first.
- **Branch:** `research/m3-r4-xai`, from `main`.
- **Decision:** Kareem, 2026-11-20 — "go with R4 then stop" (M3; ADR-0006 extras).
- **Literature:** checking where the classifier's attention falls relative to the tumour mask was partly done before (Filvantorkaman 2025, idea-matrix R4). **Linking it to correctness and to routing is the open part** [idea-matrix, Phase 4].

## Questions
1. **Where do the classifiers look?** The crop-32 classifier (on predicted masks) and the full-image classifier: inside the tumour, at its border, or elsewhere?
2. **Is it linked to correctness?** Do wrong predictions look outside the tumour more than correct ones? Does an attribution feature that can be computed at test time add to MSP for predicting errors?
3. **Can it be a routing signal?** Can attribution signal "crop wrong, full right" better than `ens:disagree` (val AUROC 0.625, EXP-006), and does a router driven by it do better?

## Pre-specified method (fixed before running)
- **Models / data:**
  - The EXP-003 ResNet-18 classifiers of each fold: `exp003_r18_crop32` (input = 32 px context crop built from the EXP-002c predicted mask, threshold 0.5 + largest component) and `exp003_r18_full`.
  - Val and test slices of all 5 folds. No training.
  - Inference in fp32. [Check] The logits must match the stored EXP-003 `logits_{val,test}_{pred|gt}.npz` (max |Δ| < 0.05, identical argmax on ≥ 99.9 % of slices); otherwise stop.
- **Grad-CAM:**
  - Target layer: the last block (`layer4`, 8×8 for a 256² input). Gradients of the **predicted class** logit. Positive part (ReLU), bilinear upsampling to 256².
  - For the crop view, the map is resized to the crop box and **back-projected into the original 256² image**. It is zero outside the box. All fractions are therefore measured in original-image coordinates for both views.
  - The map is normalised to unit mass. If its mass is 0, the fractions are NaN and are treated as the most suspicious value, as in EXP-005/006.
- **Regions:**
  - Built for R = GT mask (an oracle, analysis only) and R = predicted mask (computable at test time).
  - `in` = R; `ring` = dilate(R, 16 px) \ R; `out` = everything else.
  - [I] 16 px is half a Grad-CAM cell of the full view (cells are 32 px); a thinner ring would sit below the map's resolution.
  - Empty predicted mask → the pred-region fractions are NaN.
- **Analysis A, descriptive (pooled test, 3,064 slices, 233 patients):**
  - Mean `in` / `ring` / `out` (GT regions) per view, for correct vs wrong predictions and per class.
  - Paired patient-bootstrap CI (2,000 resamples) for correct − wrong of `in`, per view.
- **Analysis B, correctness:**
  - AUROC of `1 − (in + ring)` for that view's error, using GT regions (oracle) and predicted regions (deployable), with patient-bootstrap CIs.
  - "Adds to MSP" exactly as in EXP-005: per fold, a logistic regression on val with features [−log(1−MSP)] vs [−log(1−MSP), feature]; criterion = mean within-fold test AUROC; paired bootstrap. Done for the two deployable features against each view's own error.
- **Analysis C, routing (deployable only):**
  - Two candidate signals: `cam_full` = 1 − (in + ring) of the **full-image** classifier w.r.t. the predicted mask ("the full classifier does not look where the mask is"), and `cam_crop` = the same for the crop classifier.
  - The signal is chosen on **pooled val** by AUROC for "crop32 wrong AND full right" (the EXP-006 criterion). Reported next to `ens:disagree` (0.625).
  - Router: `policy_eval.fit_params` with that score. τ1 is fitted per fold on that fold's val, exactly the frozen EXP-006 rule, kept for comparability even though EXP-006b showed it is unstable.
  - Test metrics: balanced accuracy, AURC; paired patient-bootstrap differences vs crop32+MSP, full+MSP and the frozen `ens:disagree` router.
  - **No other signal or rule is tried.** In-distribution only: shift and external are not part of this pre-specification.
- **Figures:** (1) a bar chart of `in`/`ring`/`out` by view × correct/wrong; (2) an example panel: for 3 correct and 3 wrong test slices, the image with GT/pred contours and the Grad-CAM of both views.

## Results
*(to be added after the run)*
