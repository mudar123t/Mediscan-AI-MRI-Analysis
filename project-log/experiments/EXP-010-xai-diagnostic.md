# EXP-010 — Explainability as a diagnostic (R4, M3)

- **Status:** pre-specified 2026-11-20 before any code or result (`e28ae5a`); run and analysed the same evening. Results below.
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

## Results [V: EXP-010-xai-diagnostic/results.json; logs `runs/exp010.log`, `runs/eval_exp010.log`]
- **Run:** 2026-11-20 18:57 → 19:02 (GPU, 5 folds, 12,256 slice × view rows); analysis about 5 min CPU.
- **Pre-specified logit check passed on every fold:** max |Δ| vs stored EXP-003 logits 0.034–0.048 (fp32 vs the original AMP; limit 0.05); argmax agreement 100 %.
- **Hardware:** RTX 3050 Laptop. **Seed:** 0.

### A. Where do the classifiers look? (pooled test, GT regions)
| View · outcome | n | inside tumour | 16 px border | elsewhere |
|---|---|---|---|---|
| crop 32 · correct | 2,905 | 0.275 | 0.425 | 0.300 |
| crop 32 · wrong | 159 | 0.231 | 0.378 | 0.391 |
| full · correct | 2,873 | 0.074 | 0.146 | 0.780 |
| full · wrong | 191 | 0.050 | 0.131 | 0.820 |

- **Correct − wrong ("inside" share):**
  - crop 32: +0.045 [−0.010, +0.102] (n.s.)
  - full: **+0.024 [+0.012, +0.036]**
- **Per class, inside / border share:**
  - crop 32: meningioma 0.371 / 0.426, glioma 0.264 / 0.359, pituitary 0.211 / 0.518.
  - full: meningioma 0.098 / 0.181, glioma 0.090 / 0.159, pituitary 0.027 / 0.096.
- **Context, descriptive and not pre-specified:** a *uniform* map would put 0.017 inside the tumour and 0.041 in the border (full view; mean tumour area 1.7 % of the image; pituitary 0.8 %).
  - So the full-image classifier attends to the tumour **4.3× more than chance** (in + border: 3.8×).
  - Still, 78 % of its attribution lies outside the tumour and its border.
  - [I] Grad-CAM at `layer4` has 32 px cells, so the full view's maps cannot resolve small tumours. The in-share is a lower bound on how much the tumour matters.

### B. Is attribution linked to correctness?
| Error of | Feature | AUROC [95 % CI] |
|---|---|---|
| crop 32 | 1 − (in + border), GT regions (oracle) | 0.537 [0.446, 0.628] |
| crop 32 | 1 − (in + border), predicted regions | 0.448 [0.355, 0.538] |
| crop 32 | MSP | **0.897 [0.859, 0.935]** |
| full | 1 − (in + border), GT regions (oracle) | 0.584 [0.508, 0.656] |
| full | 1 − (in + border), predicted regions | 0.571 [0.494, 0.644] |
| full | MSP | **0.871 [0.803, 0.928]** |

"Adds to MSP" (mean within-fold test AUROC, val-fitted regression):
- crop 32: 0.909 → 0.908, Δ −0.001 [−0.004, +0.002].
- full: 0.886 → 0.894, Δ +0.008 [−0.001, +0.014] (n.s.).

### C. Attribution as a routing signal
- **Pooled-val AUROC** for "crop wrong, full right": `cam:full` 0.587, `cam:crop` 0.459, against `ens:disagree` 0.625. Chosen (pre-specified rule, among the CAM signals): `cam:full`.
- **τ1 per fold (cam:full):** 0.338, 0.736, 0.726, 0.886, 0.858. Test crop share 0.41 (ens router: 0.41).

| Policy | Bal. acc [95 % CI] | AURC |
|---|---|---|
| crop32 + MSP | 0.941 [0.916, 0.961] | 0.010 |
| router (ens:disagree, EXP-006) | 0.935 [0.912, 0.953] | 0.009 |
| **router (cam:full)** | 0.927 [0.903, 0.947] | 0.020 |
| full + MSP | 0.928 [0.903, 0.948] | 0.020 |
| oracle router | 0.974 | 0.007 |

router(cam) minus:

| Baseline | Bal. acc | AURC |
|---|---|---|
| crop32 | **−0.014 [−0.026, −0.002]** | +0.010 [−0.000, +0.029] |
| full | −0.001 [−0.011, +0.008] | −0.001 [−0.004, +0.002] |
| router(ens) | **−0.008 [−0.016, −0.001]** | **+0.011 [+0.001, +0.032]** |

### Figures
- [attribution_shares.png](EXP-010-xai-diagnostic/attribution_shares.png): stacked shares by view × outcome.
- [gradcam_examples.png](EXP-010-xai-diagnostic/gradcam_examples.png): fold 0 test, the first 3 correct and first 3 wrong crop-32 predictions (pre-specified rule).

## Conclusions
1. **Descriptive (supported):**
   - The context crop concentrates attribution on the tumour and its border: about 70 % of the mass, against about 22 % for the full-image classifier.
   - The full-image classifier uses mostly non-tumour context, though still 4.3× above chance on the tumour.
   - This fits EXP-003/004: the crop works because it forces attention onto the lesion and its border.
2. **Correctness (not supported):**
   - Attribution location is a weak error signal (AUROC 0.45–0.58, at most 0.58 even with GT regions).
   - It adds nothing significant to MSP. Wrong full-image predictions look slightly less at the tumour (+0.024), but the effect is too small to use.
3. **Routing (not supported):**
   - `cam:full` is a weaker routing signal than `ens:disagree` (val 0.587 vs 0.625).
   - Its router is worse than both the crop and the ens router.
   - **R4 does not fix the router's signal bottleneck.** This is reported as a negative result.
4. **Qualitative [I, illustration only, not evidence]:**
   - In the example panel, the three wrong crop predictions are slices of one patient's sagittal meningioma near the sella (rows 620–622, PID 113098 [V]).
   - The crop classifier looks at the tumour (in-share 0.35–0.57) but calls it pituitary. The full-image classifier, using wider context, gets 2 of 3 right.
   - One plausible failure mode of the crop: **removing the location context** that separates sellar meningiomas from pituitary adenomas. Untested; a possible follow-up would be per-location error analysis.
- **Limitations:**
   - Grad-CAM at 32 px resolution.
   - One attribution method only.
   - In-distribution only.
   - The 16 px ring was chosen a priori, not tuned.
