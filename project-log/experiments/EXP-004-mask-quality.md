# EXP-004 — When does segmentation help classification? (R3, M1.1)

- **Date:** 2026-11-20. Inference only, about 50 min on the GPU, run from about 09:00 while the seed run shared the GPU.
- **Branch / code:** `exp/m1-r3-seg-helps`, merged into `exp/m1-uncertainty`. Code: `src/mediscan/degrade.py`, `scripts/run_exp004.py`, `scripts/eval_exp004.py`. Test: `tests/test_degrade.py`.
- **Config:** `configs/exp004_mask_quality.yaml`. Raw output: `runs/exp004_mask_quality/fold{k}.npz` (git-ignored).
- **Models:** the EXP-003 ResNet-18 view classifiers of each fold (masked, crop 0/8/16/32 px), trained on GT-mask views and **not retrained**. The full-image classifier is the reference.
- **Split / seed:** the EXP-003 folds, test fold k only. The degradation RNG is `default_rng([0, row, cell])`, so it is deterministic.
- **Hardware:** RTX 3050 Laptop, 4 GB.
- **Results:** [EXP-004-mask-quality/controlled.csv](EXP-004-mask-quality/controlled.csv), [predicted_by_dice.csv](EXP-004-mask-quality/predicted_by_dice.csv), [results.json](EXP-004-mask-quality/results.json), figure [accuracy_vs_mask_quality.png](EXP-004-mask-quality/accuracy_vs_mask_quality.png). Pooled test is 3,064 slices from 233 patients, with 2,000 patient-bootstrap resamples paired against the full view.

## Question and hypothesis (plan §3.3)
How does the classifier's balanced accuracy depend on mask quality, and does a context margin soften the drop? The plan expected accuracy to fall with mask quality and a margin to soften the fall (Cheng 2015 direction [V]).

## Design
1. **Controlled degradation of the GT masks.**
   - Four kinds: `erode`, `dilate`, `shift` (random direction) and `blob` (a false-positive disk far from the tumor, inside the head).
   - Target Dice values: 0.9, 0.8, 0.7, 0.5 and 0.3; cell 0 is the undegraded GT.
   - Erode and dilate are iterated until Dice ≤ target. The blob area is 2|G|(1/target − 1), so the Dice of GT ∪ blob equals the target.
   - The achieved Dice and the share of slices within ±0.05 of the target are stored in `controlled.csv`.
2. **Real predicted masks.** The EXP-002c masks are binned by their Dice against GT: [0, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.85] and [0.85, 1].

## Results [V: controlled.csv, predicted_by_dice.csv]
### Controlled degradation: balanced accuracy (full image = 0.928)
| kind | target Dice | masked | crop 0 | crop 8 | crop 16 | crop 32 |
|---|---|---|---|---|---|---|
| none (GT) | 1.0 | 0.870 | 0.881 | 0.928 | 0.939 | **0.954** |
| erode | 0.7 / 0.3 | 0.710 / 0.588 | 0.719 / 0.501 | 0.911 / 0.722 | 0.933 / 0.896 | **0.949 / 0.933** |
| dilate | 0.7 / 0.3 | 0.680 / 0.409 | 0.869 / 0.626 | 0.910 / 0.689 | 0.932 / 0.783 | **0.950 / 0.833** |
| shift | 0.7 / 0.3 | 0.738 / 0.566 | 0.844 / 0.628 | 0.913 / 0.812 | 0.939 / 0.893 | **0.949 / 0.936** |
| blob | 0.9 / 0.3 | **0.809 / 0.641** | 0.511 / 0.400 | 0.517 / 0.407 | 0.573 / 0.452 | 0.636 / 0.483 |

### Real predicted masks, by Dice bin (crop 32 vs full image, paired)
| Dice bin | slices / patients | crop 32 bal. acc | full | Δ [95 % CI] |
|---|---|---|---|---|
| [0, 0.3] | 224 / 90 | 0.641 | 0.889 | **−0.248 [−0.359, −0.138]** |
| [0.3, 0.5] | 114 / 72 | 0.941 | 0.866 | +0.075 [−0.130, +0.265] |
| [0.5, 0.7] | 278 / 113 | 0.946 | 0.864 | **+0.082 [+0.019, +0.161]** |
| [0.7, 0.85] | 640 / 162 | 0.939 | 0.910 | +0.029 [−0.006, +0.067] |
| [0.85, 1] | 1,808 / 216 | 0.947 | 0.928 | +0.019 [−0.001, +0.040] |

## Conclusions
- **The hypothesis holds for under- and mis-segmentation.** Accuracy falls with mask quality, and the 32 px margin almost removes the fall. Under erosion at Dice 0.3 the crop is still at 0.933, and under shift at Dice 0.3 at 0.936. The semester-1 masked ROI is the most fragile view under erode, dilate and shift.
- **A false-positive region breaks every crop.** It moves the crop box away from the tumor: crop 32 is at 0.636 already at Dice 0.9. The masked view is the most robust here because the tumor pixels are still inside the mask.
- **Real masks.** The crop helps wherever the predicted mask overlaps the tumor at all (Dice ≥ 0.3). It fails badly on the 224 slices with Dice < 0.3. 172 of them have zero overlap [V: `runs/exp002c_unet_r34/fold*/metrics_test.csv`], so the predicted mask is on the wrong structure. **This is the failure mode a router must catch** → motivates EXP-005/006.
- Limitation: Dice bins of real masks are small for [0.3, 0.5], so the CIs are wide.
