# Results Summary (master table)

All experiments vs the semester-1 baseline. Test-set numbers only, unless noted. Every row links to its EXP file.

**Baseline status (2026-11-19):** semester-1 numbers were recovered in EXP-000. They are **not a valid baseline**: the image-level split leaks across patients, and the test directory is contaminated. **Segmentation now has a leak-free baseline: EXP-002a, Dice 0.508 (5-fold, patient-disjoint).** **Classification baseline: EXP-003a, balanced accuracy 0.733 / accuracy 0.761 (5-fold, patient-disjoint).** The `clean50` rows are historical only.

| EXP | Date | Model / change | Test set | Seg Dice | Seg IoU | Cls acc [95 % CI] | Cls macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| [EXP-009](EXP-009-external.md) | 2026-11-20 | External BDNeuro-MRI (3,030 non-duplicate images), crop32 / router / full | external, 5 fold model sets | — | — | bal. **0.931** / 0.922 / 0.904 (mean) | — | 1,682 BDNeuro images duplicate Figshare (33 % of its test split) |
| [EXP-008](EXP-008-shift.md) | 2026-11-20 | Router under synthetic shift (13 conditions) | 5-fold OOF, corrupted | — | — | worst-case bal. router 0.607 vs crop32 0.535 vs full 0.581 | — | router − crop32 +0.145 [+0.103, +0.186] at blur s3 |
| [EXP-006](EXP-006-router.md) | 2026-11-20 | Seg-quality-aware router (ens:disagree) vs crop32 / full; conformal referral (EXP-007) | 5-fold OOF, 3,064 / 233 patients | — | — | bal. 0.935 [0.912, 0.953] (crop32 0.941, n.s.) | — | AURC vs full −0.0115 [−0.0327, −0.0006]; oracle 0.974 |
| [EXP-005](EXP-005-uncertainty.md) | 2026-11-20 | Seg-uncertainty signals (det/MC/TTA/3-seed ensemble) | 5-fold OOF, 3,064 / 233 patients | — | — | — | — | seg-failure AUROC 0.899 (ens:disagree); adds nothing to MSP (0.909) |
| [EXP-004](EXP-004-mask-quality.md) | 2026-11-20 | Accuracy vs mask quality × context margin | 5-fold OOF, 3,064 / 233 patients | — | — | crop32 0.933 at eroded Dice 0.3; 0.636 with FP blob at Dice 0.9 | — | real Dice < 0.3: crop32 −0.248 vs full |
| [EXP-003](EXP-003-classification.md) | 2026-11-20 | **ResNet-18, context crop 32 px, predicted masks (best automatic pipeline)** | 5-fold OOF, 3,064 / 233 patients | — | — | acc 0.948; **bal. 0.941 [0.916, 0.961]** | 0.942 | ECE 0.033 → 0.005 after TS |
| [EXP-003](EXP-003-classification.md) | 2026-11-20 | ResNet-18, context crop 32 px, GT masks (upper bound) | 5-fold OOF, 3,064 / 233 patients | — | — | acc 0.961; bal. 0.954 [0.931, 0.974] | 0.956 | +0.026 vs full (p 0.005) |
| [EXP-003](EXP-003-classification.md) | 2026-11-20 | ResNet-18, full image | 5-fold OOF, 3,064 / 233 patients | — | — | acc 0.938; bal. 0.928 [0.903, 0.948] | 0.928 | EffNet-B0 equal (Δ +0.002) |
| [EXP-003](EXP-003-classification.md) | 2026-11-20 | ResNet-18, masked ROI (semester-1 view), predicted masks | 5-fold OOF, 3,064 / 233 patients | — | — | acc 0.860; bal. 0.849 [0.820, 0.875] | 0.849 | −0.079 vs full |
| [EXP-003](EXP-003-classification.md) | 2026-11-20 | SimpleCNN, old recipe, full image (**fair semester-1 baseline**) | 5-fold OOF, 3,064 / 233 patients | — | — | acc 0.761; bal. 0.733 [0.703, 0.765] | 0.731 | meningioma recall 0.477 |
| [EXP-002](EXP-002-segmentation.md) | 2026-11-19 | **c: ResNet-34 U-Net, new recipe (M1 segmenter)** | 5-fold OOF, 3,064 / 233 patients | **0.786 [0.766, 0.806]** | 0.695 | — | — | HD95 14.4 px; glioma 0.685 |
| [EXP-002](EXP-002-segmentation.md) | 2026-11-19 | b: semester-1 UNet, new recipe | 5-fold OOF, 3,064 / 233 patients | 0.759 [0.735, 0.782] | 0.668 | — | — | HD95 25.0 px; +0.250 vs a |
| [EXP-002](EXP-002-segmentation.md) | 2026-11-19 | a: semester-1 UNet, old recipe (**fair semester-1 baseline**) | 5-fold OOF, 3,064 / 233 patients | 0.508 [0.475, 0.541] | 0.402 | — | — | HD95 56.8 px; 6.4 % empty preds |
| [EXP-001](EXP-001-data-pipeline.md) | 2026-11-19 | Data pipeline (no model) | all 3,064 / 233 patients | — | — | — | — | 0 patients across folds; 0 empty masks; border IoU median 0.970 |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 U-Net (5 ep) | pilot test clean50 | 0.411 | 0.309 | — | — | image-level split; possible patient leakage |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, full image | pilot test clean50 | — | — | 0.800 [0.670, 0.888] | 0.814 | dir110 (contaminated) = 0.909 |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, ROI from GT mask | pilot test clean50 | — | — | 0.740 [0.604, 0.841] | 0.746 | upper bound; needs GT at test time |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, ROI from predicted mask | pilot test clean50 | — | — | 0.460 [0.330, 0.596] | 0.456 | majority-class rate = 0.44 |
