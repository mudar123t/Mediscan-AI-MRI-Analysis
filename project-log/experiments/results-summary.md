# Results Summary (master table)

All experiments vs the semester-1 baseline. Test-set numbers only, unless noted. Every row links to its EXP file.

**Baseline status (2026-11-19):** semester-1 numbers were recovered in EXP-000. They are **not a valid baseline**: the image-level split leaks across patients, and the test directory is contaminated. **Segmentation now has a leak-free baseline: EXP-002a, Dice 0.508 (5-fold, patient-disjoint).** **Classification baseline: EXP-003a, balanced accuracy 0.733 / accuracy 0.761 (5-fold, patient-disjoint).** The `clean50` rows are historical only.

| EXP | Date | Model / change | Test set | Seg Dice | Seg IoU | Cls acc [95 % CI] | Cls macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
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
