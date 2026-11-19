# Results Summary (master table)

All experiments vs the semester-1 baseline. Test-set numbers only, unless noted. Every row links to its EXP file.

**Baseline status (2026-11-19):** semester-1 numbers were recovered in EXP-000. They are **not a valid baseline**: the image-level split leaks across patients, and the test directory is contaminated. A leak-free, patient-wise baseline must be established first (planned in Phase 6). Until then, the `clean50` rows are the best available reference, with their caveats.

| EXP | Date | Model / change | Test set | Seg Dice | Seg IoU | Cls acc [95 % CI] | Cls macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 U-Net (5 ep) | pilot test clean50 | 0.411 | 0.309 | — | — | image-level split; possible patient leakage |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, full image | pilot test clean50 | — | — | 0.800 [0.670, 0.888] | 0.814 | dir110 (contaminated) = 0.909 |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, ROI from GT mask | pilot test clean50 | — | — | 0.740 [0.604, 0.841] | 0.746 | upper bound; needs GT at test time |
| [EXP-000](EXP-000-semester1-checkpoint-eval.md) | 2026-11-19 | Semester-1 SimpleCNN, ROI from predicted mask | pilot test clean50 | — | — | 0.460 [0.330, 0.596] | 0.456 | majority-class rate = 0.44 |
