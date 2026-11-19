# EXP-000 — Re-evaluation of the semester-1 checkpoints (inference only)

- **Date:** 2026-11-19
- **Status:** done
- **Type:** audit / baseline recovery (no training)
- **Script:** [EXP-000-semester1-checkpoint-eval/eval_semester1.py](EXP-000-semester1-checkpoint-eval/eval_semester1.py)
- **Raw results:** [EXP-000-semester1-checkpoint-eval/results.json](EXP-000-semester1-checkpoint-eval/results.json)
- **Commit of code evaluated:** `390d7a0` (tag `v1.0-semester1`); checkpoints from `outputs/pilot/checkpoints/` (mtime 2025-12-23 04:40–04:50, local only, not in git).

## Hypothesis
Semester 1 recorded no metric values (see [00-project-understanding.md §6](../00-project-understanding.md)). The val/test directories contain 60 training images each (§8.1). Hypothesis: the numbers semester-1 scripts would print are inflated by these leaked images, and the clean 50-image subsets give substantially lower values.

## Config
- Models: `UNet()` (7.76 M params), 3 × `SimpleCNN()` (8.41 M params), loaded with `strict=True`.
- Segmentation metric: identical to `src/mediscan/segmentation/metrics.py` — per-image Dice and IoU at probability threshold 0.5, eps 1e-6, averaged over images.
- Classification: argmax; accuracy, macro-F1, per-class recall, confusion matrix (sklearn 1.4.2).
- Subsets of each split directory: `dir110` = all 110 files (what semester-1 scripts used), `clean50` = files in `splits/<split>.txt`, `leaked60` = the other 60 (all also in `train/`).
- Seed: not applicable (deterministic inference). Hardware: CPU only (torch 2.9.1+cpu, Python 3.12.10, Windows 11). Runtime 69.3 s.

## Results

### Segmentation (U-Net, threshold 0.5)
| Subset | n | Dice mean ± std | Dice median | IoU mean | % images Dice < 0.5 |
|---|---|---|---|---|---|
| test / dir110 | 110 | 0.364 ± 0.302 | 0.380 | 0.267 | 56.4 % |
| **test / clean50** | 50 | **0.411 ± 0.317** | 0.507 | **0.309** | 48.0 % |
| test / leaked60 | 60 | 0.326 ± 0.284 | 0.327 | 0.231 | 63.3 % |
| val / dir110 | 110 | 0.334 ± 0.306 | 0.295 | 0.244 | 59.1 % |
| val / clean50 | 50 | 0.312 ± 0.292 | 0.222 | 0.224 | 64.0 % |

### Classification (SimpleCNN), test split
| Input | dir110 acc (what semester 1 would report) | **clean50 acc [95 % Wilson CI]** | clean50 macro-F1 | leaked60 acc | clean50 recall G / M / P |
|---|---|---|---|---|---|
| full image | 0.909 | **0.800 [0.670, 0.888]** | 0.814 | 1.000 | 0.76 / 0.77 / 0.91 |
| ROI from GT mask (upper bound, needs GT at test time) | 0.873 | **0.740 [0.604, 0.841]** | 0.746 | 0.983 | 0.82 / 0.64 / 0.82 |
| ROI from predicted mask (the real pipeline) | 0.709 | **0.460 [0.330, 0.596]** | 0.456 | 0.917 | 0.76 / 0.23 / 0.45 |

Majority-class baseline on clean50 test = 22/50 = 0.44 (meningioma). Confusion matrices are in `results.json`.

## Comparison to baseline
This **is** the baseline recovery. No earlier numbers exist to compare against.

## Conclusions
1. **Leakage inflated classification by 6–25 points.** The full-image CNN scores 1.000 on the 60 leaked images, compared with 0.800 on the clean 50. The figure semester-1 scripts would print (0.909) is not a valid test result. [VERIFIED]
2. **The U-Net is weak:** clean test Dice is 0.41, and about half the images have Dice < 0.5. It gets lower Dice on leaked training images (0.33) than on clean ones, which suggests the model is **underfit** (only 5 epochs), not memorising. [VERIFIED numbers; INFERRED cause]
3. **The core semester-1 claim fails on clean data.** The fully automated pipeline (ROI from predicted mask) scores 0.46, roughly the majority-class rate (0.44). The plain full-image classifier is far better (0.80). Even ROI from perfect GT masks (0.74) does not beat the full image. Possible reasons: masking out everything except the tumor removes context, and the U-Net is weak. [VERIFIED numbers; INFERRED cause]
4. With n = 50, the confidence intervals are about ±0.11, so the full and GT-ROI results are statistically indistinguishable. The semester-2 baseline needs a larger, patient-wise test set.
5. Every clean50 number may still be optimistic because of patient-level leakage (§8.2 of the audit).

**Thesis value:** this is a strong "honest evaluation" finding. It motivates patient-wise evaluation and a redesign of how segmentation helps classification.
