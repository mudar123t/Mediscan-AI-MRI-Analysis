# EXP-002 — Segmentation baseline: recipe and encoder (M0.3)

- **Date:** 2026-11-19 (training 15:03 → 21:09 +03:00, about 6.1 h wall clock, sequential)
- **Branch / code:** `exp/m0-segmentation`. Code: `scripts/train_seg.py`, `scripts/eval_seg.py`, `src/mediscan/models/seg.py`, `src/mediscan/metrics/seg.py`, `src/mediscan/stats.py`.
- **Configs:** `configs/exp002a_unet_s1_old.yaml`, `exp002b_unet_s1_new.yaml`, `exp002c_unet_r34.yaml`. Each config is copied, with `environment.json`, into `runs/<run>/fold<k>/` (git-ignored).
- **Split:** `data/splits/figshare_fold{0..4}.json` from EXP-001. Fold k is the test set, fold (k+1) mod 5 the validation set, and the other 3 folds the training set. Every slice is tested exactly once, by a model that never saw its patient.
- **Seed:** 0, deterministic cuDNN, a single seed per fold. Seeds 1 and 2 were not run (see Limitations).
- **Hardware:** RTX 3050 Laptop with 4 GB, `.venv`, torch 2.9.1+cu128 (environment.md).
- **Raw results:** [EXP-002-segmentation/results.json](EXP-002-segmentation/results.json), written by `scripts/eval_seg.py` with 2,000 patient-bootstrap resamples.

## Question and hypotheses
How much of semester 1's poor segmentation (Dice 0.411 in EXP-000) comes from the training recipe, and how much from the architecture?
- **H1:** the same semester-1 U-Net with a modern recipe beats the old recipe on the same patient-disjoint folds.
- **H2:** an ImageNet-pretrained ResNet-34 encoder beats H1's model.
- **Planned target (plan §EXP-002):** Dice ≈ 0.85–0.93, based on Avazov 2024 [V], who report 0.93 on Figshare with a single patient 80/20 split.

## Arms
| Arm | Model | Recipe |
|---|---|---|
| a | semester-1 `UNet` (7.76 M params) | **Old recipe, the fair semester-1 baseline:** 5 epochs, batch 4, Adam at lr 1e-3, no augmentation, fp32, BCE+Dice (0.5), last-best of 5 epochs on val Dice |
| b | same `UNet` | **New recipe:** up to 100 epochs, batch 8, AdamW at lr 1e-3 with weight decay 1e-4, cosine schedule with 2 warmup epochs, augmentation (flip, affine, elastic, brightness/contrast, gamma), AMP, early stopping with patience 15 on val Dice |
| c | smp U-Net, ResNet-34 ImageNet encoder, Dropout2d(0.2) after each decoder block | New recipe with lr 3e-4 |

All arms:
- Input 1×256×256 in [0,1].
- Threshold 0.5.
- Model selection on hard Dice of the **validation** fold only. The test fold is evaluated once, with the selected weights.

## Metrics
- Per slice: Dice, IoU, HD95 in pixels, and NSD at 2 px.
- HD95 is in pixels because the dataset has no pixel spacing (EXP-001).
- Empty prediction rule: Dice 0 and HD95 equal to the image diagonal (362 px).
- Pooled over the 5 test folds (3,064 slices, 233 patients).
- 95 % CIs from the patient-level bootstrap.
- Paired differences use a paired patient bootstrap on the same slices. p = 0.000 means none of the 2,000 resamples crossed zero, so it is reported as p < 0.001.

## Results [V: results.json]
### Pooled test (3,064 slices / 233 patients)
| Arm | Dice [95 % CI] | IoU | HD95 px | NSD@2px | Patient-mean Dice | Slices with Dice < 0.5 | Empty preds | Fold Dice mean ± sd |
|---|---|---|---|---|---|---|---|---|
| a, old recipe | 0.508 [0.475, 0.541] | 0.402 | 56.8 [51.2, 62.7] | 0.338 | 0.504 | 43.6 % | 6.43 % | 0.508 ± 0.031 |
| b, new recipe | 0.759 [0.735, 0.782] | 0.668 | 25.0 [20.8, 29.8] | 0.641 | 0.756 | 13.7 % | 3.10 % | 0.760 ± 0.026 |
| **c, ResNet-34 U-Net** | **0.786 [0.766, 0.806]** | **0.695** | **14.4 [12.4, 16.4]** | **0.665** | **0.781** | **11.2 %** | **0.13 %** | 0.787 ± 0.029 |

Per-fold test Dice:
- a: 0.524 / 0.492 / 0.461 / 0.538 / 0.524
- b: 0.769 / 0.746 / 0.798 / 0.730 / 0.755
- c: 0.818 / 0.799 / 0.806 / 0.754 / 0.758

### Paired comparisons (same slices, patient bootstrap)
| Comparison | ΔDice [95 % CI] | ΔHD95 px [95 % CI] | p |
|---|---|---|---|
| b − a (recipe) | **+0.250** [+0.228, +0.273] | −31.9 [−37.7, −26.5] | < 0.001 |
| c − a (recipe + encoder) | **+0.278** [+0.255, +0.302] | −42.4 [−47.8, −37.7] | < 0.001 |
| c − b (encoder) | **+0.027** [+0.019, +0.037] | −10.6 [−14.1, −7.4] | < 0.001 |

### Per class (Dice [95 % CI], HD95 px)
| Class (patients) | a | b | c |
|---|---|---|---|
| Glioma (89) | 0.363 [0.325, 0.401], 72.2 | 0.648 [0.612, 0.684], 38.9 | **0.685 [0.654, 0.714], 23.1** |
| Meningioma (82) | 0.690 [0.651, 0.726], 41.6 | 0.884 [0.860, 0.903], 15.6 | **0.893 [0.869, 0.911], 8.3** |
| Pituitary (62) | 0.593 [0.541, 0.642], 44.8 | 0.833 [0.810, 0.853], 10.8 | **0.858 [0.842, 0.873], 5.7** |

### Largest-connected-component post-processing (ablation)
- LCC changes Dice by ≤ 0.001 in every arm: c is 0.786 raw vs 0.785 LCC.
- LCC lowers HD95: c goes from 14.4 to 11.7 px [10.2, 13.3]; b from 25.0 to 21.9; a from 56.8 to 45.9.
- [I] Spurious small blobs far from the tumor drive HD95 but not Dice. LCC helps boundary metrics only.

### Training cost [V: summary.json, history.csv]
| Arm | Epochs run per fold | Best epoch | Median epoch time (fold 0) | Train time per fold | Peak VRAM |
|---|---|---|---|---|---|
| a | 5 | 3–5 | 50.7 s (fp32, batch 4) | 253–266 s | 914–922 MiB |
| b | 61–87 | 46–72 | 33.5 s | 2,044–2,902 s | 898–900 MiB |
| c | 29–81 | 14–66 | 30.7 s (folds 0–3: 29.9–30.9 s) | 1,129–2,500 s | 990–1,087 MiB |

Total training time across the 15 folds was about 6.1 h. Every arm fits in about a quarter of the 4 GB GPU.

## Conclusions
1. **H1 is supported.** The semester-1 U-Net was held back by its recipe, not its architecture: changing only the recipe adds +0.250 Dice [0.228, 0.273] and removes 32 px of HD95.
2. **H2 is supported, but the effect is small.** The pretrained encoder adds +0.027 Dice [0.019, 0.037]. It matters much more for boundary quality: HD95 drops by 10.6 px, the share of empty predictions falls from 3.1 % to 0.13 %, and per-fold behaviour is more stable.
3. **The fair semester-1 baseline is Dice 0.508 [0.475, 0.541].** It differs from the 0.411 in EXP-000, which was measured with a different protocol: the semester-1 checkpoint, a transposed pilot subset and a 50-image test set. The two numbers are not comparable; [I] the gap is not attributable to one cause. All semester-2 comparisons use 0.508.
4. **Glioma is the hard class** in every arm (c: 0.685, versus 0.893 for meningioma). [I] This is consistent with glioma's diffuse, irregular borders. It matters for M1: R1 should route more glioma cases to "review", and that has to be checked per class.
5. **The planned target of 0.85–0.93 was not reached.** Arm c reaches 0.786 pooled; on folds 0–2 it reaches 0.80–0.82.
   - [I] Possible reasons, none tested yet:
     - Avazov 2024 uses one 80/20 split, not 5-fold CV over all 233 patients.
     - Their Dice may be averaged differently (per image vs per dataset).
     - Our models are trained at 256² and are not tuned.
   - [U] Whether we can close the gap at 256² on 4 GB.
   - Options are listed under Next steps; nothing was changed after seeing the test results.
6. **Arm c is the segmenter for M1.** It is best on every metric, has decoder dropout for MC-dropout uncertainty, and almost never predicts empty masks.

## Limitations
- Only one seed (0). The plan's seeds 1 and 2 were not run, so fold-to-fold sd (0.029 for c) is the only variability measure apart from the patient bootstrap.
- No hyperparameter search. lr and patience were set once, before training.
- Fold 4 of arm c stopped early: best epoch 14, 29 epochs run, median epoch 39.6 s. Its test Dice is 0.758. [I] An earlier stop may cost a little on that fold.
- HD95 is in pixels of the 256² grid, not mm.
- The 0.5 threshold was not tuned.

## Next steps (for discussion at the M0.4 STOP)
- **Option 1:** accept arm c (0.786) as the M1 segmenter. R1 needs *variable* segmentation quality, and 11 % of slices below Dice 0.5 gives the router something to detect.
- **Option 2:** spend 1–2 runs on improvements, chosen on the validation fold only. Candidates:
  - 2 more seeds, needed for the plan anyway.
  - Dice-focused loss weights.
  - 384² input; the 4 GB budget allows it.
  - A stronger encoder.
  - Threshold tuned on val.

## Artifacts
- `runs/exp002{a_unet_s1_old,b_unet_s1_new,c_unet_r34}/fold{0..4}/`: best.pt, history.csv, pred_{val,test}.npz, metrics_{val,test}.csv, summary.json, config.yaml, environment.json (git-ignored, local).
- `runs/probe_r34/`: a 2-epoch timing probe. It is not a result.
