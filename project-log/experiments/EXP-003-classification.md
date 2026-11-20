# EXP-003 — Classification baseline and input views (M0.4)

- **Date:** 2026-11-20. Training ran 00:01 → 04:08 +03:00 (4.1 h, sequential, 40 runs).
- **Branch / code:** `exp/m0-classification`. Code: `scripts/train_cls.py`, `scripts/eval_cls.py`, `src/mediscan/{views,calibration,stats}.py`, `models/cls.py`, `metrics/cls.py`.
- **Configs:** `configs/exp003a_simplecnn_old.yaml`, `configs/exp003_r18_{full,masked,crop0,crop8,crop16,crop32}.yaml`, `configs/exp003_effb0_full.yaml`. Each is copied into `runs/<run>/fold<k>/` (git-ignored).
- **Split:** the same patient-disjoint folds as EXP-002. Test fold k, val fold (k+1) mod 5, train on the other 3.
- **Seed:** 0, deterministic.
- **Hardware:** RTX 3050 Laptop, 4 GB.
- **Raw results:** [EXP-003-classification/results.json](EXP-003-classification/results.json). Pooled test over the 5 folds: 3,064 slices, 233 patients. 2,000 patient-bootstrap resamples; paired differences use the same resamples.

## Question and hypotheses
1. What is the fair semester-1 classification baseline on the leak-free protocol?
2. How much does a modern pretrained classifier add?
3. Which input view works best: the full image, the semester-1 masked ROI, or a context crop with a margin of 0–32 px?
4. How much is lost when the masks come from the segmenter (EXP-002c) instead of the ground truth?

Target from the plan: ≥ Cheng 2015, 91.28 % patient-wise [V]. Reference: Deepak 2019, 98 % [V].

## Design
- **Arm a, the fair semester-1 baseline.** Semester-1 `SimpleCNN` on the full image, with the semester-1 recipe from `legacy/semester1/scripts/09_train_classifier.py`: 10 epochs, batch 8, Adam at lr 1e-3, no augmentation, fp32, best val accuracy.
- **New recipe for the other arms.** ImageNet-pretrained timm models with `in_chans=1`. Up to 60 epochs, batch 32, AdamW at lr 3e-4 with weight decay 1e-4, cosine schedule with 2 warmup epochs, EXP-002 augmentation, AMP. Early stopping with patience 15 on val balanced accuracy; ties are broken by val NLL.
  - ResNet-18 with 6 views: `full`; `masked_roi` (semester-1 style, all non-tumor pixels set to 0); and `context_crop` with margins of 0, 8, 16 and 32 px. A crop is the square bounding box of the mask plus the margin, clipped to the image and resized to 256².
  - EfficientNet-B0 on the full view only.
- **Mask sources.** Training and model selection always use views built from ground-truth (GT) masks. Every mask view is then evaluated twice:
  - with GT masks, an upper bound that needs GT at test time;
  - with **predicted** masks from the EXP-002c segmenter of the same fold, thresholded at 0.5 with only the largest component kept. That segmenter never saw the test patients. This is the realistic, fully automatic pipeline.
  - 4 of the 3,064 test slices (0.13 %) get an empty predicted mask. There, the crop view falls back to the full image and the masked view is all black.
- **Calibration.** Temperature scaling, fitted per fold on that fold's val logits from the same mask source. It never sees the test fold.

## Results [V: results.json]
### Pooled test (3,064 slices / 233 patients)
| System (view : masks) | Bal. acc [95 % CI] | Acc | Macro-F1 | MCC | AUROC | ECE → after TS |
|---|---|---|---|---|---|---|
| **a: SimpleCNN, old recipe : full** | 0.733 [0.703, 0.765] | 0.761 | 0.731 | 0.625 | 0.888 | 0.174 → 0.038 |
| ResNet-18 : full | 0.928 [0.903, 0.948] | 0.938 | 0.928 | 0.902 | 0.986 | 0.022 → 0.016 |
| EfficientNet-B0 : full | 0.931 [0.905, 0.951] | 0.938 | 0.930 | 0.902 | 0.985 | 0.046 → 0.026 |
| ResNet-18 : masked ROI, GT | 0.870 [0.844, 0.894] | 0.880 | 0.871 | 0.811 | 0.969 | 0.050 → 0.015 |
| ResNet-18 : masked ROI, pred | 0.849 [0.820, 0.875] | 0.860 | 0.849 | 0.782 | 0.955 | 0.068 → 0.016 |
| ResNet-18 : crop 0 px, GT | 0.881 [0.856, 0.905] | 0.893 | 0.879 | 0.832 | 0.973 | 0.039 → 0.013 |
| ResNet-18 : crop 0 px, pred | 0.875 [0.848, 0.900] | 0.886 | 0.873 | 0.822 | 0.965 | 0.048 → 0.017 |
| ResNet-18 : crop 8 px, GT | 0.928 [0.905, 0.948] | 0.935 | 0.927 | 0.898 | 0.988 | 0.027 → 0.014 |
| ResNet-18 : crop 8 px, pred | 0.913 [0.888, 0.936] | 0.921 | 0.912 | 0.876 | 0.984 | 0.036 → 0.010 |
| ResNet-18 : crop 16 px, GT | 0.939 [0.916, 0.959] | 0.948 | 0.940 | 0.918 | 0.990 | 0.026 → 0.012 |
| ResNet-18 : crop 16 px, pred | 0.927 [0.901, 0.949] | 0.934 | 0.926 | 0.897 | 0.987 | 0.036 → 0.010 |
| **ResNet-18 : crop 32 px, GT** | **0.954 [0.931, 0.974]** | 0.961 | 0.956 | 0.940 | 0.994 | 0.022 → 0.009 |
| **ResNet-18 : crop 32 px, pred** | **0.941 [0.916, 0.961]** | 0.948 | 0.942 | 0.919 | 0.991 | 0.033 → 0.005 |

### Per-class recall (meningioma / glioma / pituitary)
- a, SimpleCNN: 0.477 / 0.792 / 0.931. Meningioma is mostly missed.
- ResNet-18 full: 0.860 / 0.955 / 0.970.
- ResNet-18 crop 32, pred: 0.897 / 0.966 / 0.960.
- Meningioma is the weakest class in every ResNet/EfficientNet arm.

### Paired differences in balanced accuracy (patient bootstrap)
| Comparison | Δ [95 % CI] | p |
|---|---|---|
| ResNet-18 full − SimpleCNN old (recipe + model) | **+0.195** [+0.165, +0.227] | < 0.001 |
| Crop 32 pred − SimpleCNN old (best automatic pipeline vs baseline) | **+0.207** [+0.172, +0.244] | < 0.001 |
| EfficientNet-B0 full − ResNet-18 full | +0.002 [−0.012, +0.016] | 0.74 |
| Masked ROI GT − full | −0.059 [−0.088, −0.030] | < 0.001 |
| Masked ROI pred − full | −0.079 [−0.110, −0.048] | < 0.001 |
| Crop 0 pred − full | −0.054 [−0.082, −0.026] | < 0.001 |
| Crop 8 pred − full | −0.015 [−0.040, +0.008] | 0.22 |
| Crop 16 pred − full | −0.001 [−0.022, +0.020] | 0.93 |
| **Crop 32 GT − full** | **+0.026** [+0.008, +0.043] | 0.005 |
| **Crop 32 pred − full** | +0.013 [−0.005, +0.029] | 0.16 |
| Crop 32 − crop 16 (GT / pred) | +0.015 [+0.002, +0.026] / +0.014 [+0.002, +0.026] | 0.016 / 0.020 |
| Pred − GT masks: masked / crop 0 / 8 / 16 / 32 | −0.021 / −0.006 / −0.015 / −0.012 / −0.013 | 0.005 / 0.35 / < 0.001 / 0.002 / 0.001 |

### Training cost [V: summary.json]
| Arm | Train time, 5 folds | Epochs run | Peak VRAM |
|---|---|---|---|
| SimpleCNN | 3.5 min | 10 | 643 MiB |
| ResNet-18 views | 23–41 min | 20–60 | 757–795 MiB |
| EfficientNet-B0 | 43 min | 31–53 | 1,915 MiB |

## Conclusions
1. **Fair semester-1 baseline: balanced accuracy 0.733 [0.703, 0.765], accuracy 0.761.** Meningioma recall is only 0.477. The EXP-000 clean50 figure of 0.80 came from a 50-image, leaky test set and is not comparable.
2. **A modern recipe with a pretrained model adds +0.195 balanced accuracy.** ResNet-18 and EfficientNet-B0 are equivalent (Δ +0.002, p = 0.74), so ResNet-18 stays: it is cheaper and uses less than half the VRAM.
3. **Context matters, and it rises steadily with margin.** Masked ROI (0.870) and crop 0 (0.881) < crop 8 (0.928) < crop 16 (0.939) < crop 32 (0.954), all with GT masks.
   - The semester-1 design, masked ROI, is significantly *worse* than the full image (−0.059 with GT masks, −0.079 with predicted masks).
   - This explains the semester-1 failure seen in EXP-000 and matches Cheng 2015's finding that a dilated ROI beats a tight one [V].
4. **The segmentation-guided view helps only when the mask is good.**
   - With GT masks, crop 32 beats the full image: +0.026 [+0.008, +0.043].
   - With predicted masks, the gain shrinks to +0.013 [−0.005, +0.029] and is no longer significant.
   - Predicted masks cost 0.6–2.1 points in every mask view.
   - [I] This is the direct motivation for R1, segmentation-quality-aware routing: use the crop when the mask is trustworthy, fall back to the full image (or refer) when it is not. EXP-004/006 will test this.
5. **The plan's target is met.** The best fully automatic pipeline (crop 32, predicted masks) reaches accuracy 0.948 and balanced accuracy 0.941 [0.916, 0.961]. ResNet-18 on the full image reaches 0.938. Both exceed Cheng 2015's 91.28 %.
   - [I] Cheng 2015 used 5-fold CV on this dataset. Whether their folds are exactly `cvind.mat` is [U]. Their number is accuracy, so it is compared with our accuracy.
   - Deepak 2019's 98 % is not reached.
6. **Calibration.** Temperature scaling, fitted on val, lowers ECE in every system. For crop 32 with predicted masks, ECE goes from 0.033 to 0.005. Calibrated confidences are ready for the M1 referral thresholds.

## Limitations
- One seed. No hyperparameter search; the recipe was set before training.
- Model selection used val views built from GT masks. The predicted-mask val logits were used only to fit the temperature.
  - The predicted masks for the val fold come from a segmenter that was itself selected on that fold (EXP-002). [I] This could make val slightly optimistic for temperature fitting; test is not affected.
- Crop 16, fold 4 hit the 60-epoch cap (best epoch 60), so it may be slightly under-trained. Every other run stopped early.
- The trend still rises at 32 px. Larger margins were not in the plan and were not run; this is noted for EXP-004, which already covers margin × mask quality.
- Slice-level metrics: a patient contributes several correlated slices, which the patient bootstrap accounts for.

## Artifacts
- `runs/exp003*/fold{0..4}/`: best.pt, history.csv, logits_{val,test}_{gt,pred}.npz, summary.json, config.yaml, environment.json (git-ignored).
- `runs/probe_exp003*`: 2-epoch timing probes, not results.
