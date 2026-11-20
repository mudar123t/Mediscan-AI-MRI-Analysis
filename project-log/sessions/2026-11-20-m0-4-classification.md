# Session 2026-11-19/20 — M0.4 classification baseline and views (EXP-003)

## Goal
M0.4 of the approved plan. Kareem said to start it with no additions to the plan, and accepted 4–7 h of training.

## Actions
- Branch `exp/m0-classification` created from `main` (`1f9377c`).
- `18e6fe1`, library code:
  - `views.py`: full, masked_roi and context_crop(margin).
  - `models/cls.py`: SimpleCNN, ResNet-18 and EfficientNet-B0 through timm, `in_chans=1`.
  - `metrics/cls.py`.
  - `calibration.py`: temperature scaling.
  - `stats.confusion_bootstrap`: paired patient bootstrap on confusion matrices.
  - `data.datasets.load_predicted_masks`.
  - `utils/optim.py`: moved out of `train_seg.py` with unchanged behaviour.
  - Tests: 37 pass.
- `7cf0f70`: `scripts/train_cls.py`, `scripts/eval_cls.py` and 8 configs.
- The semester-1 recipe for the SimpleCNN baseline was read from `legacy/semester1/scripts/09_train_classifier.py`: 10 epochs, batch 8, Adam at 1e-3, best val accuracy.
- Timing probes on fold 0, 2 epochs (`runs/probe_exp003*`): ResNet-18 ~9 s/epoch, EfficientNet-B0 ~10 s/epoch, SimpleCNN ~4 s/epoch.
- `eval_cls.py` was smoke-tested on copies of the probe outputs in the scratchpad.
- Training: `runs/run_exp003.sh` ran 40 runs from 00:01 to 04:08 (4.1 h).
- Aggregation: `eval_cls.py` → `experiments/EXP-003-classification/results.json`.

## Findings [V: results.json]
- Fair semester-1 baseline: balanced accuracy 0.733 [0.703, 0.765], accuracy 0.761.
- ResNet-18 on the full image: 0.928 (+0.195). EfficientNet-B0 is equivalent.
- Balanced accuracy rises steadily with crop margin. The masked ROI is worse than the full image (−0.059 with GT masks, −0.079 with predicted).
- Best fully automatic pipeline (crop 32, predicted masks): 0.941 [0.916, 0.961], accuracy 0.948. That is above Cheng 2015's 91.28 %.
- Crop 32 beats the full image with GT masks (+0.026, p 0.005), but not with predicted masks (+0.013, CI crosses 0). This motivates R1.
- Temperature scaling lowers ECE everywhere.

## Open questions (M0.4 STOP)
- Segmenter: accept EXP-002c (Dice 0.786) for M1, or tune it?
- Should the classifier for M1 be crop 32? The context trend has not plateaued yet. EXP-004 already covers margin × mask quality.
- Push `exp/m0-classification` and update `main`?
