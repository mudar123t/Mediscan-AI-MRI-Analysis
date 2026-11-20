# EXP-005 — Which segmentation-uncertainty signal predicts failure? (M1.2)

- **Date:** 2026-11-20.
  - Seed segmenters trained 08:52 → 16:04, with a crash and resume (see below).
  - det/mc/tta scores took about 30 min and ens scores about 1 min, since they reuse the stored probability maps.
- **Branch / code:** `exp/m1-uncertainty`.
  - `src/mediscan/uncertainty.py`, `scripts/run_exp005.py`, `scripts/eval_exp005.py`.
  - Seed configs: `configs/exp005_unet_r34_s{1,2}.yaml`.
  - Test: `tests/test_uncertainty.py`.
- **Config:** `configs/exp005_uncertainty.yaml`. Raw output in `runs/exp005_uncertainty/scores_fold{k}_{det,mc,tta,ens}.csv` (git-ignored).
- **Segmenter:** EXP-002c (ResNet-34 U-Net, seed 0). The ensemble adds two more seeds with the same recipe: `exp002c_unet_r34_s1` and `_s2`.
- **Hardware:** RTX 3050 Laptop, 4 GB.
- **Results:** [EXP-005-uncertainty/auroc_table.csv](EXP-005-uncertainty/auroc_table.csv), [results.json](EXP-005-uncertainty/results.json). Pooled test set: 3,064 slices, 233 patients. CIs come from 2,000 patient-bootstrap resamples.

## Question and hypothesis (plan §3.3)
Which image-level uncertainty score predicts:
- (a) segmentation failure (Dice of the largest component < 0.7), and
- (b) downstream classification error?

Hypothesis: at least one segmentation-uncertainty score predicts classification error better than chance **and adds to MSP** (MSP baseline from Jaeger 2023 [V]).

## Design
- **Methods:**
  - `det`: one deterministic pass.
  - `mc`: MC dropout with T = 20 (the decoder's Dropout2d 0.2 kept active).
  - `tta`: the 8 dihedral transforms, each inverted back.
  - `ens`: the 3 seeds.
- **Scores per image:**
  - `ent_total`: mean pixel entropy.
  - `ent_ring`: mean entropy of the mean probability over the reference mask M dilated by 5 px, i.e. the tumor plus its border. If M is empty, pixels with p > 0.1 are used instead.
  - `maxprob_ring`: mean max(p, 1−p) over the same region, negated so higher = less trustworthy.
  - `disagree`: 1 − mean Dice between each sample's binary mask and M.
  - `area_cv`: coefficient of variation of the predicted area.
  - Empty reference mask → the most suspicious value.
- **Baselines:** classifier MSP (crop 32 and full), and predicted mask area (small = suspicious).
- **"Adds to MSP":**
  - Per fold, a logistic regression is fitted on that fold's **val** slices with features [−log(1−MSP)] or [−log(1−MSP), score], then applied to the fold's test slices.
  - The criterion is the mean of the 5 within-fold test AUROCs, compared with a paired bootstrap.
  - *Fix logged 2026-11-20:* the first version pooled the 5 per-fold regression outputs into one AUROC. This mixed differently scaled models: pooled 0.746 against 0.85–0.97 per fold. The criterion was changed to the within-fold mean, and MSP was log-transformed. It was fixed before any conclusion was drawn.

## Results [V: auroc_table.csv, results.json]
Positives (test): seg failure 616, crop-32 error 159, full-image error 191, out of 3,064.

| Score | Spearman vs −Dice | AUROC seg failure | AUROC crop-32 error | AUROC full error |
|---|---|---|---|---|
| det:ent_ring | 0.707 | 0.844 [0.818, 0.870] | 0.460 | 0.449 |
| mc:maxprob_ring | 0.712 | 0.846 [0.819, 0.872] | 0.468 | 0.457 |
| tta:disagree | 0.699 | 0.849 [0.824, 0.874] | 0.493 | 0.443 |
| ens:ent_ring | 0.756 | 0.881 [0.859, 0.902] | 0.497 | 0.449 |
| **ens:disagree** | **0.773** | **0.899 [0.878, 0.918]** | 0.525 [0.432, 0.632] | 0.470 |
| mask area (small) | 0.055 | 0.504 | 0.596 | 0.579 |
| **MSP crop 32 (classifier)** | −0.026 | 0.520 | **0.897 [0.859, 0.935]** | 0.802 |
| MSP full (classifier) | −0.080 | 0.496 | 0.731 | **0.871** |

Adds to MSP (crop-32 error; mean within-fold AUROC): MSP only **0.909 [0.883, 0.944]**. MSP + any score minus MSP: between −0.004 and +0.000. Every CI includes 0; the best is `ens:disagree` at −0.001 [−0.011, +0.020].

## Conclusions
- **Segmentation failure is well detected.** The seed ensemble is best (`ens:disagree` AUROC 0.899). MC dropout and TTA are close behind (0.84–0.85). This part is supported.
- **Classification error is not predicted by segmentation uncertainty** (AUROC 0.43–0.53, near chance), and nothing adds to MSP. **The hypothesis is not supported.**
  - Reason (from EXP-004): most crop-32 errors happen on *good* masks, where the full view also errs. The broken-mask slices are few: 224 have Dice < 0.3.
  - MSP itself is blind to segmentation failure (AUROC 0.52). So the two signals are complementary in what they detect, but segmentation failure is too rare to move the error ranking.
- Implication for EXP-006: the router's score must target "crop wrong *and* full right", not any crop error.

## Seed-run notes
- **Val Dice per fold** (seed 0 / 1 / 2) [V: `runs/exp002c_unet_r34*/fold*/summary.json`]:
  - fold 0: 0.799 / 0.793 / 0.800
  - fold 1: 0.811 / 0.812 / 0.812
  - fold 2: 0.770 / 0.735 / 0.779
  - fold 3: 0.795 / 0.789 / 0.800
  - fold 4: 0.790 / 0.807 / 0.806
- **Crash and resume.**
  - Seed 2 fold 2 crashed at epoch 26 (about 13:20) with `torch.AcceleratorError: CUDA error: unknown error` during a host-to-device copy. The GPU was healthy afterwards [I: transient driver fault].
  - The run script's log filter hid the failure, which was fixed with `set -eo pipefail`.
  - The fold was restarted from epoch 1 at 14:02.
- **Not bit-exact.** Seeded training is not bit-exact on this GPU: fold 2 epoch-1 loss was 0.7709 vs 0.7684 across the two attempts [V: `runs/exp005_seeds.log`].
