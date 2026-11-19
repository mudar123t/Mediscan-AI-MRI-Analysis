# Session 2026-11-19 — M0.3 segmentation baseline (EXP-002)

## Goal
M0.3 of the approved plan: a fair semester-1 segmentation baseline on the patient-disjoint folds, plus an improved segmenter for M1.

## Actions
- Code committed on branch `exp/m0-segmentation`:
  - `abc5efd`: model factory (`unet_s1`, `unet_r34` with decoder Dropout2d for MC dropout), segmentation metrics (Dice, IoU, HD95 in px, NSD@2px, empty-prediction rule, LCC) and patient bootstrap.
  - `6145871`: `scripts/train_seg.py` and the configs for arms a, b and c.
  - `813903a`: `scripts/eval_seg.py` and a vectorised `mean_ci`.
- Tests: 28 pass.
- Probe: fold 0, 2 epochs, r34 → 26 s/epoch, 987 MiB. The probe is not a result; it lives in `runs/probe_r34/`.
- Training: `runs/run_exp002.sh` ran a → b → c over 5 folds each, from 15:03 to 21:09 (about 6.1 h). All 15 `summary.json` files were present at the end.
- Aggregation: `scripts/eval_seg.py` → `experiments/EXP-002-segmentation/results.json`.

## Findings [V: results.json]
- Fair semester-1 baseline (a): Dice 0.508 [0.475, 0.541].
- Same U-Net with the new recipe (b): Dice 0.759; paired Δ +0.250 [0.228, 0.273].
- ResNet-34 U-Net (c): Dice 0.786 [0.766, 0.806] and HD95 14.4 px; Δ vs b +0.027 [0.019, 0.037].
- Glioma is the hardest class (c: 0.685).
- LCC helps HD95 only.
- The planned target of 0.85–0.93 was **not** reached. Reasons are inferred, not tested; see EXP-002 Conclusions 5.

## Outcomes
- Arm c is the candidate M1 segmenter. Whether to tune it further is decided at the M0.4 STOP.
- Logs updated: EXP-002, results-summary, thesis-map, presentation-bank, environment, HANDOFF and CHANGELOG.

## Open questions (for the M0.4 STOP)
- Accept 0.786, or spend 1–2 validation-only tuning runs (seeds, 384², loss, threshold)?
- Run seeds 1 and 2 for arm c now or later? About 2 h each [I, from measured times].
