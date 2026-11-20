# Session 2026-11-20 — M1: mask quality, uncertainty, routing, shift and external test (EXP-004 … EXP-009)

## Goal
M1.1–M1.4 of the approved plan (Kareem, 2026-11-20):
- push M0.4;
- keep the EXP-002c segmenter (no tuning);
- start with the seed run;
- do M1.4 without asking.

STOP after M1.4 with the H-B1..H-B3 report.

## Actions (branch `exp/m1-uncertainty`, includes `exp/m1-r3-seg-helps`)
- `b7ac0ee`: seed configs. Seed run `runs/run_exp005_seeds.sh`, 08:52 → 16:04.
  - Seed 2 fold 2 crashed with a transient CUDA "unknown error". The log filter had masked the exit code; this was fixed with `pipefail`. The run was resumed at 14:02.
- `8fab45b`: EXP-004 degradation study (`degrade.py`, `run_exp004.py`, `eval_exp004.py`).
- `54e8fbc`: EXP-005 sampling and scores (`uncertainty.py`, `run_exp005.py`).
- `3f6187e`: `metrics/selective.py`, `router.py`, `eval_exp005.py`, `eval_exp006.py`, and `prepare_bdneuro.py` with the duplicate audit.
- `9e8b014`: `policy_eval.py`, `corrupt.py`, `pipeline.py`, `run_shift_external.py`. The pipeline was verified against the stored EXP-003 logits (max |Δ| 0.008).
- `6f149b8`, two fixes:
  - EXP-005 "adds to MSP" pooling artefact → mean within-fold AUROC with a −log(1−MSP) feature.
  - The router's score target changed to "crop wrong, full right". This was set after the EXP-004/005 analyses and before EXP-006 ran.
- `cfe8429`: `eval_shift_external.py`.
- Preliminary EXP-006 and EXP-008/009 used det/mc/tta.
- The final EXP-006 chose `ens:disagree`, so EXP-008/009 inference was re-run with det + ens (16:07–16:46). The preliminary outputs are kept in `runs/*_prelim_tta/`.
- Logs: EXP-004, EXP-005, EXP-006 (with EXP-007), EXP-008 and EXP-009.

## Findings
1. **EXP-004.**
   - A 32 px context margin makes classification almost insensitive to eroded, dilated or shifted masks (≥ 0.93 even at Dice 0.3).
   - False-positive blobs break every crop.
   - Real masks with Dice < 0.3 (224 slices) cost the crop 0.248 against the full image.
2. **EXP-005.**
   - The seed ensemble detects segmentation failure best (`ens:disagree` AUROC 0.899).
   - No segmentation-uncertainty score predicts classification error or adds to MSP (MSP alone 0.909).
3. **EXP-006/007.**
   - Router 0.935 vs crop32 0.941 (n.s.).
   - Router better than the full view on AURC (−0.0115) and AUGRC.
   - τ1 is unstable across folds.
   - Oracle 0.974.
   - CP marginal coverage is OK, but meningioma coverage is 0.75 at α 0.1.
4. **EXP-008.**
   - The routing gain over the crop grows with noise and blur severity (+0.072 / +0.145 at s3).
   - Under gamma the router protects against the full view's collapse (+0.09).
   - The router has the best worst case (0.607) and the best mean AURC over the 13 conditions.
5. **EXP-009.**
   - 1,682 of 5,154 BDNeuro tumor images duplicate Figshare slices; 33 % of BDNeuro's own test split.
   - On the 3,030 clean images: crop32 0.931, router 0.922, full 0.904.
   - The router's result follows each fold's τ1, not per-image decisions.

## Outcome
M1 experiments are complete. The STOP report on H-B1..H-B3 goes to Kareem. Nothing is pushed yet.
