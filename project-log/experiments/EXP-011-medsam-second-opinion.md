# EXP-011 — MedSAM second opinion as a routing signal (R2, M3)

- **Status:** pre-specified 2026-11-21 00:58, **before any code or result** (except the preprocessing check below). This file is committed first.
- **Branch:** `research/m3-r2-medsam`, from `main`.
- **Decision:** Kareem — "R2 later at night when I will sleep", "go r2" (2026-11-21 ~00:50). Run unattended.
- **Why:** M1 showed the router's bottleneck is its signal. `ens:disagree` detects "crop wrong, full right" only weakly (val AUROC 0.625), while the oracle reaches 0.974. EXP-010's attribution signal was weaker still (0.587). R2 tests a **different kind of segmenter**, MedSAM (a promptable foundation model), as a second opinion on the U-Net mask. Idea-matrix R2: "not found for classification gating"; risk: SAM-type models are weak with poor boxes.

## Model [V]
- `wanglab/medsam-vit-base` (Hugging Face, transformers `SamModel`, Apache-2.0, ungated, 93.7 M parameters). MedSAM (Ma et al. 2024), a fine-tuned SAM ViT-B (Kirillov et al. 2023, arXiv:2304.02643).
- Inference only, fp16 autocast. Measured 175 ms per image (embedding + decoding) and 1.4 GB peak VRAM on the RTX 3050.
- `transformers==5.23.0` added; `regex` pinned to 2026.9.29 (the newest release was 1 day old).

## Preprocessing check (done before this file, **val only**, ground-truth boxes) [V]
- 120 val slices of fold 0 (val fold 1), box = GT bounding box, Dice of MedSAM vs GT:
  - (A) min-max to [0, 1], no normalisation (original MedSAM script): mean 0.718;
  - (B) the HF port's preprocessor config (ImageNet mean/std after [0, 1] scaling): **mean 0.818**.
- **(B) is used.** No test data was involved.

## Pre-specified method (fixed now)
- **Input:** the 256² slice, cubic-resized to 1024², min-max scaled to [0, 1], 3 channels, ImageNet-normalised.
- **Output:** the low-resolution mask logits (256²), sigmoid > 0.5. This equals our image grid.
- **Prompt:** the bounding box of the fold's U-Net reference mask M (EXP-002c, threshold 0.5 + largest component), in 1024 coordinates [x0, y0, x1, y1] (end-exclusive × 4).
- **Jitter:** 8 extra boxes; each coordinate is moved by U(−0.1, 0.1) × the box width (x) or height (y), then clipped to [0, 1024]. RNG `default_rng([0, row, fold])` (external: `[1, ext_row, fold]`; shift: `[2, row, fold, kind, sev]`).
- **Candidate signals** (higher = less trustworthy):
  1. `sam:disagree` = 1 − Dice(M, S0), where S0 is MedSAM's mask with the original box;
  2. `sam:jitter` = 1 − mean_j Dice(M, S_j) over the 8 jittered boxes;
  3. `sam:selfvar` = 1 − mean pairwise Dice among S0..S8 (MedSAM's own instability).
- Empty M → no box → the signals are NaN → treated as the most suspicious value (as in EXP-005/006).
- **A. Descriptive (test, pooled):** MedSAM-with-U-Net-box Dice vs GT, against the U-Net's own Dice (EXP-002c 0.786).
- **B. Segmentation-failure detection (test):** AUROC for Dice_lcc < 0.7 of each SAM signal vs `ens:disagree` (EXP-005: 0.899), with patient-bootstrap CIs and paired differences.
- **C. Routing (in-distribution):**
  - The signal is chosen among the 3 on **pooled val** by AUROC for "crop32 wrong AND full right" (EXP-006 criterion).
  - Router: `policy_eval.fit_params` with that score. τ1 is fitted per fold on that fold's val (the frozen EXP-006 rule).
  - Test: balanced accuracy and AURC vs crop32+MSP, full+MSP and router(ens:disagree); paired patient bootstrap, 2,000 resamples.
- **D. Shift and external, with the signal frozen from C:**
  - Policy parameters fitted on clean val.
  - **EXP-008 conditions:** the corrupted test images are regenerated with the same RNG as EXP-008; [check] their U-Net Dice must equal the stored EXP-008 `dice_lcc`.
    - Per condition: router(sam) vs router(ens), crop32 and full.
    - Worst-case and mean over the 13 conditions, with the paired bootstrap (same resample in all conditions, as EXP-006b (d)).
  - **EXP-009 external:** per fold model set, image-level bootstrap.
- **No other signal, rule or combination is tried.**

## Run safety (GPU reliability note, 2026-11-20)
- Outputs are written per (mode, condition, fold) unit; finished units are skipped.
- A wrapper retries up to 6 times with a 90 s pause after any failure; `set -o pipefail`; the log is checked for Tracebacks.
- Image embeddings are not stored (23 GB free disk).

## Results
*(to be added after the run)*
