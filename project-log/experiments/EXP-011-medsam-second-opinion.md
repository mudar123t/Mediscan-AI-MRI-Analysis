# EXP-011 — MedSAM second opinion as a routing signal (R2, M3)

- **Status:** pre-specified 2026-11-21 00:58, **before any code or result** (except the preprocessing check below), committed first (`a397cc2`). **Done 2026-11-21 ~05:55: H-R2 not supported** (see Results).
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

## Run [V `runs/exp011.log`]
- 2026-11-21 00:58 → 05:50 (+03:00), `bash scripts/run_exp011_retry.sh`, no failure and no retry needed (0 Tracebacks).
  - Clean: 5 folds, 326–529 s each. External: 3,030 images × 5 folds. Shift: 60 units (12 conditions × 5 folds), 159–291 s each.
- **Pre-specified check passed:** in all 60 shift units, the regenerated U-Net Dice equals the stored EXP-008 `dice_lcc` (gap ≤ 1e-6).
- External consistency [V]: the fold-mean balanced accuracy of full / crop32 / router(ens) is 0.904 / 0.931 / 0.922, identical to EXP-009.
- Analysis: `scripts/eval_exp011.py --parts ABC` (`runs/eval_exp011_abc.log`) and `--parts D` (`runs/eval_exp011_d.log`) → [results.json](EXP-011-medsam-second-opinion/results.json). Patient bootstrap, 2,000 resamples, paired.

## Results [V results.json]
### A. MedSAM with the U-Net box vs ground truth (test, pooled)
| | MedSAM | U-Net (EXP-002c) |
|---|---|---|
| All | 0.593 [0.566, 0.620] | 0.785 [0.764, 0.805] |
| Meningioma | 0.733 | 0.889 |
| Glioma | 0.451 | 0.687 |
| Pituitary | 0.704 | 0.855 |
- Difference −0.192 [−0.206, −0.178]. 4 test slices had an empty U-Net mask (no box).
- With GT boxes on val, MedSAM reached 0.818 (preprocessing check). So most of the drop comes from the imperfect U-Net box. This matches the known risk (SAM-type models are weak with poor boxes).

### B. Segmentation-failure detection (Dice_lcc < 0.7; 616 positives)
| Signal | AUROC | − ens:disagree |
|---|---|---|
| sam:disagree | 0.717 [0.686, 0.744] | −0.182 [−0.210, −0.156] |
| sam:jitter | 0.712 [0.681, 0.741] | −0.186 [−0.214, −0.160] |
| sam:selfvar | 0.673 [0.638, 0.704] | −0.226 [−0.260, −0.193] |
| ens:disagree (EXP-005) | 0.899 [0.878, 0.918] | — |

### C. Routing, in distribution
- **Signal choice on pooled val** (AUROC for "crop32 wrong and full right"): sam:disagree 0.543, sam:jitter 0.534, sam:selfvar 0.504 (ens:disagree 0.625). Chosen: **sam:disagree**.
- τ1 per fold: [0.243, 0.021, 0.152, 0.714, 0.404]. Unlike ens, no fold flips to always-full. Test crop share 0.435 (ens 0.405).

| Policy (test, 5-fold OOF) | Bal. acc | AURC |
|---|---|---|
| full + MSP | 0.928 | 0.0203 |
| crop32 + MSP | 0.941 | 0.0097 |
| router(ens) | 0.935 | 0.0089 |
| **router(sam)** | **0.9355** | **0.0128** |
| oracle router | 0.974 | 0.0068 |

- router(sam) − router(ens): bal. acc +0.0003 [−0.007, +0.008], AURC +0.004 [−0.001, +0.013].
- router(sam) − crop32: bal. acc −0.005 [−0.017, +0.009]. router(sam) − full: AURC −0.0075 [−0.020, −0.0003].

### D. Shift (13 conditions incl. clean) and external, signal and parameters frozen
**Worst case and mean over the 13 conditions** (same patient resample in every condition):

| Policy | Worst bal. acc | Mean bal. acc | Worst AURC | Mean AURC |
|---|---|---|---|---|
| full + MSP | 0.581 | 0.793 | 0.242 | 0.090 |
| crop32 + MSP | 0.535 | 0.801 | 0.324 | 0.090 |
| router(ens) | **0.607** [0.583, 0.630] | **0.813** | **0.192** | **0.072** |
| router(sam) | 0.594 [0.570, 0.618] | 0.809 | 0.210 | 0.074 |
| oracle router | 0.681 | 0.886 | 0.152 | 0.047 |

- router(sam) − router(ens): **worst bal. acc −0.0125 [−0.022, −0.003]**; mean −0.0035 [−0.011, +0.005]; worst AURC +0.017 [−0.016, +0.039]; mean AURC +0.003 [−0.003, +0.009].
- router(sam) − crop32: worst bal. acc +0.060 [+0.037, +0.098]; mean AURC −0.016 [−0.027, −0.004].
- router(sam) − full: worst bal. acc +0.013 [−0.012, +0.051] (n.s.); mean AURC −0.015 [−0.027, −0.004].

Per condition, router(sam) − router(ens) in balanced accuracy:
- Significantly negative: noise s1 −0.021 [−0.040, −0.005] and noise s3 −0.013 [−0.022, −0.003].
- Not significant: all other conditions, from blur s3 −0.020 [−0.044, +0.004] to blur s1 +0.011 [−0.003, +0.026].
- Per-condition numbers are in `results.json` (`D_shift_per_condition`).

**External (BDNeuro, 3,030 clean images, image-level bootstrap):**
- Fold-mean bal. acc: router(sam) 0.923 vs router(ens) 0.922, crop32 0.931, full 0.904.
- Per fold, router(sam) − router(ens): +0.020, −0.004, +0.008, −0.007, −0.014. Three folds have CIs that exclude 0, in opposite directions.
- The crop share also varies widely between folds: sam 0.48 / 0.00 / 0.25 / 0.96 / 0.71, ens 0.00 / 0.19 / 0.03 / 0.89 / 0.93.

## Conclusion
- **H-R2 is not supported.** MedSAM's disagreement with the U-Net is a weaker signal than the U-Net ensemble's own disagreement:
  - for segmentation failure (0.717 vs 0.899);
  - for "crop wrong, full right" (val 0.543 vs 0.625).
- In distribution, router(sam) ties router(ens). Under shift it is **worse in the worst case** (−0.0125, CI excludes 0). On external data the comparison is mixed.
- It keeps the benefit of routing over the fixed views: it beats crop32 in the worst case and has a lower mean AURC than both fixed views. That benefit comes from the router design, not from MedSAM.
- **Why [I]:** with the U-Net's own box, MedSAM is a weaker segmenter (Dice 0.593 vs 0.785). Its disagreement mostly reflects MedSAM's own errors, not the U-Net's.
- **For the thesis:** this is a clean, pre-specified negative result. A foundation-model second opinion does not beat a cheap 3-seed ensemble as the routing signal on this data. Together with R4 (EXP-010), the ensemble is the strongest available signal; the limit is the signal (oracle 0.974).
- **Not tried** (not pre-specified): GT-quality boxes, other MedSAM variants, or combining the signals.
