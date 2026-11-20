# EXP-008 — Routing under synthetic acquisition shift (M1.4, H-B3)

- **Date:** 2026-11-20.
  - Preliminary run (det + tta): 10:35 → about 11:30.
  - **Final run (det + ens, the frozen EXP-006 score):** 16:07 → 16:46, together with EXP-009.
  - Logs: `runs/exp008_009_ens.log`, `runs/eval_shift_external_final.log`.
- **Branch / code:** `exp/m1-uncertainty`.
  - `src/mediscan/corrupt.py`, `src/mediscan/pipeline.py`, `scripts/run_shift_external.py`, `scripts/eval_shift_external.py`.
  - Test: `tests/test_corrupt.py`.
- **Pipeline:** the end-to-end pipeline runs per fold k on test fold k:
  - EXP-002c segmenter (plus seeds 1 and 2 for `ens`) → largest component → views → EXP-003 classifiers.
  - [V] `pipeline.run_fold` reproduces the stored EXP-003 logits (max |Δ| 0.008, identical argmax) and the EXP-005 det scores. The clean condition reproduces EXP-006 exactly: router 0.935, crop32 0.941.
- **Policy parameters:**
  - Fitted per fold on the **clean** val fold: temperatures, τ1, combined regression and conformal q, with score `ens:disagree` frozen in EXP-006.
  - Applied unchanged to the corrupted test images. No refitting on shifted data.
- **Corruptions** (`corrupt.LEVELS`, applied to the image scaled to [0, 1] before segmentation, then clipped and returned to uint8):
  - Gaussian noise σ 0.03 / 0.06 / 0.10.
  - Gaussian blur σ 1 / 2 / 3 px.
  - Gamma x^1.5 / x^2 / x^3.
  - Multiplicative bias field exp(s·g), with s 0.2 / 0.4 / 0.6 and g smooth noise (σ 40 px) normalised to [−1, 1].
  - Levels were fixed before any result [I: mild to strong; not taken from a benchmark].
  - RNG `default_rng([0, row, kind, severity])`.
- **Raw:** `runs/exp008_shift/<cond>_fold{k}.csv`, 65 files (git-ignored). Preliminary det+tta outputs are kept in `runs/exp008_shift_prelim_tta/`.
- **Results:** [EXP-008-shift/results.json](EXP-008-shift/results.json), figure [accuracy_vs_severity.png](EXP-008-shift/accuracy_vs_severity.png). Per condition: 3,064 slices from 233 patients, 2,000 paired patient-bootstrap resamples.

## Hypothesis
**H-B3 (plan):** the benefit of routing grows with shift severity.

## Results [V: results.json]
Balanced accuracy at 100 % coverage. Mean Dice is the EXP-002c predicted-mask Dice under that corruption. "Crop share" is the fraction of slices routed to the crop.

| Condition | Mean Dice | Crop share | Full | Crop 32 | **Router** | Oracle | Router − crop32 [95 % CI] | Router − full [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| clean | 0.785 | 0.41 | 0.928 | 0.941 | 0.935 | 0.974 | −0.006 [−0.019, +0.008] | +0.007 [−0.002, +0.017] |
| noise s1 | 0.735 | 0.36 | 0.772 | 0.800 | 0.802 | 0.856 | +0.002 [−0.013, +0.017] | **+0.030 [+0.011, +0.049]** |
| noise s2 | 0.620 | 0.28 | 0.678 | 0.666 | 0.690 | 0.745 | **+0.024 [+0.007, +0.041]** | +0.011 [−0.001, +0.024] |
| noise s3 | 0.342 | 0.19 | 0.600 | 0.535 | 0.607 | 0.681 | **+0.072 [+0.048, +0.096]** | +0.007 [−0.004, +0.017] |
| blur s1 | 0.759 | 0.39 | 0.898 | 0.907 | 0.892 | 0.973 | −0.015 [−0.036, +0.006] | −0.006 [−0.024, +0.010] |
| blur s2 | 0.660 | 0.30 | 0.777 | 0.748 | 0.774 | 0.913 | +0.026 [−0.004, +0.055] | −0.003 [−0.028, +0.022] |
| blur s3 | 0.556 | 0.24 | 0.701 | 0.539 | 0.684 | 0.833 | **+0.145 [+0.103, +0.186]** | −0.017 [−0.043, +0.009] |
| gamma s1 | 0.768 | 0.40 | 0.854 | 0.925 | 0.904 | 0.960 | **−0.021 [−0.038, −0.006]** | **+0.050 [+0.027, +0.078]** |
| gamma s2 | 0.735 | 0.37 | 0.739 | 0.860 | 0.829 | 0.904 | **−0.032 [−0.054, −0.011]** | **+0.090 [+0.058, +0.125]** |
| gamma s3 | 0.619 | 0.30 | 0.581 | 0.664 | 0.642 | 0.750 | −0.022 [−0.056, +0.009] | **+0.060 [+0.035, +0.088]** |
| bias s1 | 0.784 | 0.40 | 0.927 | 0.945 | 0.936 | 0.975 | −0.009 [−0.022, +0.005] | +0.010 [+0.000, +0.021] |
| bias s2 | 0.781 | 0.41 | 0.927 | 0.941 | 0.940 | 0.976 | −0.001 [−0.015, +0.013] | **+0.013 [+0.004, +0.024]** |
| bias s3 | 0.778 | 0.40 | 0.920 | 0.939 | 0.932 | 0.973 | −0.007 [−0.021, +0.005] | **+0.012 [+0.001, +0.024]** |

Summary over the 13 conditions (point estimates; derived from results.json, no CI yet):

| Policy | Worst bal. acc (condition) | Mean bal. acc | Worst AURC (condition) | Mean AURC |
|---|---|---|---|---|
| full + MSP | 0.581 (gamma s3) | 0.792 | 0.242 (gamma s3) | 0.090 |
| crop32 + MSP | 0.535 (noise s3) | 0.801 | 0.324 (blur s3) | 0.090 |
| masked + MSP | 0.372 (noise s3) | 0.709 | 0.460 (noise s3) | 0.154 |
| **router** | **0.607 (noise s3)** | **0.813** | **0.192 (noise s3)** | **0.072** |
| oracle router (labels) | 0.681 | 0.886 | 0.152 | 0.047 |

Conformal referral under shift (α = 0.1, q fitted on clean val):
- Coverage collapses: router 0.402, crop32 0.305 and full 0.426 at noise s3; router 0.336 at blur s3; router 0.767 at gamma s2.
- The referral rate rises to 0.51 (noise s3), 0.59 (blur s3) and 0.17 (gamma s2), against 0.086 clean.

## Conclusions
- **H-B3 is partly supported.**
  - Against the crop view, the routing gain grows monotonically with severity under noise (+0.002 → +0.024 → +0.072) and under blur (−0.015 → +0.026 → +0.145). These are the corruptions that break the segmenter (Dice 0.34 / 0.56 at s3).
  - Mechanism: as segmentation degrades, `ens:disagree` rises and the router sends more slices to the full view (crop share 0.41 → 0.19).
  - Under gamma the segmenter barely degrades but the full-image classifier does. The router is then worse than the crop (−0.02 to −0.03) but much better than the full view (+0.05 to +0.09).
  - Bias field changes almost nothing.
- **The router is never the worst policy and never the best fixed view.** It has the best worst case and the best mean of all deployable policies: worst 0.607 vs 0.581 (full) and 0.535 (crop); mean AURC 0.072 vs 0.090. [I] Its value is **robustness when the type of shift is unknown**, rather than a gain in any single condition. CIs for the worst-case and mean summaries are not computed yet.
- **Conformal guarantees do not survive shift**, as expected, since exchangeability is violated. Coverage falls far below 0.9, but the referral rate rises sharply. As an alarm, CP referral still reacts to shift.
- The oracle gap stays large (e.g. blur s3: router 0.684 vs oracle 0.833), so better routing signals would pay off most under shift.
