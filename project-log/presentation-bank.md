# Presentation Bank

The strongest results, figures and "wow" points for the final presentation, collected as we go. Each item has a source.

## Story beats
1. **"We audited ourselves."** The semester-1 test set contained 60 training images (55 %). On those, the full-image classifier is 100 % correct; on the clean 50 it is 80 %. Source: `experiments/EXP-000-semester1-checkpoint-eval.md`. *Honest-evaluation slide; shows scientific maturity.*
2. **"Segmentation-guided classification failed in its naive form."** Predicted-ROI classifier: 0.46 on clean test, close to the majority-class rate (0.44); the full image gets 0.80. Motivates the semester-2 redesign. Source: EXP-000.

3. **Even the "clean" test set was not clean.** 43 of its 50 images came from patients already seen in training (verified with patient IDs). Reason for patient-wise evaluation. Source: `00-project-understanding.md §8.2`.

4. **"We are not alone: the field's benchmarks leak."** Slice-level splits inflate accuracy by 29–55 % (Yagis 2021). Kaggle merges: 28.8 % near-duplicates, 95.5 % patient overlap (Vangala 2026). Our patient-disjoint protocol is a strength. Source: `research/research-summary.md §1A`.
5. **"Even the dataset's authors knew context matters."** Cheng 2015: tight ROI 83.5 % → dilated ROI 88.2 %. This matches our ROI finding. Source: literature-matrix #1.
6. **Positioning slide:** grid of "common" vs "open" angles (`research-summary.md §1B–C`).
7. **"We even found the images were sideways."** Semester-1 conversion never transposed the MATLAB arrays: every image was mirrored across the diagonal. The new pipeline is checked against the hand-drawn tumor borders (median IoU 0.970, 3,064/3,064 slices ≥ 0.846). Source: EXP-001.

8. **"Same network, +0.25 Dice."** The semester-1 U-Net with only a new training recipe goes from Dice 0.508 to 0.759 on the same patient-disjoint folds (paired Δ +0.250 [0.228, 0.273]). A pretrained ResNet-34 encoder adds +0.027 and cuts boundary error from 25.0 to 14.4 px HD95. Source: EXP-002.
9. **Glioma is the hard case.** Dice 0.685 for glioma vs 0.893 for meningioma (EXP-002c). Sets up the router: hard cases should go to a radiologist.

10. **"Context is the missing ingredient."** Balanced accuracy climbs with crop margin: masked ROI 0.870 → crop 0 px 0.881 → 8 px 0.928 → 16 px 0.939 → 32 px 0.954 (GT masks). Semester 1's masked ROI is significantly worse than the full image (−0.079 with predicted masks). Source: EXP-003.
11. **"Segmentation helps only when it is good."** Crop 32 beats the full image with GT masks (+0.026, p 0.005) but not with predicted masks (+0.013, CI crosses 0). The bridge to R1, quality-aware routing. Source: EXP-003.
12. **Baseline → now:** classification 0.733 → 0.941 balanced accuracy (+0.207 [0.172, 0.244]); segmentation Dice 0.508 → 0.786. Same patients, same folds. Sources: EXP-002, EXP-003.

13. **"The crop forgives bad masks, except false positives."** Crop 32 keeps 0.933 balanced accuracy with masks eroded to Dice 0.3, but drops to 0.636 when a false-positive blob is added at Dice 0.9. Source: EXP-004.
14. **"The router never collapses."** Over 13 synthetic shifts, its worst balanced accuracy is 0.607, against 0.535 for the crop and 0.581 for the full image. At blur s3 it beats the crop by +0.145 [+0.103, +0.186]. With CIs: worst case − crop +0.072 [+0.051, +0.110], − full +0.025 [+0.001, +0.063]; mean referral risk (AURC) −0.018 vs both. Sources: EXP-008, EXP-006b.
15. **"A public 'external' test set is one-third our own training data."** 1,682 BDNeuro-MRI images duplicate Figshare slices (215 of 233 patients), including 33 % of BDNeuro's own test split. On the 3,030 clean images the pipeline still scores 0.931. Source: EXP-009.
16. **Honest negative result:** segmentation uncertainty detects broken masks (AUROC 0.899) but does not predict classifier errors (≈ 0.5), and in-distribution routing ties the crop. The oracle head-room (0.974) shows what better routing could gain. Sources: EXP-005/006.
17. **Pre-registered honesty:** we wrote the τ1 stabilisation rule down and committed it before running; it failed (the val objective is flat), and we report that. Source: EXP-006b.

## Figures to make
- EXP-001 sample grid (`experiments/EXP-001-data-pipeline/samples.png`): 3 classes × 4 slices with mask contours.
- Bar chart: accuracy on contaminated vs clean test, for each of the 3 inputs (data in EXP-000 `results.json`).
- Diagram: image-level split vs patient-wise split (leakage illustration).
- Before/after repo hygiene: 225 MB → 210 KB (ADR-0001), as an optional engineering slide.
- EXP-002: bar chart of Dice/HD95 for arms a/b/c with patient CIs; per-class Dice bars (data in `experiments/EXP-002-segmentation/results.json`).
- EXP-003: balanced accuracy vs crop margin (masked, 0, 8, 16, 32, full) for GT and predicted masks, with CIs (data in `experiments/EXP-003-classification/results.json`).
- EXP-004: `experiments/EXP-004-mask-quality/accuracy_vs_mask_quality.png` (ready).
- EXP-006: `experiments/EXP-006-router/risk_coverage.png` (ready).
- EXP-008: `experiments/EXP-008-shift/accuracy_vs_severity.png` (ready).
