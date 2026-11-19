# Sharma 2026 — MC Dropout + entropy-thresholded selective prediction for brain tumor MRI triage

- **Verified URL:** https://arxiv.org/abs/2607.16317 (opened 2026-11-19)
- **Author:** Medhansh Sharma (sole author). arXiv preprint, submitted 15 Jul 2026, cs.CV/cs.AI/cs.LG. 18 pages, 8 figures, 5 tables. No peer-reviewed venue listed.
- **BibTeX key:** `sharma2026monte`

## What it does (from the abstract page)
- 4-class classification (glioma, meningioma, pituitary, no tumor) on 7,200 images. The split is by perceptual-hash cluster to stop near-duplicate leakage.
- Uses MC dropout with T = 20 passes. Predictive entropy decides which cases are deferred to a radiologist.
- Argues that an auxiliary learned confidence head collapses to a near-constant output.
- Compares ViT-B/16 and ResNet-50 over 5 seeds. There is no significant architecture difference (0/5 seeds at p < 0.05).

## Reported numbers
| Metric | Value |
|---|---|
| Accuracy | 0.962 (ViT), 0.964 (ResNet) |
| Macro-AUC | 0.994 |
| ECE after single-temperature scaling | 0.016–0.020 |
| Accuracy at 95% coverage (defer 5%) | ~0.98 |
| AURC | 0.010–0.011 |

## Why it matters for Mediscan E13
- It is the **closest competitor** found. It already claims "uncertainty-aware brain tumor MRI triage with risk–coverage and calibration." E13 must therefore *not* claim novelty for classifier-only selective prediction.
- What it does **not** do (inference from the abstract):
  - no segmentation stage;
  - no use of segmentation uncertainty;
  - no routing between ROI and full-image views;
  - not on the Figshare/Cheng patient-disjoint folds.

  These are E13's differentiators.
- Use its metric set (ECE, AURC, accuracy at fixed coverage) as the minimum reporting bar. Its MSP/entropy gate is a mandatory baseline against E13's segmentation-uncertainty gate.
- Its warning that learned confidence heads collapse supports using MC dropout or TTA rather than a learned gate on a small dataset.
