# Asare 2026 — Benchmarking CNNs, ViTs and Foundation Models for Explainable Brain Tumor MRI Classification

- **Verified URL:** https://arxiv.org/abs/2611.11793 (opened 2026-11-19)
- **Author:** Akwasi Asare. arXiv v1, submitted 9 Nov 2026, cs.CV. Under review at *Scientific Reports* (per page).
- **BibTeX key:** `asare2026benchmarking`

## Reported (abstract page)
- Six models on 7,216 public 4-class brain tumor MRI images: EfficientNetV2-M, ConvNeXt-V2-Tiny, Swin-V2-S, ViT-L/16, DINOv2-L, MedSigLIP-448.
- Internal test accuracy is 94.5–95.6%, with no significant pairwise differences after Holm correction.
- **36% of test images had a near-duplicate in training.** After removing 773 duplicate or shared-patient test images, accuracy fell to **90.3–91.7%**.
- Ensembles did not beat the best single model.
- External test, BDNeuro-MRI (1,675 images): accuracy 77.6% (MedSigLIP-448) to 95.0% (ViT-L/16).
- Calibration and decision analysis were done with quantities fitted on validation only. ECE values were not shown on the page.
- **Attribution maps "localized expert-annotated tumors poorly"** while passing a model-randomization test. The exact metric was not shown on the page.

## Relevance to Mediscan
1. **XAI-faithfulness novelty:** Quantifying attribution vs tumor masks on brain-tumor classification is now being done, here and in Filvantorkaman 2025 on Figshare. It is no longer a clean "first." Mediscan can still contribute by linking faithfulness to correctness and to the ROI-vs-full-image routing (inference).
2. **Leakage:** This is strong external justification for Mediscan's patient-disjoint Figshare folds. Inflated public-benchmark accuracies, such as the ~95–99% common on Kaggle sets, are partly leakage.
3. **Context vs tumor:** Poor tumor localization by attribution maps is consistent with classifiers using non-tumor or peritumoral context. Together with Cheng 2015 (dilated ROI helps), this supports Mediscan's observation that full image (0.80) > ROI-from-GT (0.74).
