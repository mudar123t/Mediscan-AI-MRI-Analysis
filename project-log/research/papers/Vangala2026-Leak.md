# Vangala et al. 2026: Scores That Hold, Benchmarks That Leak

**Citation:** Vangala BP, Guda S, Peddi L, Vangala N. *Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification.* arXiv:2610.00421, submitted 30 Sep 2026. **This is a preprint and has not been peer reviewed.** BibTeX key: `vangala2026scores`.
**Verified links (opened 2026-11-19):** https://arxiv.org/abs/2610.00421 and https://arxiv.org/html/2610.00421

## Problem
High accuracy on public brain-tumor MRI benchmarks does not show that the test data is independent of the training data.

## Method
A three-layer contamination framework:
1. duplicate leakage (near-twin or byte-identical images),
2. patient leakage (same patient in train and test),
3. source-label leakage (acquisition or source artifacts that predict the label).

The authors test 9 architectures under 3 evaluation conditions, with a pediatric chest X-ray negative control.

## Data
- C1: Nickparvar, 7,023 images, 4 classes, official 5600/1600 split. It is a merger that includes the figshare (Cheng) data.
- C2: SARTAJ, 3,264 images.
- C3: Navoneel, 253 images.

## Results (as reported)
- 28.8% of C1's official test split has a near-duplicate in training. 22.3% of C2's test images are byte-identical to training images.
- The authors recover patient IDs and state that the figshare release has 3,064 slices from 233 patients (about 13.2 per patient).
- 201 of C1's 1,600 test images trace to figshare originals. Of those, **95.5% share a patient with training** (a conditional estimate).
- File-header metadata alone separates tumor from no-tumor at 0.959 balanced accuracy.
- Deduplicating the test set barely changes balanced accuracy (mean change about −0.009; ConvNeXt-T 0.950 both ways). The conclusion: "stable performance after deduplication does not establish benchmark integrity."
- They claim "every published split on this benchmark is image-level."

## Limitations
- It never retrains or evaluates under a **patient-disjoint** split, so it cannot quantify the accuracy inflation caused by patient leakage.
- The claim that all published splits are image-level is contradicted by Cheng 2015 and Deepak & Ameer 2019, which used patient-level 5-fold CV on figshare. It may apply only to the Kaggle-merged benchmark (C1).

## What we can reuse
- Strong, current motivation for our ADR on patient-disjoint evaluation and for documenting the semester-1 leakage honestly.
- **Opening:** a measured comparison of image-level vs patient-disjoint splits for the full seg→cls pipeline on figshare, which we can produce cheaply with the official `cvind` folds. Yagis 2021 gives the expected size of the effect on brain MRI (about 30 to 55% inflation).
- Their released recovered patient IDs and dedup lists could help check overlap if we use C1/BRISC-type merged sets as external tests.
