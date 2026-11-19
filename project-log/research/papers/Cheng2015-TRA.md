# Cheng et al. 2015: Tumor Region Augmentation and Partition

**Citation:** Cheng J, Huang W, Cao S, Yang R, Yang W, Yun Z, Wang Z, Feng Q. *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition.* PLoS ONE 10(10): e0140381, 2015. DOI 10.1371/journal.pone.0140381. A correction was issued Dec 2015. BibTeX key: `cheng2015enhanced`.
**Verified link (opened 2026-11-19):** https://journals.plos.org/plosone/doi?id=10.1371/journal.pone.0140381

## Problem
Classify glioma / meningioma / pituitary tumors on 2D T1-CE MRI slices. The authors argue that spatial pyramid matching suits brain tumors poorly because the tumors vary so much in shape and size.

## Method
- Start from the expert tumor mask (the ROI).
- **Tumor region augmentation:** dilate the mask so the ROI includes surrounding tissue.
- **Ring partition:** split the augmented ROI into ring-shaped sub-regions by distance from the tumor border, and pool features per ring.
- Features: intensity histogram, GLCM, bag-of-words. Classifier: SVM.

## Data and protocol
This is the paper that introduced our dataset: 3,064 slices from 233 patients. It uses **5-fold, patient-disjoint** cross-validation, so slices from one patient never appear in both train and test. These are the `cvind` folds shipped with the dataset.

## Results (accuracy)
| Feature | Tumor only | Augmented ROI | Augmented + ring |
|---|---|---|---|
| Intensity hist. | 71.39% | 82.31% | 87.54% |
| GLCM | 78.18% | 84.75% | 89.72% |
| BoW | 83.54% | 88.19% | 91.28% |

Accuracy peaked at about 8 px dilation. Larger radii slightly reduced it, which the authors attribute to irrelevant information. Augmentation increased meningioma/glioma confusion somewhat, but overall accuracy still improved.

## Limitations
- Hand-crafted features. It needs the expert mask at test time (GT ROI), so segmentation error is not studied.
- Results are from 2015, before CNNs.

## What we can reuse
1. **Explanation for our semester-1 result.** Our GT-ROI classifier (0.74) did worse than full image (0.80). Cheng shows that "tumor tissues also provide important clues", from the surrounding tissue, and that a tight mask throws them away.
2. **A cheap ablation:** classify on dilated ROIs of k ∈ {0, 4, 8, 16, 32} px, using both GT and predicted masks. Nobody we verified has redone this with CNNs and *predicted* masks.
3. **For E13 (gating):** make the dilation or fallback depend on segmentation uncertainty. Dilate more, or fall back to the full image, when the mask is uncertain.
4. **Benchmark comparability:** 91.28% under patient-disjoint folds is the honest reference baseline for this dataset.
