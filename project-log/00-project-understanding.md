# 00 — Project Understanding (Phase 1 Audit)

- **Date:** 2026-11-19
- **Scope:** read-only audit of the local folder, git state, and all documents in `docs/`.
- **Labels:** **[V]** = VERIFIED (seen in a file, command output, or run); **[I]** = INFERRED (reasoning); **[U]** = UNKNOWN.
- **Git HEAD at audit time:** `323890b` (local `main`); `origin/main` = `3337b1a`.

---

## 1. Git and GitHub state

| Item | Finding | Label |
|---|---|---|
| Remote | `origin` = `https://github.com/mudar123t/Mediscan-AI-MRI-Analysis` (fetch and push) | [V] |
| Linked? | Yes. Local history (`2167d62` → `dbe1761` → `323890b`) is an ancestor of `origin/main`. | [V] |
| Sync | Local `main` is **3 commits behind** `origin/main` (`d9baf30`, `ea75e9e`, `3337b1a`, all "Update README.md", made on GitHub on 2026-02-28). The only difference is +7 lines in `README.md` (embeds the 3 images in `images/`). No local commits are missing from GitHub. | [V] (`git diff HEAD origin/main`) |
| Working tree | Only `docs/` is untracked. Nothing else modified. | [V] |
| Branches / tags | Only `main`; no tags. | [V] |
| Push access | `gh` logged in as `mudar123t`; `viewerPermission: ADMIN`; token scopes include `repo`. | [V] |
| Visibility | **PUBLIC** repository. | [V] (`gh repo view`) |
| Repo size | `.git` = 225 MB locally; GitHub `diskUsage` = 228,343 KB. | [V] |

### 1.1 Repository hygiene problems
| Problem | Evidence | Label |
|---|---|---|
| Ignore file is inactive: it is named `.gitignore.txt`, not `.gitignore`. | file listing | [V] |
| 8,652 files are tracked, including the **whole converted dataset** (`dataset/output/**`, 6,128 PNGs + `cvind.mat`), the pilot data (`data/pilot/**`, 2,483 files), **4 checkpoints** (`outputs/pilot/checkpoints/*.pt`, 31–34 MB each, ~132 MB total) and 8 `__pycache__/*.pyc` files. | `git ls-files` | [V] |
| README says "the raw dataset is not included in this repository", but the converted dataset is tracked and public. | `README.md:25` vs `git ls-files` | [V] |
| Dataset licence / redistribution terms of the Figshare dataset. | not in repo | [U] — to verify in Phase 4. |
| No secrets found (`git grep` for key/token/password patterns). | command output | [V] |

---

## 2. Folder map (local)

```
final project/
├── README.md, requirements.txt, future.txt, conv.py, .gitignore.txt
├── dataset/                 # converted full dataset (tracked)
│   ├── README 2024.txt      # original Figshare dataset README
│   ├── cvind.mat            # 5-fold CV indices (MATLAB v7.3 / HDF5)
│   └── output/{glioma,meningioma,pituitary}/{images,masks}/<idx>.png
├── data/pilot/              # pilot subset (tracked)
│   ├── splits/{train,val,test}.txt
│   ├── processed/{images,masks}/{train,val,test}/<cls>_<idx>.png
│   ├── roi_gt/images/{split}/   # image × GT mask
│   └── roi_pred/images/{split}/ # image × predicted mask
├── src/mediscan/            # library code
│   ├── preprocessing/basic.py
│   ├── data/{dataset_seg.py, dataset_cls.py}
│   ├── segmentation/{unet.py, losses.py, metrics.py}
│   └── classification/{cnn.py, metrics.py}
├── scripts/01..10_*.py      # numbered pipeline steps
├── notebooks/01_dataset_explore.ipynb   # 0 bytes (empty file)
├── outputs/pilot/checkpoints/{unet_best, cnn_full_best, cnn_roi_gt_best, cnn_roi_pred_best}.pt
├── images/                  # 3 PNGs used in README
└── docs/                    # untracked: semester-1 reports and slides
```
Raw `.mat` files are **not** in the folder (`conv.py` reads `dataset\all`, which does not exist here). [V]

### 2.1 Documents in `docs/` (all read)
| File | Date (mtime) | What it is |
|---|---|---|
| `final report1.pdf` | 2026-01-08 | **Latest semester-1 thesis** ("Software Engineering Design I", 01.2026). Read in full. |
| `final report.pdf` / `final report.docx` | 2026-01-04 | Earlier version of the same report. |
| `Final project presentation.pdf` | 2026-01-09 | Semester-1 slides (18 slides). |
| `ara rapor MUDAR & IBRAHIM.pdf`, `ara rapor.pdf`, `ara rapor (1).docx`, `ara rapor.doc` | 2025-11 / 2026-01 | Interim ("ara") report, 11.2025. `.doc` is legacy binary; checked via `strings` only. |
| `final project notes (1).pdf` | 2025-11-21 | Draft notes (same text family as the interim report). |
| `Final_Report.docx` | 2025-12-21 | **Unrelated**: assignment template for an "Introduction to Data Science" course. |

---

## 3. Problem and dataset

**Problem.** Two-stage 2D pipeline on single brain-MRI slices: (1) binary tumor segmentation with a U-Net, (2) 3-class tumor-type classification (glioma, meningioma, pituitary) with a CNN, comparing three classifier inputs: full image, ROI from ground-truth mask, ROI from predicted mask. [V] (`README.md`, `final report1.pdf` §3)

**Dataset actually used: Figshare brain tumor dataset (Cheng et al.)**, identified from `dataset/README 2024.txt` [V]:
- 3,064 T1-weighted contrast-enhanced slices from **233 patients**; meningioma 708, glioma 1,426, pituitary 930.
- 512×512 in-plane, 0.49×0.49 mm², 6 mm slice thickness; Nanfang Hospital and Tianjin Medical University General Hospital, 2005–2010.
- Each `.mat` holds `label`, **`PID` (patient ID)**, `image`, `tumorBorder`, `tumorMask`.
- `cvind.mat`: 5-fold CV indices for all 3,064 images (fold sizes 542/679/572/628/643) [V]. Cheng et al. describe these folds as patient-wise [I — to verify against the paper in Phase 4].
- Local copy `dataset/output/` has exactly 1,426 / 708 / 930 image+mask pairs, 256×256 PNG [V].

**Citation mismatch.** All reports cite the dataset as *Nickparvar (2021), Brain Tumor MRI Dataset, Kaggle* (ref [2]) and README says "Available on Kaggle". The files and README in `dataset/` are the Figshare/Cheng dataset. [V] The Nickparvar Kaggle dataset is, to my knowledge, a 4-class classification set without masks [I — verify in Phase 4]. The workflow diagram (`images/workflow_diagram.png`) also shows a **"No Tumor"** output class that does not exist in this dataset or the code [V].

### 3.1 Conversion (`conv.py`) [V]
`.mat` (h5py) → min-max scaled to uint8 → `PIL.resize((256,256))` with the default resampler for **both** image and mask → PNG per class. Notes:
- Masks are interpolated by the resize, so they contain 63 distinct grey values (sample `glioma/masks/1866.png`) [V]. Later they are binarized with `> 0`, which slightly **inflates** the tumor (sample: 3,605 px at `>0` vs 3,429 px at `>127`) [V].
- h5py reads MATLAB arrays transposed, so images and masks may be transposed relative to the original orientation (consistently for both) [I].
- **PID is discarded**; output files are named only by original index. [V]

### 3.2 Pilot subset (`scripts/02_prepare_pilot_data.py`) [V]
- 500 images sampled at random (seed 42) from all 3,064, then split 80/10/10 → 400/50/50 **by image, not by patient, not stratified**.
- Preprocessing: grayscale read → /255 → Gaussian blur 3×3 → saved back as uint8 PNG; mask `>0` → {0,255}.
- Class counts from the split lists:

| Split | glioma | meningioma | pituitary | total |
|---|---|---|---|---|
| train | 188 | 102 | 110 | 400 |
| val | 24 | 8 | 18 | 50 |
| test | 17 | 22 | 11 | 50 |

---

## 4. Architectures and training setup [V]

| Stage | Model | Details | Source |
|---|---|---|---|
| Seg | U-Net, 1→1 ch | encoder 32-64-128-256, bottleneck 512, Conv-BN-ReLU ×2, ConvTranspose up, 1×1 head; **7,762,465 params** | `src/mediscan/segmentation/unet.py`, param count from run |
| Seg training | | 5 epochs, batch 4, Adam lr 1e-3, BCE+Dice (0.5/0.5), **no augmentation, no scheduler, no seed** in the training script; best checkpoint by val Dice (threshold 0.5) | `scripts/05_train_unet_pilot.py` |
| ROI | | U-Net probs > **0.3** → keep largest connected component → ROI = image with non-mask pixels set to 0 (masking, not cropping, at full 256×256) | `scripts/07_build_roi_dataset.py` |
| Cls | SimpleCNN, 1 ch in | 3×(Conv3×3-ReLU-MaxPool) 16-32-64 → FC 65,536→128 → Dropout 0.2 → FC 3; **8,412,419 params** (99.7% in the first FC layer) | `src/mediscan/classification/cnn.py` |
| Cls training | | 3 separate runs (full / roi_gt / roi_pred), 10 epochs, batch 8, Adam lr 1e-3, CE, no augmentation, no class weighting, no seed; best by val accuracy | `scripts/09_train_classifier.py` |
| Cls eval | | test accuracy + confusion matrix printed to console | `scripts/10_eval_classifier.py` |

All four checkpoints load into the code models with `strict=True` [V].

## 5. End-to-end pipeline flow [V]
1. `conv.py`: `.mat` → PNG per class (256×256).
2. `01_dataset_explore.py`: pair checks + visual overlays.
3. `02_prepare_pilot_data.py`: 500-image pilot, preprocess, split, write lists.
4. `03_check_pilot_data,py` (sic, comma in filename), `04_test_seg_dataset.py`: sanity checks.
5. `05_train_unet_pilot.py` → `unet_best.pt`.
6. `06_vis_unet_preds.py`: visualize val predictions.
7. `07_build_roi_dataset.py` → `roi_gt/`, `roi_pred/` for all splits.
8. `08_check_roi_dataset.py`: visual ROI check.
9. `09_train_classifier.py` → three `cnn_*_best.pt`.
10. `10_eval_classifier.py`: test accuracy + confusion matrix.

There is **no single inference entry point** (image in → mask + class out) and no demo/UI. [V]

---

## 6. Semester-1 metric values

**No semester-1 metric value is recorded anywhere in the project.** [V]
- No log files, CSVs, result JSONs, or saved figures with numbers exist.
- `final report1.pdf`, both earlier reports and the slides describe Dice, IoU, accuracy and confusion matrices but **report no numbers**. The slides say "Results used for feasibility and validation only".
- The only numbers in the documents are from cited papers (e.g. Dice 0.9339, IoU 0.8665, 99.5 % accuracy), not from this project.
- No segmentation test-set evaluation script exists; val Dice/IoU was only printed during training.

Therefore the semester-1 baseline must be **re-measured** (see §9 and Questions).

> **Update 2026-11-19:** re-measured in [EXP-000](experiments/EXP-000-semester1-checkpoint-eval.md) (inference only). Clean 50-image test: U-Net Dice 0.411; CNN accuracy full image 0.800, GT-ROI 0.740, predicted-ROI 0.460. The contaminated 110-image test gives 0.909 / 0.873 / 0.709.

---

## 7. Status: done / partial / missing

| Done [V] | Partial [V] | Missing [V] |
|---|---|---|
| `.mat` → PNG conversion | Evaluation: accuracy + confusion matrix only; no precision/recall/F1 (promised in interim report) | Any recorded results |
| Pilot (500) data prep | Preprocessing: interim report promised histogram equalization, skull stripping, augmentation; none implemented (final report correctly says no hist-eq) | Patient-wise split (promised in interim report §3.2) |
| U-Net, BCE+Dice, Dice/IoU | ROI: masking only (diagram says "crop") | Seg test evaluation, HD95 etc. |
| ROI (GT + pred) generation | Visual checks exist but are interactive `plt.show()` only | Full-dataset training |
| SimpleCNN 3-way comparison | | Augmentation, schedulers, seeds in training |
| 4 trained checkpoints | | Inference/demo app, tests, config files, experiment tracking |
| | | Notebook (file is empty) |

---

## 8. Weaknesses and risks

### 8.1 CRITICAL — Train/val/test contamination in the actual runs [V]
- `data/pilot/processed/images/val/` and `.../test/` each contain **110** files, not 50: the 50 listed files (written 2025-12-23 04:29) **plus 60 stale files** from earlier runs of step 2 (written 03:16 and 03:46). Step 2 never clears old outputs.
- **All 120 stale files are also in `train/`** (train∩val = 60, train∩test = 60, val∩test = 0; 500 unique files across 620).
- The datasets (`SegmentationDataset`, `ClassificationDataset`) **glob the directories**; they ignore `splits/*.txt`.
- `unet_best.pt` (04:40), the ROI sets (04:43) and the three CNNs (04:47–04:50) were all produced **after** the 04:29 split, so **every val and test number from semester 1 was computed on sets where 55 % of the images (60/110) were training images**. Model selection (best epoch) was also done on this contaminated val set.
- The thesis states the test set is held out "preventing data leakage" (`final report1.pdf` §4.6). This is not true for the runs that produced the checkpoints.

### 8.2 HIGH — Patient-level leakage even in the clean 50-image lists [I, strong evidence]
- Split is by image; the dataset has ~13 slices per patient (3,064/233) [V numbers], and PID was discarded.
- Evidence: `cvind` folds form 492 contiguous index runs over 3,064 files, i.e. files are grouped in blocks [V]. **19/50 test** and **24/50 val** images have a training image within ±3 indices in the **same CV fold** (i.e. very likely the same patient) [V computation, I interpretation].
- Normalized cross-correlation of each clean test image to its nearest training image: median 0.767; 15/50 > 0.8; 4/50 > 0.9 [V].

### 8.3 Other weaknesses
| # | Issue | Label |
|---|---|---|
| 1 | Very small, single run: 400 train images, 5 U-Net epochs, 10 CNN epochs, no seeds, no repeats, no confidence intervals → results not statistically meaningful. | [V] |
| 2 | Class imbalance not handled (glioma ≈ 47 % of train); test list has only 11 pituitary; meningioma has 8 val images. Accuracy alone is misleading. | [V] |
| 3 | `roi_gt` classifier uses **ground-truth masks at test time** — an upper bound, not a deployable result; must be reported as such. | [V] |
| 4 | `roi_pred` classifier is trained on U-Net predictions for **its own training images** (near-perfect masks) but tested on test-image predictions (worse) → train/test distribution shift. | [I] |
| 5 | Masking ROI removes all context (location, surrounding anatomy), which is itself highly discriminative for these 3 classes (e.g. pituitary location). ROI-only may perform **worse** than full image; that is a legitimate finding, not a failure. | [I] |
| 6 | Mask interpolation + `>0` binarization inflates GT masks. | [V] |
| 7 | Dice metric is per-image mean at threshold 0.5; ROI uses 0.3; no boundary metrics; no per-class seg results. | [V] |
| 8 | SimpleCNN: 99.7 % of params in one FC layer → overfitting-prone; no pretrained backbone. | [V] |
| 9 | Documentation inconsistencies: wrong dataset citation (Kaggle/Nickparvar), "No Tumor" class in diagram, "crop" vs masking, interim promises not met. Reference list uses placeholder URLs (e.g. "https://arxiv.org/", "https://www.sciencedirect.com/ or related journal"). | [V] |
| 10 | Novelty: the semester-1 design (U-Net → ROI → CNN on Figshare) is a common pattern in the literature — consistent with the university's feedback. | [I] |

---

## 9. Reproducibility assessment (not re-run)

| Aspect | Status | Label |
|---|---|---|
| Hardcoded absolute paths `C:\Users\shawa\Desktop\final project` in every script; the folder is now under `Downloads\Archive\...` → scripts fail as-is. | ✗ | [V] |
| No package install (`pyproject`/`setup.py`); imports need `PYTHONPATH=src`. | ✗ | [V] |
| `requirements.txt` unpinned (`>=`), and missing `h5py` (used by `conv.py`). | ✗ | [V] |
| Local environment: Python 3.12.10, **torch 2.9.1+cpu (CPU-only build, CUDA not available)**, numpy 2.2.6, opencv 4.12.0. | ✗ for GPU | [V] |
| GPU: **NVIDIA RTX 3050 Laptop, 4,096 MiB VRAM**, driver 610.60. | — | [V] (`nvidia-smi`) |
| Seeds only in data prep; training is non-deterministic. | ✗ | [V] |
| Split lists exist but are not used by loaders; directories are contaminated. | ✗ | [V] |
| Raw `.mat` files (with PID) not present locally. | ✗ | [V] |
| Checkpoints present and loadable. | ✓ | [V] |

**Verdict:** the semester-1 *numbers* cannot be reproduced (none were recorded) and the *evaluation protocol* behind the checkpoints is invalid (§8.1). The *code* can be re-run after fixing paths/env. A new, leak-free baseline (patient-wise split, same models) is needed before any semester-2 improvement can be claimed. [I]

---

## 10. Answers received (2026-11-19)
1. Fast-forward: "do what is better" → fast-forwarded to `3337b1a` before tagging.
2. History: **rewrite** (option c) → done, see [ADR-0001](decisions/ADR-0001-remove-data-and-weights-from-git-history.md).
3. Raw `.mat`: Kareem will re-download them. He is **open to a better-suited dataset** if one exists (evaluate in Phase 4).
4. Inference-only evaluation: approved → [EXP-000](experiments/EXP-000-semester1-checkpoint-eval.md). Kareem recalls the scripts ran fine in semester 1.
5. All documents are in `docs/`; there are no other sources.
6. No exact university wording. They want a creative addition. The supervisor is open to **publishing** a good idea → [university-requirements.md](university-requirements.md).
7. CUDA PyTorch: acceptable if it has no downsides (to be checked in Phase 6).

## 10a. Original open questions (for Kareem)
1. Fast-forward local `main` to `origin/main` (3 README-only commits) before tagging `v1.0-semester1`?
2. The public repo tracks the full dataset + 132 MB of checkpoints. Keep as is, stop tracking going forward (history keeps them), or rewrite history (needs explicit approval)?
3. Do you still have the original Figshare `.mat` files (they contain the patient ID)? Otherwise we rely on `cvind.mat` folds as the patient grouping.
4. May I run **inference-only** evaluation of the existing checkpoints (CPU, no training) to record the semester-1 numbers on both the contaminated 110-image and the clean 50-image test sets?
5. Which file is the canonical semester-1 thesis: `final report1.pdf` (2026-01-08)? Is there an editable source for it?
6. Exact university wording of the "add something novel" requirement, and the semester-2 deadlines (report, demo, presentation).
7. Is installing a CUDA build of PyTorch acceptable (needed to use the GPU; Phase 6 only)?
