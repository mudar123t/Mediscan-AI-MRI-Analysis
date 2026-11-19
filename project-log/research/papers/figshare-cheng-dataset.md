# Figshare brain tumor dataset (Cheng) — dataset card

- **Citation (dataset):** Cheng, Jun. *brain tumor dataset*. figshare. Dataset. doi:10.6084/m9.figshare.1512427 (current v8: 10.6084/m9.figshare.1512427.v8). Figshare's auto-citation prints the year as 2017.
- **Citation (paper):** Cheng J, Huang W, Cao S, Yang R, Yang W, Yun Z, Wang Z, Feng Q. Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition. *PLoS ONE* 10(10): e0140381 (2015). doi:10.1371/journal.pone.0140381 (correction: 10.1371/journal.pone.0144479).
- **Verified links (2026-11-19):** https://api.figshare.com/v2/articles/1512427 ; https://api.figshare.com/v2/articles/1512427/versions/1 ; README https://ndownloader.figshare.com/files/51340418 ; https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381 . (figshare.com HTML returned 403 to the fetch tool.)

## Facts
- Licence **CC BY 4.0** (v1 and v8). Redistribution and derivatives allowed with attribution + licence link + statement of changes.
- Versions: v1 2015-08-17 (readme.txt); v8 2024-12-21 (README 2024.txt adds acquisition protocol and MATLAB jpg-export code; adds `cvind.mat` in the file list).
- 3,064 T1-CE slices, 233 patients: meningioma 708, glioma 1,426, pituitary 930 (≈13 slices/patient on average).
- Files: four zips of 766 slices each (~214–232 MB), `cvind.mat` (5-fold CV indices), README.
- `.mat` struct `cjdata`: `label` (1 men, 2 gli, 3 pit), `PID`, `image`, `tumorBorder` (manual border points), `tumorMask` (binary).
- Acquisition: post Gd-DTPA, Nanfang Hospital (Guangzhou) + General Hospital of Tianjin Medical University, Sep 2005 – Oct 2010; 512×512, 0.49×0.49 mm², 6 mm slices, 1 mm gap; 0.1 mmol/kg Gd at 2 ml/s.
- Cheng 2015 used **patient-wise 5-fold CV**; best accuracy 91.28% (BoW + region augmentation + ring partition).

## What this means for Mediscan
- Correct the semester-1 citation (it is not Nickparvar/Kaggle).
- Use `cvind.mat` (or PID-grouped folds) so results are comparable with Cheng 2015 and leakage-free.
- Mixed planes (axial/coronal/sagittal) are present [plane counts not in README — UNVERIFIED]; report this as a dataset property.
- When publishing derived data (PNG masks, split files), add: "Derived from Cheng (figshare, CC BY 4.0); converted from .mat to PNG; no other changes."
- Pitfall noted by Vangala et al. 2026: arrays are column-major — read without transposing they don't match Kaggle copies (relevant for de-duplication against external sets).
