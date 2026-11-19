# T4 — Datasets, benchmarks, evaluation standards, leakage, semester-1 reference check

Verified: 2026-11-19. Every "verified" item below was opened in this session at the URL given. Anything not opened is marked **[UNVERIFIED]**. BibTeX: `T4.bib`. Deeper notes: `../papers/figshare-cheng-dataset.md`, `../papers/metrics-reloaded.md`, `../papers/BRISC2025.md`.

---

## 1. Figshare / Cheng brain tumor dataset (our current dataset)

| Field | Verified value | Source opened |
|---|---|---|
| Title (exact, lower-case on Figshare) | **brain tumor dataset** | https://api.figshare.com/v2/articles/1512427 |
| Author | Jun Cheng (ORCID 0000-0001-5493-961X) | same |
| DOI (concept / all versions) | 10.6084/m9.figshare.1512427 | same; also cited in Cheng 2015 |
| Current version | **v8**, DOI 10.6084/m9.figshare.1512427.v8, published 2024-12-21 | same |
| First version | v1 published 2015-08-17 (readme.txt + 4 zips) | https://api.figshare.com/v2/articles/1512427/versions/1 |
| Figshare-generated citation | "Cheng, Jun (2017). brain tumor dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427.v8" (note: year shows 2017 although v1 is 2015 and v8 is 2024) | API record |
| **Licence** | **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/) — same in v1 and v8 | API v1 and v8 |
| Files (v8) | 4 zips (`brainTumorDataPublic_1-766.zip` ... `_2299-3064.zip`, ~214–232 MB each, ~879 MB total), `cvind.mat` (5.7 KB), `README 2024.txt` | API v8 |
| Content (README 2024) | 3,064 T1-weighted contrast-enhanced slices from **233 patients**: meningioma 708, glioma 1,426, pituitary 930. `.mat` struct `cjdata` with `label` (1=meningioma, 2=glioma, 3=pituitary), `PID`, `image`, `tumorBorder` (manually delineated border points), `tumorMask` (binary). 5-fold CV indices in `cvind.mat`. | README downloaded from https://ndownloader.figshare.com/files/51340418 |
| Acquisition (README 2024) | Post Gd-DTPA T1-CE, Nanfang Hospital (Guangzhou) and General Hospital of Tianjin Medical University, 2005-09 to 2010-10; 512×512, 0.49×0.49 mm², slice thickness 6 mm, gap 1 mm; Gd 0.1 mmol/kg at 2 ml/s. "No further details are available." | same |
| README citation request | Cheng et al. 2015 PLoS ONE 10(10) and Cheng et al. 2016 PLoS ONE 11(6) (retrieval paper) | same |

**Licence implications.** CC BY 4.0 allows copying, redistribution and adaptation (including commercial) **with attribution** and an indication of changes. We may redistribute derived files (e.g., PNG conversions, patient-disjoint split lists) if we (a) credit Cheng, (b) link the licence, (c) state what we changed. Recommended citation pair in the thesis: the dataset DOI (v8, or the version actually downloaded) **plus** Cheng et al. 2015 PLoS ONE. (Note: the official Figshare HTML page returned HTTP 403 to the fetch tool; all metadata above came from the official Figshare API and the README file.)

**Cheng et al. 2015, PLoS ONE 10(10): e0140381**, DOI 10.1371/journal.pone.0140381 (opened https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381). Authors: Jun Cheng, Wei Huang, Shuangliang Cao, Ru Yang, Wei Yang, Zhaoqiang Yun, Zhijian Wang, Qianjin Feng. A correction exists (DOI 10.1371/journal.pone.0144479). **CV was patient-wise**: five-fold CV with patients split into five subsets — "Partitioning according to patients ensures that slices from the same patient will not exist in the training [and test sets]". Results: tumor region augmentation raised accuracy from 71.39→82.31% (intensity histogram), 78.18→84.75% (GLCM), 83.54→88.19% (BoW); with ring-form partition 87.54%, 89.72%, **91.28%**. These are the honest patient-wise baselines; most later >98% papers use image-level splits.

## 2. Nickparvar Kaggle "Brain Tumor MRI Dataset"

Opened: https://www.kaggle.com/api/v1/datasets/view/masoudnickparvar/brain-tumor-mri-dataset (the HTML page is JS-only; Kaggle's public API metadata was used).

- Owner Masoud Nickparvar; v1 released 2021-09-24; **v2 updated 2026-02-13**; licence **CC BY 4.0**; ~168 MB.
- v2 description: **7,200 images, 4 classes** (glioma, meningioma, pituitary, no tumor), 1,400 train + 400 test per class; merged from **figshare (Cheng), SARTAJ and Br35H**; no-tumor images from Br35H; SARTAJ glioma images with label problems replaced by figshare images; v2 "removed duplicates, balanced classes, eliminated train/test overlap". v1 was the widely used 7,023-image version (count reported by Vangala et al. 2026; see §4).
- **No segmentation masks, no patient IDs**, no provenance per image, no suggested citation.

**Is it a valid citation for our work? No.** We use the Figshare `.mat` files with `tumorMask` and `PID`; Nickparvar contains neither masks nor patient IDs, includes a no-tumor class we do not use, and redistributes Figshare images mixed with other sources. Citing it misstates our data provenance and invites the patient-leakage criticism. Correct practice: cite Cheng's Figshare DOI + Cheng 2015. Mention Nickparvar only if it is used (e.g., as a contaminated "external" set we deliberately avoid).

## 3. Dataset comparison

| Dataset | Year | Modality | 2D/3D | Size | Labels | Masks | Licence / access | Fit for us | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| **Figshare (Cheng)** | 2015 (v8 2024) | T1-CE | 2D slices 512×512 | 3,064 slices / 233 pts; ~0.88 GB | glioma / meningioma / pituitary | Yes (binary, manual) + PIDs + CV folds | CC BY 4.0, open download | **Best primary dataset** for our exact task (only one with masks + PIDs + 3 classes) | https://api.figshare.com/v2/articles/1512427 |
| Nickparvar Kaggle | 2021 (v2 2026) | MRI (mixed) | 2D | 7,200 imgs (v2); ~168 MB | 4 incl. no-tumor | No | CC BY 4.0 (Kaggle) | **Not valid as external test** — contains Figshare images | https://www.kaggle.com/api/v1/datasets/view/masoudnickparvar/brain-tumor-mri-dataset |
| SARTAJ "Brain Tumor Classification (MRI)" | 2020 (v3 2025) | MRI | 2D | ~93 MB; 3,264 imgs per Vangala 2026 | 4 classes | No | MIT (Kaggle) ; provenance not stated | Poor: unknown source, label noise; 81.8% contained in Nickparvar | https://www.kaggle.com/api/v1/datasets/view/sartajbhuvaji/brain-tumor-classification-mri |
| Br35H | 2020 (v12 2021) | MRI | 2D | 3,060 imgs (1,500 yes / 1,500 no / 60 pred); ~92 MB | tumor vs no-tumor only | No | "Data files © Original Authors" (unclear licence) | Only for a tumor/no-tumor gate; no type labels | https://www.kaggle.com/api/v1/datasets/view/ahmedhamada0/brain-tumor-detection |
| **BRISC 2025** (Fateh et al., Sci Data 13:361, 2026) | 2025 | T1-CE (as stated) | 2D, axial/coronal/sagittal | 6,000 imgs (5,000/1,000); 4,793 tumor images with masks; ~277 MB | glioma 1,401 / meningioma 1,635 / pituitary 1,757 / no-tumor 1,207 | Yes (PNG, physician-validated) | CC BY 4.0 (Kaggle DOI 10.34740/kaggle/ds/7632487) | Useful as **extra training/segmentation data and a second benchmark**, but **NOT independent**: derived from Nickparvar → contains Figshare images; no patient IDs; authors state subject independence "cannot be guaranteed" | https://arxiv.org/abs/2506.14318 ; https://arxiv.org/html/2506.14318v5 |
| **BDNeuro-MRI** (Hira et al., Mendeley v7) | 2026 | T1-CE | 2D | 5,941 imgs (glioma 2,191 / men 1,478 / pit 1,485 / no 787) | 4 classes | No | CC BY 4.0, DOI 10.17632/zwr4ntf94j.7, open | **Best candidate for external classification test** (different hospital: Epic & CSCR Hospital, Bangladesh; dedup by MD5 + pHash). No PIDs, no masks | https://data.mendeley.com/datasets/zwr4ntf94j/7 |
| Daffodil multi-class (Jamil et al., Mendeley v3) | 2026 | T1 and T2 | 2D | 1,200 imgs (300/class) | glioma / meningioma / pituitary macroadenoma / healthy | No | CC BY 4.0, DOI 10.17632/82mtzd8x72.3 | Second small external classification test (filter to T1 if possible) | https://data.mendeley.com/datasets/82mtzd8x72/3 |
| BraTS 2021 (Baid et al.) | 2021 | mpMRI (T1, T1-CE, T2, FLAIR) [sequences: standard BraTS, not listed on abs page] | 3D | 2,040 patients (incl. hidden) | glioma sub-regions (ET/TC/WT) + MGMT | Yes (3D) | Synapse (registration) [exact terms UNVERIFIED]; size [UNVERIFIED] | Glioma only, 3D, multi-modal → only usable as 2D T1-CE slice source for **segmentation** external check; class mismatch | https://arxiv.org/abs/2107.02314 |
| BraTS-MEN 2023 (LaBella et al., Sci Data 11:496, 2024) | 2023/24 | T1, T1-CE, T2, FLAIR | 3D | 1,141 public (1,000 train + 141 val); 283 private test | ET / NETC / SNFH | Yes (3D) | Synapse DOI 10.7303/syn51514106 [licence UNVERIFIED]; size [UNVERIFIED] | External **segmentation** test for meningioma on T1-CE slices; label mismatch (union of ET+NETC ≈ our "tumor") | https://pmc.ncbi.nlm.nih.gov/articles/PMC11096318 ; https://arxiv.org/abs/2305.07642 |
| BraTS-METS 2023 (Moawad et al.) | 2023 | T1, T2, FLAIR, T1-CE | 3D | 402 public training studies (1,303 annotated total) | metastasis lesions | Yes | Synapse | Not our classes; skip (maybe OOD test) | https://arxiv.org/abs/2306.00838 |
| BraTS-PEDs 2023 (Kazerooni et al.) | 2023 | mpMRI | 3D | [case count UNVERIFIED] | pediatric high-grade glioma | Yes | Synapse | Pediatric; skip | https://arxiv.org/abs/2305.17033 |
| **Pituitary NET dataset** (Pandit et al., Sci Data 12:80, 2025) | 2025 | T1-CE (+T2 for most) | 3D NIfTI 0.5 mm iso | 136 patients; **47.2 GB zip** | pituitary tumor + carotid masks | Yes (tumour.nii.gz) | CC BY 4.0, Figshare DOI 10.6084/m9.figshare.27894084 | External **pituitary segmentation** test after slicing; heavy download; 1 class only | https://pmc.ncbi.nlm.nih.gov/articles/PMC11735639/ ; https://api.figshare.com/v2/articles/27894084 |
| LGG segmentation (Buda, Kaggle; TCGA-LGG via TCIA) | 2019 | FLAIR (+pre/post-contrast) | 2D slices from 3D | 110 patients; ~1.06 GB | FLAIR abnormality | Yes | **CC BY-NC-SA 4.0** | Glioma only, FLAIR masks ≠ T1-CE enhancing tumor; weak fit | https://www.kaggle.com/api/v1/datasets/view/mateuszbuda/lgg-mri-segmentation |
| IEEE DataPort pituitary adenoma seg. (Wang & Wang, DOI 10.21227/66ks-t035) | — | 3D T1-CE | 3D | 55 cases | pituitary mask | Yes | Login required | [UNVERIFIED — only seen in search snippet, not opened] | — |

**Recommendation.**
1. **Keep Figshare (Cheng) as the primary dataset** — no public dataset is better for *2D T1-CE + binary masks + 3 classes + patient IDs*. Use the official `cvind.mat` patient-wise folds (or PID-grouped splits).
2. **External classification test: BDNeuro-MRI (Bangladesh, 2026)** — different country/hospital/scanner, 3 matching classes (drop no-tumor), CC BY 4.0. Optional second test: Daffodil set. Before use, check for near-duplicates against Figshare (pHash) — cheap and itself a reportable contribution.
3. **External segmentation test: BraTS-MEN (meningioma) and the Pandit pituitary set**, sliced to 2D T1-CE with tumor = union of sub-labels; BraTS 2021 for gliomas. Report as a domain-shift experiment, not as equal-protocol comparison.
4. **Do not** use Nickparvar, SARTAJ or BRISC as "external" tests: they contain Figshare images (BRISC via Nickparvar). BRISC can be used as additional *training* data only if Figshare-derived images are removed (pHash matching) — otherwise test leakage.
5. All candidates fit a 4 GB GPU if handled as 2D slices at 256×256 with small batches; the 3D sets need offline slicing (disk, not VRAM, is the constraint; pituitary set is 47 GB).

## 4. Leakage literature

| Paper | Verified URL | Key verified finding |
|---|---|---|
| Yagis et al. 2021, *Sci Rep* 11:22544, DOI 10.1038/s41598-021-01681-w | Europe PMC REST record (PMID 34799630) | 2D CNNs on 3D MRI: slice-level CV inflated slice-level test accuracy by ~30% (OASIS), 29% (ADNI), 48% (PPMI), 55% (local PD set). Randomly labelled data: ~96% accuracy with slice-level split vs ~50% (chance) with subject-level split. |
| Vangala et al. 2026, arXiv:2610.00421 "Scores That Hold, Benchmarks That Leak" | https://arxiv.org/abs/2610.00421 ; https://arxiv.org/html/2610.00421v1 | Nickparvar (7,023 imgs): 460/1,600 test images (28.75%) are near-duplicates of training images; of 201 test images traceable to Figshare (125 patients), 95.5% have a same-patient slice in training. SARTAJ: 22.3% of test images byte-identical to training; 81.8% of SARTAJ is contained in Nickparvar. Header-only features reach 0.959 balanced accuracy (tumor vs no-tumor). Dedup alone barely changes accuracy → "stable performance after deduplication does not establish benchmark integrity". (Paper reports split as 5,600/1,600 for 7,023 images, which does not add up — treat counts with care.) |
| Kapoor & Narayanan 2023, *Patterns* 4(9):100804, DOI 10.1016/j.patter.2023.100804 | Europe PMC REST | Taxonomy of leakage across ML-based science (bibliographic record verified; content summary not re-checked). |
| Varoquaux & Cheplygina 2022, *npj Digit Med* 5:48, DOI 10.1038/s41746-022-00592-y | Europe PMC REST | Methodological failures in medical imaging ML (bibliographic record verified). |
| BRISC 2025 (arXiv v5) | https://arxiv.org/html/2506.14318v5 | Authors themselves state "complete subject-level independence cannot be guaranteed" because source data lacked patient IDs. |
| Preetha et al. 2025 (our ref [5]) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11929897/ | Uses a 90/10 stratified split by images (2,758/306) on Figshare with no mention of PIDs → its Dice 0.9339 is likely leakage-inflated. |

**Implication for us:** semester-1 found 43/50 test images from training patients — consistent with the literature. Report only patient-disjoint results (cvind.mat folds), quote Cheng 2015 (91.28%) as the patient-wise baseline, and when comparing with literature, flag whether each paper used patient-level splits.

## 5. Evaluation standards

**Metrics Reloaded** (Maier-Hein et al., *Nat Methods* 21(2):195–212, 2024, DOI 10.1038/s41592-023-02151-z; opened PMC11182665 and arXiv:2206.01653):
- Segmentation: overlap metric (DSC or IoU) **plus** a boundary metric; NSD is the default boundary metric when reference annotations are imprecise. (HD95 is not mentioned in the visible text — use it as a secondary metric, with care.)
- Compute metrics **per image, then aggregate**; respect hierarchy (images nested in patients → aggregate per patient, too).
- Multi-class classification: a multi-class counting metric (Balanced Accuracy or MCC given class imbalance; Expected Cost when error costs differ) + AUROC as multi-threshold metric + per-class one-vs-rest metrics; calibration metrics if probabilities are used.
- Report **confidence intervals for all metrics** (especially small test sets), the full **confusion matrix**, raw-value visualisations, multiple runs for non-determinism; avoid ranking by closely related metrics.
- Companion pitfalls paper: Reinke et al., *Nat Methods* 21(2):182–194, 2024, DOI 10.1038/s41592-023-02150-0 (abstract verified; specific Dice/HD pitfalls not re-read here). Well-known pitfalls (general knowledge, [not re-verified in this session]): Dice is undefined/unstable for empty reference masks and penalises small structures; Hausdorff distance is outlier-sensitive (hence HD95), and is undefined when a prediction is empty.

**CLAIM 2024 update** (Tejani et al., *Radiol Artif Intell* 6(4):e240300, DOI 10.1148/ryai.240300; opened PMC11304031): **44 items**. Key: data sources incl. public datasets (item 7); partitioning method (19) and partitions "disjoint at the patient level or higher" (20); reference standard & annotators (14, 16, 17); external test data (33); metrics with uncertainty, e.g., "95% confidence intervals" (37, 38); statistical significance tests (29); failure analysis with confusion matrix (39); code/model availability (43).

**TRIPOD+AI** (Collins et al., *BMJ* 385:e078378, 2024, DOI 10.1136/bmj-2023-078378; opened PMC11019967): **27 main items** (+13-item abstract checklist). Key: data sources separately for development and evaluation (5a), sample-size justification (10), partitioning & internal validation (12a, 12c), clustering/heterogeneity (12d, 23b), performance with confidence intervals incl. subgroups (23a), calibration (12e), fairness (14), data/code availability (18e, 18f). (TRIPOD+AI targets prediction models; CLAIM is the primary checklist for an imaging paper — use CLAIM, mention TRIPOD+AI.)

**Concrete reporting plan for Mediscan:** patient-disjoint 5-fold CV (cvind.mat); per-image and per-patient Dice + IoU + NSD/HD95 (with empty-mask rule stated); classification balanced accuracy, macro-F1, MCC, AUROC, confusion matrix, calibration (ECE/reliability plot); 95% CIs by patient-level bootstrap; paired comparisons by patient-level bootstrap or Wilcoxon signed-rank across folds; external test on BDNeuro-MRI; CLAIM checklist as an appendix.

## 6. Semester-1 reference check

Counts: **VERIFIED as cited: 0 · VERIFIED with corrections: 9 · NOT FOUND: 2 (+1 unverifiable image credit)**. Serious mismatches: [2] wrong dataset; [3], [5], [9] wrong authors; [4] wrong title; [6], [7] wrong journal; [12] attributed to non-existent authors (numbers match a different paper); [11] not found.

| # | Status | Corrected citation | URL opened | Notes |
|---|---|---|---|---|
| 1 | NOT VERIFIABLE | Freepik (n.d.), image — no URL/title given | (freepik.com/legal/terms-of-use now redirects to magnific.com) | Need the exact image URL + licence. Better: replace with own figure or a CC-licensed image with full attribution. |
| 2 | VERIFIED with corrections (but **wrong source for our work**) | Nickparvar, M. (2021). *Brain Tumor MRI Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (v2 2026, CC BY 4.0) | Kaggle API (above) | Replace with Cheng, J. *brain tumor dataset*, figshare, doi:10.6084/m9.figshare.1512427 + Cheng et al. 2015. |
| 3 | VERIFIED with corrections | Ahamed, M. F., Hossain, M. M., Nahiduzzaman, M., Islam, M. R., Islam, M. R., Ahsan, M., & Haider, J. (2023). A review on brain tumor segmentation based on deep learning methods with federated learning techniques. *Computerized Medical Imaging and Graphics*, 110, 102313. doi:10.1016/j.compmedimag.2023.102313 | Crossref + Europe PMC (PMID 38011781) | **Authors wrong** ("Gaurav, Ahmed" are not authors); title truncated. Abstract does not mention "BraTS 2018" specifically (>100 papers reviewed; BraTS is a keyword) — "reviews BraTS 2018 models" [UNVERIFIED]. |
| 4 | VERIFIED with corrections | Bakas, S., Reyes, M., Jakab, A., Bauer, S., Rempfler, M., et al. (2018). Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. arXiv:1811.02629 | https://arxiv.org/abs/1811.02629 | **Title wrong** ("Methods… Lessons from" is not the title); author list in report (Akbari, Sotiras) does not match the first authors; 427 authors; v3 2019-04-23; no journal ref. |
| 5 | VERIFIED with corrections | Preetha, R., Jasmine Pemeena Priyadarsini, M., & Nisha, J. S. (2025). Brain tumor segmentation using multi-scale attention U-Net with EfficientNetB4 encoder for enhanced MRI analysis. *Scientific Reports*, 15, 9914. doi:10.1038/s41598-025-94267-9 | Crossref; Europe PMC (PMID 40121246); PMC11929897 | **Co-authors wrong** ("Jeyaraj, Sivamani" not authors). Dice 0.9339 on Figshare **matches** (also IoU 0.8795, acc 99.79%). Image-level 90/10 split → likely leakage. |
| 6 | VERIFIED with corrections | Demiroğlu, U. (2024). Tumor Detection by Classification of Brain MRI Images Using the Vision Transformers. *Adıyaman University Journal of Science*, 14(2), 140–156. doi:10.37094/adyujsci.1572289 | https://dergipark.org.tr/en/pub/adyujsci/article/1572289 | **Journal wrong** (not BAJECE). ViT 78.26%; ViT features + Tree 81.9%. SVM/k-NN are mentioned; dataset appears to be a small Kaggle tumor/no-tumor set. |
| 7 | VERIFIED with corrections | Çay, T. (2024). Deep learning-based brain tumor segmentation: A comparison of U-Net and SegNet algorithms. *Bozok Journal of Engineering and Architecture*, 3(2), 99–109. doi:10.70700/bjea.1581404 | https://dergipark.org.tr/en/download/article-cite-file/1581404/type/2 ; https://api.crossref.org/works/10.70700/bjea.1581404 | **Journal wrong** (not BAJECE) and it is a **comparison study, not a survey**. U-Net 96% vs SegNet 94% accuracy **matches**. |
| 8 | VERIFIED with corrections (minor) | Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *MICCAI 2015*, LNCS (pp. 234–241). Springer. doi:10.1007/978-3-319-24574-4_28; arXiv:1505.04597 | https://arxiv.org/abs/1505.04597 ; Crossref | Add full title, pages, DOI. |
| 9 | VERIFIED with corrections | Wang, B., Yang, J., Peng, H., Ai, J., An, L., Yang, B., You, Z., & Ma, L. (2021). Brain Tumor Segmentation via Multi-Modalities Interactive Feature Learning. *Frontiers in Medicine*, 8, 653925. doi:10.3389/fmed.2021.653925 | https://api.crossref.org/works/10.3389/fmed.2021.653925 | **Authors wrong** (Li, Menze, Shen are not authors) and **journal wrong** (Frontiers in Medicine, not Neuroscience). Uses BraTS 2018. |
| 10 | VERIFIED with corrections (minor) | Gupta, A., Dixit, M., Mishra, V. K., Singh, A., & Dayal, A. (2023). Brain Tumor Segmentation from MRI Images using Deep Learning Techniques. arXiv:2305.00257 (12th Int. Advanced Computing Conf.) | https://arxiv.org/abs/2305.00257 | Authors/title correct; add arXiv ID. R2U-Net mean IoU **0.8665 matches**; dataset = Figshare (3,064/233). |
| 11 | **NOT FOUND** | — | Crossref query "YOLObb" (0 results); Europe PMC; 2 web searches | No paper "YOLObb" by Jha, Shohan, Salam found. Nearest: Dipu, N. M., Shohan, S. A., & Salam, K. M. A. (2021). Deep Learning Based Brain Tumor Detection and Classification. *2021 Int. Conf. on Intelligent Technologies (CONIT)*, 1–6. doi:10.1109/CONIT51480.2021.9498384 (YOLOv5 on BraTS 2018 subset, 85.95% per search snippet [UNVERIFIED content]). Remove or replace. |
| 12 | **NOT FOUND as cited** (likely misattribution) | Probably: Abdusalomov, A. B., Mukhiddinov, M., & Whangbo, T. K. (2023). Brain Tumor Detection Based on Deep Learning Approaches and Magnetic Resonance Imaging. *Cancers*, 15(16), 4172. doi:10.3390/cancers15164172 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10453020 ; Crossref author search | No Nawaz/Baig/Rehman YOLOv7 paper found. Abdusalomov et al. matches all report claims: YOLOv7 (+CBAM, BiFPN feature fusion, SPPF+), **10,288 images, 4 classes, 99.5%**. Confirm with the student before replacing. |

## 7. [UNVERIFIED] / tool failures
- Figshare HTML page (figshare.com/articles/...) → HTTP 403; used Figshare API + README instead. Nature.com pages (Sci Data, Sci Rep) → cookie redirect; used arXiv/PMC/Europe PMC instead. PubMed web → cookie wall. Kaggle HTML → JS-only; used Kaggle public API. arXiv export API, Semantic Scholar API, one Crossref call → HTTP 429.
- BraTS 2021 / BraTS-MEN / METS / PEDs dataset download size in GB and exact Synapse licence terms — not verified.
- BraTS-PEDs case count — not verified.
- IEEE DataPort pituitary dataset (55 cases) — seen only in a search snippet.
- Freepik image [1] — no URL provided; cannot verify.
- Ref [3] claim "reviews BraTS 2018 models" — not confirmed from the abstract.
- Ref [11] Dipu et al. accuracy figure — from search snippet only.
- Dice/HD95 pitfall specifics — general knowledge; Reinke et al. 2024 full text not re-read.
