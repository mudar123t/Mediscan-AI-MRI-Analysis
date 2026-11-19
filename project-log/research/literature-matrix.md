# Literature Matrix

- **Last updated:** 2026-11-19 (Phase 4).
- **Rule:** a paper is listed only after its DOI / arXiv / publisher page was opened and checked. Otherwise it is marked `[UNVERIFIED]` or left out.

## How this is organised
- **Full evidence tables**, about 116 verified items, each with a verified URL. They are split by topic:
  - [T1 — SOTA segmentation/classification, multi-task, leakage](topics/T1-sota-seg-cls-multitask.md) (25 verified)
  - [T2 — Uncertainty, calibration, selective prediction, XAI, HITL](topics/T2-uncertainty-xai-hitl.md) (38 verified)
  - [T3 — Foundation models, generative data, edge, report generation](topics/T3-foundation-generative-edge-reports.md) (27 verified)
  - [T4 — Datasets, evaluation standards, leakage, semester-1 reference check](topics/T4-datasets-evaluation-references.md) (33 bib entries)
- **BibTeX:** [references.bib](references.bib) (116 unique entries, merged and de-duplicated).
- **Deeper notes:** [papers/](papers/) (13 notes).
- **Core table below:** the papers that most directly shape our decisions.

## Independent spot-check (2026-11-19, by the main agent via a separate fetch)
| Item | Result |
|---|---|
| Sharma 2026, arXiv 2607.16317 | ✅ Title, single author (M. Sharma), 15 Jul 2026. Abstract confirms MC-dropout entropy deferral to a radiologist, 4-class MRI, ECE 0.016–0.020, ~0.98 acc at 95 % coverage, AURC 0.010–0.011. **No segmentation stage in the abstract.** |
| Vangala 2026, arXiv 2610.00421 | ✅ Title, 4 authors, 30 Sep 2026; 28.8 % near-twins, 95.5 % patient overlap of traceable test images. ⚠ The abstract does **not** name Figshare/Kaggle. That mapping comes from the full HTML text as read by T1/T4 and was not re-checked here. |
| Figshare dataset API record 1512427 | ✅ "brain tumor dataset", Jun Cheng, **CC BY 4.0**, v8 published 2024-12-21. ⚠ The API also shows "firstOnline 2017-04-02", while T4 reports v1 as 2015-08-17 (from the versions endpoint). Cite the version actually downloaded. |
| Cheng 2015, PLOS ONE e0140381 | ✅ Title, J. Cheng, 2015. Methods: "Partitioning according to patients …" (patient-wise 5-fold). Best 91.28 % (BoW + augmentation + ring). ⚠ A correction notice exists (e0144479), not yet read. |

## Core table
| # | Short name | Year | Venue | Method | Dataset | Split | Key metrics (as reported) | Key idea | Relevance to Mediscan | Verified URL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cheng2015-TRA | 2015 | PLOS ONE | Tumor-region augmentation (dilation) + ring partition, BoW/GLCM/IH + SVM | Figshare (dataset paper) | patient 5-fold | Tight ROI 83.54 % → dilated 88.19 % → + ring 91.28 % (BoW) | Peritumoral context carries class information | Explains our full 0.80 > GT-ROI 0.74; it is the **honest patient-wise reference** | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381 |
| 2 | Deepak2019-GoogLeNet | 2019 | Comput Biol Med | GoogLeNet features + SVM/KNN | Figshare | patient 5-fold | mean acc 98 % | Transfer learning works with a patient split | Strong patient-level classification target | see T1 R3 |
| 3 | Avazov2024-LightUNet | 2024 | Bioengineering | Lightweight attention U-Net | Figshare (+BraTS) | patient 80/20 | Dice 0.93 ± 0.005 | A light U-Net is enough | **Segmentation target**: our 0.41 is a recipe problem | https://pmc.ncbi.nlm.nih.gov/articles/PMC11727338/ |
| 4 | Preetha2025-MSAUNet | 2025 | Sci Rep | EfficientNetB4-encoder attention U-Net | Figshare | image 90/10 | Dice 0.9339 | Pretrained encoder | Semester-1 ref [5]; Dice likely leakage-inflated | https://pmc.ncbi.nlm.nih.gov/articles/PMC11929897/ |
| 5 | Isensee2021-nnUNet | 2021 | Nature Methods | Self-configuring U-Net | 49 tasks | — | beats most bespoke pipelines | Recipe > architecture | Reference segmentation baseline | https://arxiv.org/abs/1904.08128 |
| 6 | Ferreira2024-MTLvsST | 2024 | J Health Inform | Shared encoder seg+cls vs single task | SARTAJ+Figshare | not stated | MTL 95.99 % vs ST 95.87 % | Multi-task gives no accuracy gain | D12 is not a contribution | https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/1296 |
| 7 | Gupta2021-MAGNet | 2021 | arXiv | Multi-task attention U-Net | Figshare | n/e | n/e | Joint seg+cls on our data | D12 prior art | https://arxiv.org/abs/2107.12321 |
| 8 | Zaman2025-IUTthesis | 2025 | BSc thesis | MRI + predicted-mask 2-channel → InceptionV3 | Figshare | not stated | Dice 0.80; acc 93.23 % | Mask as an extra channel | D11 prior art (grey literature) | https://repository.iutoic-dhaka.edu/handle/123456789/2645 |
| 9 | Alqudah2019-CropCmp | 2019 | IJATCSE | CNN on uncropped / cropped / lesion-only | Figshare (3,064) | not stated | 99.0 / 98.93 / **97.62 %** | Lesion-only input is worst | Supports the context finding | https://arxiv.org/abs/2001.08844 |
| 10 | Yagis2021-Leakage | 2021 | Sci Rep | Slice- vs subject-level CV | OASIS, ADNI, PPMI, local | both | 29–55 % inflation; random labels ~96 % vs ~50 % | Slice-level splits inflate accuracy | Justifies patient-disjoint folds | see T1 R18 |
| 11 | Vangala2026-Leak | 2026 | arXiv | 3-layer contamination audit | Kaggle merges incl. Figshare | audit | 28.8 % near-twins; 95.5 % patient overlap | Benchmarks leak | Audits but never retrains patient-disjoint, which leaves an opening | https://arxiv.org/abs/2610.00421 |
| 12 | Asare2026-BTBenchmark | 2026 | arXiv (under review) | 6 models, calibration, decision analysis, external test | 7,216 public 4-class + BDNeuro-MRI | dedup + patient | 94.5–95.6 % → 90.3–91.7 % after dedup; external 77.6–95.0 %; attributions localise tumors poorly | Rigorous re-benchmark + external test | Closest "rigour" competitor; template | https://arxiv.org/abs/2611.11793 |
| 13 | Sharma2026-BTTriage | 2026 | arXiv | MC-dropout entropy deferral + TS | 7,200 4-class | pHash cluster | acc 0.962; ECE 0.016–0.020; ~0.98 @95 % cov; AURC 0.010 | Selective prediction for brain-tumor type | **Direct competitor for the classifier-referral part of E13** (no segmentation) | https://arxiv.org/abs/2607.16317 |
| 14 | Mehta2022-CascadeUQ | 2022 | IEEE TMI | Upstream uncertainty maps as input to the downstream network | MS, BraTS-2018, ADNI | — | n/e | Uncertainty propagation in cascades | Closest prior art for seg-uncertainty → next stage (input, not gate) | https://api.crossref.org/works/10.1109/TMI.2021.3114097 |
| 15 | DeVries2018-SegQuality | 2018 | arXiv | Uncertainty map → image-level segmentation-quality prediction | n/e | — | n/e | Predict segmentation failure | Ancestor of the "is this mask trustworthy?" gate | https://arxiv.org/abs/1807.00502 |
| 16 | Jungo2019-Reliability | 2019 | MICCAI | Voxel uncertainty comparison | 2 datasets | — | n/e | Calibrated per dataset ≠ per subject | Gate must be validated per image | https://arxiv.org/abs/1907.03338 |
| 17 | Jaeger2023-FailureDetection | 2023 | ICLR | Benchmark of confidence-scoring functions | multiple | — | MSP best overall | Simple baselines win | Must beat a max-softmax gate | https://arxiv.org/abs/2211.15259 |
| 18 | Traub2024-AUGRC | 2024 | arXiv | AUGRC metric | 6 datasets | — | changes rankings on 5/6 | Better selective-prediction metric | Report alongside AURC | https://arxiv.org/abs/2407.01032 |
| 19 | Guo2017-Calibration | 2017 | ICML | Temperature scaling | — | — | — | NNs are mis-calibrated | Calibrate before gating | https://arxiv.org/abs/1706.04599 |
| 20 | Wang2019-TTA | 2019 | Neurocomputing | Test-time augmentation uncertainty | fetal brain, brain tumor | — | n/e | TTA ≥ dropout | Cheap 2D gate signal | https://arxiv.org/abs/1807.07356 |
| 21 | Filvantorkaman2025-Fusion | 2025 | arXiv | Ensemble + Grad-CAM++ vs expert regions | Figshare | 5-fold | acc 91.7 %; CAM Dice ≤ 0.88 | XAI vs mask on our data | XAI-faithfulness is not novel as a headline | https://arxiv.org/abs/2508.06891 |
| 22 | Ali2024-SAMBrain / Ludwig2025 | 2024–25 | Sci Rep / — | SAM on brain tumour MRI | BraTS etc. | — | Dice 0.56–0.68 realistic boxes; up to 0.894 with accurate boxes | SAM needs good prompts | Second-opinion idea must handle imperfect boxes | see T3 |
| 23 | UR-SAM / MedSAM-U | 2023–24 | arXiv | Prompt-jitter uncertainty for SAM | general medical | — | n/e | Training-free SAM uncertainty | Prior art for SAM uncertainty (not brain, not gating) | see T3 |
| 24 | BTReport | 2026 | MIDL | Deterministic features + LLM only for wording | BraTS (3D) | — | n/e | Grounded reports cut hallucination | Template for a 2D grounded report | https://arxiv.org/abs/2602.16006 |
| 25 | MetricsReloaded | 2024 | Nature Methods | Metric selection framework | — | — | — | Dice + boundary metric (NSD); per-image → per-patient; balanced acc / MCC + AUROC + calibration; CIs | Our reporting standard | see T4 §5 |
| 26 | CLAIM 2024 | 2024 | Radiology: AI | 44-item checklist | — | — | — | Patient-disjoint partitions, external test, 95 % CIs, code availability | Thesis appendix checklist | see T4 §5 |
| 27 | BDNeuro-MRI | 2026 | Mendeley Data | Dataset, 5,941 T1-CE, different hospital | — | — | — | Independent external test set | **External classification test** | https://data.mendeley.com/datasets/zwr4ntf94j/7 |
