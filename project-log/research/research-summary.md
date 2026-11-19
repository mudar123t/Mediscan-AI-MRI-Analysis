# Research Summary (Phase 4)

- **Date:** 2026-11-19
- **Basis:** about 116 verified sources in [literature-matrix.md](literature-matrix.md) and the T1–T4 topic files. Four key sources were independently re-checked.
- **Labels:** **[V]** = stated by a verified source (row named); **[I]** = my inference; **[U]** = unknown.
- **Novelty limits:** every novelty statement means "not found in our searches" (roughly 20 + 6 + limited queries across agents, with no Scopus or Google Scholar). It does not mean "proven new". [I]

---

## 1. Key takeaways

### A. Our semester-1 problems are well-known failure modes. Fixing them is expected, not novel.
1. **Leakage is the field's open wound.** Slice- or image-level splits inflate brain-MRI accuracy by 29–55 % (Yagis 2021) [V]. Public Kaggle merges that contain Figshare have 28.8 % near-duplicate test images and 95.5 % patient overlap in traceable test images (Vangala 2026) [V]. A 2026 re-benchmark found accuracy drops from about 95 % to about 91 % after deduplication and patient separation (Asare 2026) [V]. Many headline numbers on Figshare (Dice 0.934, acc 98.7 %) come from image-level or unstated splits (Preetha 2025, Tummala 2022) [V].
2. **The honest patient-wise reference points on our dataset are** Cheng 2015 at 91.28 % (hand-crafted features) and Deepak 2019 at 98 % (deep features + SVM). For segmentation, Avazov 2024 reaches Dice 0.93 on a patient split with a light attention U-Net [V]. **Our Dice of 0.41 is a training-recipe problem, not a hardware limit** [I, from Avazov 2024 + EXP-000].
3. **Our "ROI hurts" finding is consistent with the dataset's own paper.** A tight ROI scores 83.5 %, a dilated ROI 88.2 % (Cheng 2015), and lesion-only input is worst (Alqudah 2019) [V]. For other tasks such as glioma grading, ROI helps (Pitarch 2023) [V]. **Whether segmentation helps classification depends on mask quality and context** [I].

### B. What is already common (not a contribution on its own)
| Angle | Verdict | Evidence |
|---|---|---|
| D12 multi-task seg+cls | **Very common**; no reliable accuracy gain (95.99 vs 95.87 %) | MAG-Net 2021, Díaz-Pernas 2021, Ferreira 2024, Benzorgat 2025, BrainTumNet 2025 [V] |
| D11 mask/ROI as extra channel | **Common**, even in student theses on Figshare | Zaman 2025 [V] |
| Segmentation uncertainty (MC dropout, TTA, ensembles) for brain tumors | **Established** | QU-BraTS 2022, Wang 2019, Mehrtash 2020 [V] |
| Classifier uncertainty → refer to radiologist (brain-tumor type) | **Now exists** (Jul 2026) | Sharma 2026 [V, spot-checked] |
| Calibration (ECE, temperature scaling) | **Expected** | Guo 2017, Sharma 2026 [V] |
| Grad-CAM; XAI vs tumor mask | **Common; XAI vs mask done on Figshare** | Filvantorkaman 2025; Asare 2026 finds poor localisation [V] |
| SAM/MedSAM zero-shot on brain MRI; laptop-size SAM | **Common** | Ali 2024, Ludwig 2025, LiteMedSAM, CVPR 2024 challenge [V] |
| SAM prompt-jitter uncertainty | **Published** (general medical, not brain) | UR-SAM, MedSAM-U [V] |
| GAN/diffusion augmentation | **Common, mixed results** (sometimes no gain) | Yapici 2023, Khan 2026, Med-DDPM [V] |
| Quantised / edge classifiers | **Very common** | Islam 2024, Kanneti 2026 [V] |
| Brain-tumor report generation | **Emerging**, all 3D BraTS | BTReport 2026, Brain3D 2026 [V] |

### C. What we did not find (the openings) [I]
1. **Classification that is conditional on segmentation quality.** No verified seg→cls or multi-task paper measures how classification degrades as mask quality drops, and none routes cases by segmentation confidence. All of them trust the mask (T1 §2, T2 §2). The closest prior work uses uncertainty as an *input* to the next stage (Mehta 2022) or predicts segmentation failure without a downstream classifier (DeVries 2018) [V].
2. **Cross-model disagreement (U-Net vs MedSAM) as a gate signal** for a downstream tumor-type classifier, on 2D Figshare (T3 §4).
3. **Grounded 2D report with uncertainty statements and a number-consistency check.** BTReport is 3D BraTS (T3 §4).
4. **Clean, patient-disjoint re-benchmark of a full seg→cls pipeline, with external testing.** Vangala audits but never retrains. Asare re-benchmarks classification only (T1 §3, T2 table).

## 2. Where the field is going [I, grounded in the cited trends]
- **From accuracy to reliability.** Selective prediction, AURC/AUGRC, calibration and "evaluate uncertainty by the decisions it enables" (Jaeger 2023, Traub 2024, Maganti 2026, Sharma 2026).
- **From leaderboards to integrity.** Leakage audits, deduplication and external validation (Vangala 2026, Asare 2026, CLAIM 2024 requires patient-level partitions and external tests).
- **Foundation models as components.** MedSAM-class models as promptable second opinions, made laptop-sized (LiteMedSAM, MobileSAM).
- **Grounded generation.** Measure deterministically and let the LLM only phrase the text (BTReport 2026), with hallucination checks.
- **Human–AI teaming.** Deferral and referral policies that clinicians value (CoDoC, Nature Medicine 2023), plus uncertainty-guided correction (Hart 2026).

## 3. Datasets (T4)
- **Keep Figshare (Cheng)** as the primary dataset. No public dataset combines 2D T1-CE, binary masks, these 3 classes and patient IDs [V]. The licence is **CC BY 4.0** [V, spot-checked]: derived files may be shared with attribution and a statement of changes. Cite the dataset DOI plus Cheng 2015. **Do not cite Nickparvar/Kaggle**: it has no masks or PIDs and mixes in Figshare images [V].
- **External classification test:** BDNeuro-MRI (Bangladesh hospital, 5,941 T1-CE images, CC BY 4.0, 2026) [V]. Check pHash near-duplicates against Figshare before use [I]. A second small set (Daffodil) is optional.
- **External segmentation test (domain shift):** BraTS-MEN (meningioma) and Pandit pituitary (47 GB), both sliced to 2D T1-CE [V facts; licence/size of BraTS UNVERIFIED].
- **Do not use as external tests:** Nickparvar, SARTAJ and BRISC all contain Figshare images [V].

## 4. Evaluation standard to adopt (T2 §4, T4 §5)
- Patient-disjoint 5-fold CV using `cvind.mat`.
- Segmentation: Dice + IoU + a boundary metric (NSD and/or HD95, with an explicit empty-mask rule), per image and per patient (Metrics Reloaded).
- Classification: balanced accuracy, macro-F1, MCC, AUROC, confusion matrix, per-class recall.
- Calibration: ECE, Brier, reliability diagram, before and after temperature scaling.
- Selective prediction: risk–coverage, AURC, AUGRC, accuracy at fixed coverage, referral rate. **Always compared against a max-softmax baseline.**
- 95 % CIs by patient-level bootstrap; paired tests across folds; external test; **CLAIM 2024 checklist** as a thesis appendix.

## 5. Semester-1 references: must be fixed in the thesis (T4 §6)
- 0 of 12 were correct as cited; 9 can be corrected; 2 were not found; 1 cannot be verified.
- Serious problems: [2] wrong dataset; [3], [5], [9] wrong authors; [4] wrong title; [6], [7] wrong journal; [11] "YOLObb" not found; [12] attributed to non-existent authors (its numbers match Abdusalomov et al. 2023, *Cancers* 15:4172).
- **Action:** rebuild the reference list from `references.bib`. This is critical for an AA grade and for any publication.

## 6. Implications for Phase 5 (candidates to score; not yet decided) [I]
| ID | Candidate | Builds on | Why it may count as novel | Main risk |
|---|---|---|---|---|
| **R1** | **Segmentation-quality-aware routing** ("know when the mask can be trusted"): per image, choose (dilated) ROI view, full-image view, or refer to a radiologist, using segmentation uncertainty (MC dropout / TTA / fold-ensemble disagreement) | E13 + Cheng 2015 + EXP-000 | Not found: seg-uncertainty-driven view selection + referral for tumor type. Gives a 3-way decision instead of binary deferral | Must beat simple baselines (always-full, MSP gate, Sharma-style). Needs a decent U-Net first |
| **R2** | **Cross-model disagreement gate**: U-Net vs frozen MedSAM/LiteMedSAM (box-prompted by the U-Net) + box-jitter spread as the uncertainty signal for R1 | T3 gap 1 | Not found for brain-tumor classification gating | SAM degrades with poor boxes; extra integration effort |
| **R3** | **"When does segmentation help classification?" study**: classification accuracy as a function of mask quality (GT → degraded → predicted) and context margin (tight → dilated → full) | T1 gaps 1 & 3, Cheng 2015 | Not measured systematically for CNN pipelines on Figshare | Analytic, less "product"; best as the motivation for R1 |
| **R4** | **XAI-as-diagnostic**: measure where the classifier looks (in mask / peritumoral ring / elsewhere) with GT masks; link to correctness and routing | T2 gap 4 | Linking faithfulness to pipeline decisions is under-done | XAI-vs-mask alone is not novel (Filvantorkaman 2025) |
| **R5** | **Grounded, uncertainty-aware 2D report** with a number-consistency (hallucination) check, in the demo | T3 gap 4, BTReport | 2D single-sequence + uncertainty statements appears open | Engineering-heavy; LLM optional (a template is enough) |
| **R6** | **Integrity-first benchmark**: patient-disjoint re-evaluation + pHash overlap audit + external test on BDNeuro-MRI | T1 gap 2, T4 | Full seg→cls pipeline with external test is rare | Methodological, not "creative". Strong supporting contribution |
| **R7** | **Conformal referral**: 3-class prediction sets, where set size > 1 means refer (coverage guarantee) | T2 §1c | Moderately fresh for this task | Small add-on to R1 |

**Initial view [I]:** R1 is the headline, motivated by R3 and enabled by the groundwork (P0, A1–A3, B4, D10). R2, R4, R5 and R7 are optional extensions that strengthen novelty and the demo. R6 is woven in as the evaluation protocol. Phase 5 will score these properly with Kareem.
