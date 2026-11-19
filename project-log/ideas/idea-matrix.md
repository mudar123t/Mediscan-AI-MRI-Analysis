# Idea Matrix

- **Last updated:** 2026-11-19 (Phase 4: novelty verdicts from verified literature added; research-sourced ideas R1–R7 added)
- **Labels:** [V] VERIFIED · [I] INFERRED · [U] UNKNOWN.
- **Novelty judgements here are [I]** (my knowledge of the field). Phase 4 will check them against verified literature and may change them.
- **Numeric 1–5 scores are given in Phase 5.** Phase 3 gives qualitative assessments only.
- **Scoring axes (Phase 5):** N novelty/creative factor · F feasibility on an RTX 3050 4 GB · T time/effort (5 = cheapest) · M measurable gain over baseline · D demo/presentation impact · Th thesis value.

## Facts this analysis relies on
| Fact | Source | Label |
|---|---|---|
| Dataset: 3,064 slices, 233 patients (82 meningioma, 89 glioma, 62 pituitary patients); 1–38 slices per patient (median 13); no patient has more than one label | raw `.mat` in `dataset/dataset/data/`, read 2026-11-19 | [V] |
| Masks are **binary** ({0,1}); no tumor sub-regions; single T1-CE modality, 2D slices | raw `.mat` | [V] |
| `cvind.mat` 5 folds are patient-disjoint (0 patients span more than one fold) | raw `.mat` + `cvind.mat` | [V] |
| Patient leakage in the semester-1 "clean" lists: 43/50 test and 41/50 val images come from patients also in train | `data/pilot/splits` vs PID | [V] |
| 3,049 images are 512×512 and 15 are 256×256 | raw `.mat` | [V] |
| Semester-1 baseline (clean50 test): Dice 0.411; acc full 0.800, GT-ROI 0.740, pred-ROI 0.460 | [EXP-000](../experiments/EXP-000-semester1-checkpoint-eval.md) | [V] |
| GPU: RTX 3050 Laptop, 4,096 MiB; local torch is a CPU-only build | `nvidia-smi`, pip | [V] |

## Prerequisite (from the audit, not from future.txt)
**P0 — Patient-wise, leak-free evaluation protocol.** Use the full dataset with patient-wise 5-fold CV via `cvind.mat`, or a fixed patient-wise train/val/test split. Loaders must read split files, and outputs must be cleaned before each run. **Every other idea depends on this**, because without it no improvement claim is valid. Effort: ~1–2 days. Novelty: none (good practice), but it is the foundation of credibility. [I]

---

## Per-idea analysis (source: `future.txt`)

### Evaluation metrics (`future.txt` lines 1–18)

| ID | Restated precisely | Feasibility (HW/data) | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **E1** HD95 | 95th-percentile symmetric Hausdorff distance between predicted and GT tumor boundaries, in pixels and converted to mm (0.49 mm/px at 512² → ~0.98 mm/px at 256² [I]) | CPU only; trivial [I] | 0.5 day (use a library, e.g. MONAI/medpy, or scipy distance transforms) | Standard in BraTS-style evaluation; **not novel** [I] | Undefined when the prediction is empty: needs an explicit policy (e.g. a max-distance penalty). The pixel spacing after resize must be stated |
| **E2** ASD/ASSD | Average (symmetric) surface distance | as E1 | shares code with E1 | not novel [I] | same empty-mask issue |
| **E3** Volumetric similarity | 1 − abs(\|P\| − \|G\|) / (\|P\| + \|G\|); in 2D this is **area** similarity | trivial | <0.5 day | not novel [I] | "Volumetric" is misleading for 2D slices; call it area similarity |
| **E4** ECE (calibration) | Expected Calibration Error of classifier softmax confidence (and optionally pixel-level calibration of the U-Net) + reliability diagrams | trivial | 0.5 day (+ temperature scaling: 0.5 day) | Calibration reporting is uncommon in student brain-tumor projects and is an established method in the literature [I] | Needs a held-out calibration set; with small test sets the ECE estimate is noisy (use CV folds) |
| **E5** Pixel ROC-AUC | Threshold-free pixel-level ROC-AUC (or better, PR-AUC, since tumor pixels are ~1–5 % of the image [I]) | trivial | <0.5 day | not novel [I] | ROC-AUC is inflated by the huge number of background pixels; PR-AUC is more honest |
| **E6** Boundary F1 | Precision/recall of boundary pixels within a tolerance d (BF-score) | easy | 0.5 day | not novel [I] | Must choose the tolerance d (e.g. 2 px) and justify it |

**Verdict:** E1–E6 are **evaluation hygiene**. Together they cost about 2 days, make the thesis look rigorous and are reusable for every experiment. They do **not** address the university's novelty concern (R1). E4 (calibration) is the most valuable because it links to the uncertainty ideas (E13).

### A) Data-level improvements

| ID | Restated precisely | Feasibility | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **A1** Full dataset | Train and evaluate on all 3,064 slices / 233 patients instead of the 500-image pilot | Easy: 3,064 images at 256² ≈ 200 MB as uint8 [I]; fits RAM; U-Net training is feasible on 4 GB with batch ~8–16 at 256² [I]. GPU needs a CUDA torch install | 1 day (with P0) | none [I] | Data is ready [V]. Training time per run goes up roughly 6× vs the pilot [I]. Must regenerate from raw `.mat` (also fixes the mask-interpolation issue by using nearest-neighbour resize) |
| **A2** Stratified sampling / class balance | `future.txt` says "equal samples per class". Restated: (a) **stratified patient-wise split** so each fold has similar class proportions, and (b) handle imbalance (glioma 1,426 vs meningioma 708 slices [V]) with class-weighted loss or a balanced sampler, **not** by discarding data | trivial | 0.5 day | none [I] | "Equal samples per class" by undersampling throws away about half the glioma data. Report macro metrics as well as accuracy |
| **A3** Data augmentation | Online augmentation: rotations, flips, elastic deformation, intensity shifts/gamma, applied identically to image and mask | Easy on CPU workers | 1 day | none [I] | Flips: left/right is anatomically plausible; up/down is less so (check). Strong elastic deformation may distort pituitary anatomy. Measure with an ablation |

**Verdict:** A1+A2 are mandatory and come bundled with P0. A3 is standard. These are expected to give the **largest measurable gain** over the semester-1 baseline [I], but **zero novelty**.

### B) Segmentation model improvements

| ID | Restated precisely | Feasibility | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **B4** Longer training + scheduler + early stopping | e.g. 100+ epochs, cosine or ReduceLROnPlateau, early stopping on val Dice (patient-wise val) | feasible; time is the main cost (rough guess: minutes per epoch on the GPU at full data — **to be measured**) [I/U] | 0.5 day of code + GPU hours | none [I] | EXP-000 suggests the 5-epoch U-Net is **underfit** (lower Dice on train images than on test) [V numbers, I cause], so this is likely the single cheapest big gain |
| **B5** Focal-Dice / Tversky loss | Tversky loss with α/β to trade false positives vs false negatives (β > α penalises FN, good for small tumors); Focal-Tversky adds the γ exponent | trivial | 0.5 day + runs | not novel [I] | Needs a small α/β sweep; the gain over BCE+Dice is often modest [I] |
| **B6** Multi-class segmentation (edema / core / enhancing) | Segment tumor sub-regions instead of binary tumor | **Not feasible on this dataset**: masks are binary, with no sub-region labels [V]. Would require a different dataset (e.g. BraTS-style multi-modal 3D data [I]), whose tumor types and modalities do not match our 3-class classification task [I] | Very high (new data, 3D or multi-modal, memory limits on 4 GB) | not novel (the standard BraTS task) [I] | Breaks the link to classification. **Recommend dropping it** unless Phase 4 finds a dataset with sub-regions *and* our 3 classes [U] |
| **B7** Architecture upgrades | Attention U-Net, Residual U-Net, nnU-Net-style self-configuring pipeline | Attention/Res U-Net: feasible on 4 GB at 256² [I]. nnU-Net v2: 2D config probably feasible; Windows support and 4 GB VRAM are **[U]** and must be tested | 1–2 days each; nnU-Net 1–3 days to set up | **not novel** (widely used, as `future.txt` itself notes) [I] | As `future.txt` says, do this only after data and loss are fixed. nnU-Net would be a strong, credible baseline to compare against |

**Verdict:** B4 and B5 are cheap baseline strengthening. Of B7, an nnU-Net comparison is the most valuable as a **reference baseline** that reviewers respect. B6 should be dropped (data does not support it).

### C) ROI extraction improvements

| ID | Restated precisely | Feasibility | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **C8** Adaptive thresholding | Replace the fixed 0.3 threshold with a per-image (e.g. Otsu on the probability map, percentile) or a validation-tuned global threshold | trivial, no retraining | 0.5 day | not novel [I] | Small effect if the U-Net is well trained; risk of tuning on the test set (tune on val only) |
| **C9** Morphological refinement | Post-process masks: closing, hole filling, smoothing (largest-component filtering already exists [V]) | trivial | 0.5 day | not novel [I] | Can remove real multi-focal tumor parts. Gains are usually small after a good model [I] |

**Verdict:** cheap post-processing. Measure it as a small ablation. Not a contribution.

### D) Classification improvements

| ID | Restated precisely | Feasibility | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **D10** Better backbone | Replace SimpleCNN (99.7 % of params in one FC layer [V]) with ImageNet-pretrained ResNet-18 / EfficientNet-B0 (grey image → 3 channels, or adapt the first conv) | feasible on 4 GB at 224–256² [I] | 1 day | not novel [I] | Likely a large gain [I]. Pretrained weights need internet download once |
| **D11** Multi-input classifier | Feed **full image + ROI + mask** as separate input channels (3-channel input), so the classifier sees the tumor *and* its context | feasible, same cost as D10 | 1 day | Channel-stacking of image and mask is used in the literature; prevalence for this task is **[U]** (Phase 4) | Directly addresses the EXP-000 finding that masking out the context hurts (pred-ROI 0.46 vs full 0.80) [V numbers]. Risk: the classifier may learn to ignore the mask, so compare with ablations (with and without the mask, GT vs predicted mask) |
| **D12** Multi-task learning | One network with a shared encoder and two heads: segmentation decoder + classification head, trained jointly (weighted loss) | feasible on 4 GB with a small encoder at 256² [I] | 3–5 days | Joint seg+cls on brain MRI exists in the literature; **how common on the Figshare dataset is [U]** (Phase 4). More research-flavoured than A–C | Loss balancing between tasks; risk that one task hurts the other (report both). Interesting question: does segmentation supervision improve classification and its *localisation faithfulness*? |

**Verdict:** D10 is mandatory baseline strengthening. **D11 and D12 are the first candidates with research flavour**, and both follow directly from our own EXP-000 evidence. Their novelty must be checked in Phase 4.

### E) End-to-end system improvements

| ID | Restated precisely | Feasibility | Effort | Novelty | Risks / dependencies |
|---|---|---|---|---|---|
| **E13** Uncertainty-aware pipeline | Estimate the segmentation's uncertainty per image (e.g. MC-dropout, test-time augmentation, or a small deep ensemble; summarise as mean entropy inside the predicted ROI). Then **gate** the pipeline: if uncertainty is high, reject the ROI and fall back to full-image classification, or flag the case for a human ("refer to radiologist") | MC-dropout/TTA: cheap at inference (N forward passes) [I]. Ensembles: N× training cost | 3–5 days | Uncertainty estimation itself is well established [I]. **Uncertainty-gated routing between ROI-based and full-image classification**, plus selective prediction (risk–coverage curves), may be less common; **[U] until Phase 4** | Needs calibrated uncertainties (link E4). Evaluate with risk–coverage / AURC, accuracy at X % coverage, and how many errors are referred. Strong demo story ("the system knows when it doesn't know"). Depends on P0, A1, B4 |
| **E14** Clinical-style evaluation | Per-class sensitivity/specificity, false-negative rate, confusion matrices, CIs (bootstrap over patients), patient-level aggregation | trivial | 0.5–1 day | not novel; **expected by reviewers** [I] | "False-negative rate" needs a definition in a 3-class tumor-type problem (all images contain a tumor [V]): per-class FNR = 1 − per-class recall |

**Verdict:** E14 is mandatory reporting. **E13 is the strongest `future.txt` candidate for the university's "creative addition" requirement (R1)**: it extends the pipeline in a clinically motivated way, is feasible on our hardware, and produces demo-friendly outputs. Its novelty must be checked in Phase 4.

---

## Summary matrix (qualitative; numeric scores in Phase 5)

| ID | Idea | Category | Feasible on 4 GB laptop | Effort (days) | Novelty for R1 | Expected measurable gain | Recommendation |
|---|---|---|---|---|---|---|---|
| P0 | Patient-wise leak-free protocol | foundation | yes | 1–2 | none | makes all results valid | **Mandatory, first** |
| A1 | Full dataset | data | yes | 1 | none | high [I] | Mandatory (with P0) |
| A2 | Stratified split + class weighting | data | yes | 0.5 | none | medium on macro metrics [I] | Mandatory |
| A3 | Augmentation | data | yes | 1 | none | medium [I] | Baseline strengthening |
| B4 | Longer training + scheduler + early stop | seg | yes (GPU hours) | 0.5 | none | high (U-Net underfit) [I] | Baseline strengthening |
| B5 | Tversky / Focal-Tversky | seg | yes | 0.5 | none | low–medium [I] | Optional ablation |
| B6 | Multi-class sub-region segmentation | seg | **no data** | very high | none | — | **Dropped** (ADR-0005; binary masks only [V]) |
| B7 | Attention/Res U-Net, nnU-Net | seg | yes / nnU-Net [U] | 1–3 | none | medium [I] | nnU-Net as reference baseline |
| C8 | Adaptive threshold | ROI | yes | 0.5 | none | low [I] | Small ablation |
| C9 | Morphological refinement | ROI | yes | 0.5 | none | low [I] | Small ablation |
| D10 | Pretrained ResNet-18 / EfficientNet | cls | yes | 1 | none | high [I] | Mandatory baseline |
| D11 | Multi-input (image+ROI+mask) | cls | yes | 1 | low–medium [U] | medium–high (fixes the context loss) [I] | **Candidate** |
| D12 | Multi-task shared encoder | seg+cls | yes | 3–5 | medium [U] | medium [I] | **Candidate** |
| E1–E3, E5, E6 | Boundary / area / AUC metrics | eval | yes | ~2 total | none | — (reporting) | Do (cheap rigour) |
| E4 | Calibration (ECE) | eval | yes | 0.5–1 | low–medium [I] | — (enables E13) | Do |
| E13 | Uncertainty-gated pipeline with fallback / referral | system | yes | 3–5 | **medium–high [U]** | coverage-vs-accuracy gains [I] | **Strongest candidate from future.txt** |
| E14 | Clinical-style per-class evaluation | eval | yes | 0.5–1 | none | — (reporting) | Mandatory |

## Overall Phase 3 conclusion [I]
1. About 80 % of `future.txt` is **baseline strengthening and evaluation hygiene**: P0, A1–A3, B4–B5, B7, C8–C9, D10, E1–E6, E14. Doing these is necessary for valid, strong results (AA-level rigour), but **none of them answers the committee's novelty concern (R1)**.
2. The research-flavoured candidates are **E13** (uncertainty-gated routing / selective prediction), **D12** (multi-task) and **D11** (context-preserving multi-input). They also combine naturally: e.g. a multi-task network whose segmentation uncertainty gates how the classifier uses the ROI.
3. **B6 should be dropped** for this dataset.
4. Phase 4 must check: (a) how common E13/D11/D12 are on this exact task and dataset, (b) the angles `future.txt` does not cover (XAI and faithfulness, foundation models, cross-dataset robustness, report generation, edge deployment, human-in-the-loop demo), and (c) whether a better or additional dataset exists for external validation (Kareem is open to it).
5. Rough budget for "mandatory" items (P0, A1, A2, A3, B4, D10, E1–E6, E14) is about 8–10 working days plus GPU time [I]. This leaves room for one ambitious contribution.


---

## Phase 4 update — novelty verdicts from verified literature
Source: [research/research-summary.md](../research/research-summary.md) and the T1–T4 topic files.

| ID | Phase 3 novelty guess [I] | Phase 4 verdict | Evidence |
|---|---|---|---|
| D11 | low–medium [U] | **Common**: mask-as-channel already done on Figshare | T1 R12 (Zaman 2025); R1/R11 context studies |
| D12 | medium [U] | **Very common**; no reliable gain | T1 R5, R6, R10 (95.99 vs 95.87 %), R14, R15 |
| E13 | medium–high [U] | **Components common.** Classifier-only referral now exists (Sharma 2026). **Seg-uncertainty-driven view routing + referral not found** | T2 §2; Sharma 2026 spot-checked |
| E4 | low–medium | **Expected** (not novel) | Guo 2017; Sharma 2026 |
| A3 (GAN/diffusion variant) | — | **Common, mixed results** | T3 (Yapici 2023, Khan 2026) |
| B7 | none | Confirmed: nnU-Net is the reference recipe; a light attention U-Net reaches Dice 0.93 on a patient split | T1 R8, R22 |

## Research-sourced candidates (source: Phase 4)
| ID | Idea | Source | Novelty (Phase 4) | Feasible on 4 GB | Effort (days) [I] | Notes |
|---|---|---|---|---|---|---|
| R1 | Segmentation-quality-aware routing: (dilated) ROI / full image / refer, gated by segmentation uncertainty | E13 + Cheng 2015 + EXP-000 | **Not found** (strongest) | yes (MC dropout / TTA / fold ensemble) | 5–8 after groundwork | Must beat always-full, MSP gate, Sharma-style gate |
| R2 | U-Net vs MedSAM/LiteMedSAM disagreement + box-jitter as gate signal | T3 gap 1 | Not found for cls gating | yes (inference; precomputed embeddings) | 3–5 | SAM is weak with poor boxes (Ali 2024) |
| R3 | "When does segmentation help classification?" (mask quality × context margin study) | T1 gaps | Not measured systematically for CNN pipelines on Figshare | yes | 3–4 | Motivates R1 |
| R4 | XAI-as-diagnostic (in-mask / ring / elsewhere attribution, linked to correctness and routing) | T2 gap 4 | Partially done (Filvantorkaman 2025); linkage open | yes | 2–4 | Secondary contribution |
| R5 | Grounded uncertainty-aware 2D report with a number-consistency check | T3 gap 4 | 2D version appears open | yes (template; LLM optional) | 3–6 | Demo layer |
| R6 | Integrity-first benchmark (patient-disjoint + pHash audit + BDNeuro-MRI external test) | T1/T4 | Rare for a full pipeline | yes | 3–5 | Evaluation backbone |
| R7 | Conformal referral (prediction sets) | T2 §1c | Moderately fresh for this task | yes | 1–2 | Add-on to R1 |
