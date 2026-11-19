# T2 — Uncertainty, calibration, selective prediction, XAI faithfulness, HITL

Literature review for Mediscan AI (2D U-Net → CNN glioma/meningioma/pituitary on the Figshare/Cheng dataset).
Compiled 2026-11-19. Every paper in the main table was opened in this session (arXiv abstract page, publisher page, OATML page, institutional research portal, or the Crossref API record). Numbers come only from the opened page; "n.e." means not extracted.
Anything not confirmed is in the [UNVERIFIED] section at the end.

Notes on verification sources:
- arXiv abstract pages show the abstract and metadata only. Body-only claims, such as "the paper defines the pointing game" or "the paper defines AURC", are marked *not confirmed on page* when the abstract did not show them.
- Statements marked **(inference)** are my reasoning, not claims made by any paper.

---

## 1. Verified paper table

### 1a. Uncertainty estimation, calibration, selective prediction (foundations and evaluation)

| Short name | Year | Venue | Method | Dataset | Key metrics (as reported) | Key idea | Relevance to Mediscan | Verified URL |
|---|---|---|---|---|---|---|---|---|
| Gal-MCDropout | 2016 | ICML | MC dropout as approximate Bayesian inference | n.e. (abstract page) | n.e. | Use dropout at test time to estimate model uncertainty cheaply | Cheapest segmentation-uncertainty source for a 4 GB GPU (T forward passes, no extra training) | https://arxiv.org/abs/1506.02142 |
| Lakshminarayanan-DeepEns | 2017 | NIPS | Deep ensembles | n.e. | n.e. | Ensembles are simple, parallel, well calibrated, and more uncertain on OOD inputs | Strong baseline. The 5 patient-disjoint CV folds already give a free 5-member ensemble **(inference)** | https://arxiv.org/abs/1612.01474 |
| Guo-Calibration | 2017 | ICML | Temperature scaling | n.e. | n.e. | Modern NNs are mis-calibrated, and a single temperature parameter fixes most of it | Calibrate the classifier before using its confidence for gating or reporting ECE | https://arxiv.org/abs/1706.04599 |
| Geifman-SelectiveCls | 2017 | arXiv preprint | Selective classifier with risk guarantee (SGR) | ImageNet | 2% top-5 error with 99.9% probability at almost 60% coverage | Reject inputs to meet a user-set risk with a guarantee | Formal basis for "refer to radiologist" and risk–coverage curves | https://arxiv.org/abs/1705.08500 |
| Geifman-BiasReduced | 2019 | ICLR (per arXiv comment) | Bias-reduced uncertainty estimation for classifiers | n.e. | n.e. | Uncertainty-estimation quality for selective classification | Commonly cited as the source of AURC/E-AURC. The acronyms were *not confirmed on page* | https://arxiv.org/abs/1805.08206 |
| Jaeger-FailureDetection | 2023 | ICLR (oral) | Large benchmark of confidence-scoring functions | multiple (n.e.) | Softmax-response baseline performs best overall | Failure detection, OOD, and calibration are evaluated in separate silos. The paper calls for one holistic protocol | Mediscan must compare any fancy gate against the plain max-softmax baseline | https://arxiv.org/abs/2211.15259 |
| Traub-AUGRC | 2024 | arXiv preprint (v2, Oct 2024) | New metric: AUGRC | 6 datasets, 13 confidence-scoring functions | AUGRC changes metric rankings on 5 of 6 datasets | Area under the generalized risk–coverage curve is the "average risk of undetected failures" | Report AUGRC alongside AURC for E13 | https://arxiv.org/abs/2407.01032 |
| Mehta-QU-BraTS | 2022 | MELBA vol. 1 (DOI 10.59275/j.melba.2022-354b) | Challenge score for segmentation uncertainty | BraTS 2020 (14 teams) | n.e. | The score rewards confidence on correct voxels and low confidence on incorrect ones. Uncertainty adds complementary value | Shows brain-tumor segmentation uncertainty is established (3D, multimodal), so this alone is not novel | https://arxiv.org/abs/2112.10074 |
| Wang-TTA | 2019 | Neurocomputing (DOI 10.1016/j.neucom.2019.01.103) | Test-time augmentation for aleatoric uncertainty | Fetal brain and brain tumor, 2D and 3D MRI | n.e. | TTA uncertainty beats dropout-only, and TTA reduces overconfident errors | TTA is cheap on 2D slices and directly usable as the gate signal | https://arxiv.org/abs/1807.07356 |
| Jungo-Reliability | 2019 | MICCAI | Comparison of voxel-wise uncertainty methods | 2 datasets (unnamed on page) | n.e. | Well calibrated at dataset level but miscalibrated per subject, which weakens patient-level failure detection | Key warning for E13: the gate needs an image-level aggregate that is validated per image | https://arxiv.org/abs/1907.03338 |
| Mehrtash-SegCalib | 2020 | IEEE TMI (DOI 10.1109/TMI.2020.3006437) | CE vs Dice loss, ensembles for calibrating U-Nets | Brain, heart, prostate | n.e. | Dice-loss U-Nets are overconfident. Ensembling consistently improves calibration and predicts segmentation quality | The Mediscan U-Net (Dice 0.41) is likely miscalibrated, so ensemble it or use TTA **(inference)** | https://arxiv.org/abs/1911.13273 |
| DeVries-SegQuality | 2018 | arXiv preprint | Two-stage: spatial uncertainty map → image-level quality prediction | n.e. | n.e. | Predict image-level segmentation failure so bad cases can be pulled from the pipeline | Closest conceptual ancestor of the "is this mask trustworthy?" gate | https://arxiv.org/abs/1807.00502 |

### 1b. Uncertainty in cascades, deferral, learning-to-defer (E13 novelty check)

| Short name | Year | Venue | Method | Dataset | Key metrics (as reported) | Key idea | Relevance to Mediscan | Verified URL |
|---|---|---|---|---|---|---|---|---|
| Mehta-CascadeUQ | 2022 | IEEE TMI 41(2):360–373 | Propagate upstream uncertainty maps as extra input to the downstream network | MS lesion trial data, BraTS-2018, ADNI hippocampus | n.e. | Passing uncertainty, not just a hard mask, improves downstream detection, segmentation and regression | **Closest prior art for "segmentation uncertainty feeds the next stage."** It uses uncertainty as an input feature, not as a gate or deferral. No tumor-type classification | https://oatml.cs.ox.ac.uk/publications/202109_Gal_Propagating_Uncertainty_Medical_Imaging.html (also Crossref: https://api.crossref.org/works/10.1109/TMI.2021.3114097) |
| Feiner-PipelineUQ | 2023 | Springer LNCS (DOI 10.1007/978-3-031-44336-7_1; venue not named on page) | Propagate mean and variance through cascades and attribute aleatoric uncertainty per stage | Undersampled brain and knee MR → brain volume, knee side, sex | n.e. | Joint uncertainty across stages with per-stage attribution | Shows "uncertainty in pipelines" is an active niche, and it is still not about routing | https://arxiv.org/abs/2309.16831 |
| Mozannar-L2D | 2020 | ICML | Learning to defer: joint classifier and rejector via cost-sensitive surrogate | Various | n.e. | Learn when to hand a case to a human expert | Theoretical frame for "refer to radiologist." Mediscan lacks expert labels, so it uses a simpler threshold-based deferral **(inference)** | https://arxiv.org/abs/2006.01862 |
| Dvijotham-CoDoC | 2023 | Nature Medicine 29(7):1814–1820 | Deferral model on AI confidence plus clinician opinion | UK mammography; TB triage | Blog: ~25% fewer false positives on UK mammography; ~two-thirds fewer cases needing review in simulation | Decide per case whether to accept the AI or defer to the clinician | High-impact evidence that deferral is clinically valued | https://api.crossref.org/works/10.1038/s41591-023-02437-x ; https://deepmind.google/blog/codoc-developing-reliable-ai-tools-for-healthcare/ |
| Sharma-BTTriage | 2026 (Jul) | arXiv preprint | MC dropout (T=20), entropy-thresholded deferral, temperature scaling | 7,200-image 4-class brain tumor MRI (pHash-cluster split) | Acc 0.962/0.964; macro-AUC 0.994; ECE 0.016–0.020 after TS; ~0.98 accuracy at 95% coverage; AURC 0.010–0.011 | Selective prediction for brain tumor classification is architecture-agnostic | **Direct competitor for the "classifier uncertainty → refer" part.** Classification-only, with no segmentation stage and no ROI/full-image routing | https://arxiv.org/abs/2607.16317 |
| Maganti-SegDecision | 2026 (Apr) | arXiv preprint | MC dropout vs TTA × 3 deferral policies | DRIVE, STARE, CHASE_DB1 (retinal vessels) | Up to 80% of errors removed while deferring 25% of pixels | Evaluate uncertainty by the decisions it enables. Better calibration did not mean better decisions | Supports making E13's evaluation decision-centric (risk–coverage), not ECE-centric | https://arxiv.org/abs/2604.13262 |
| Kaliaperumal-UNetDegrade | 2026 (Jul) | arXiv preprint | MC-dropout U-Net / Attention U-Net under 8 corruptions | **Synthetic** BraTS-like phantom MRI | WT Dice 0.990 clean → 0.089 severe noise; uncertainty–error Pearson r = 0.53; failure AUROC 0.843 | Uncertainty as a safety layer, with selective coverage | Shows brain-tumor segmentation failure detection is active. It uses synthetic data | https://arxiv.org/abs/2607.22727 |
| Sadib-ThyroidSelective | 2026 | Intelligence-Based Medicine 16:100517 (per arXiv) | 5-member ensemble, MI / variance / MSP, triage policy | TN5000 (internal), TN3K (external) | AUC 0.9377, ECE 0.0138 internal; AUC 0.7870, ECE 0.1899 external; MSP and variance beat MI on AURC and AUGRC | Three-way triage (act / act / radiologist review) evaluated with AURC and AUGRC | Template for E13's reporting (AURC + AUGRC + ECE + Brier + review rate) | https://arxiv.org/abs/2607.12075 |

### 1c. Conformal prediction

| Short name | Year | Venue | Method | Dataset | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|
| Angelopoulos-GentleCP | 2021 | arXiv tutorial | Split conformal prediction | Code notebooks | n.e. | Distribution-free prediction sets around any pretrained model | Cheap add-on: 3-class prediction sets, where set size > 1 triggers "refer" **(inference)** | https://arxiv.org/abs/2107.07511 |
| Angelopoulos-CRC | 2022 | arXiv preprint | Conformal risk control (monotone losses) | CV/NLP examples | n.e. | Control expected FNR and other losses | Could bound mask FNR for the U-Net stage | https://arxiv.org/abs/2208.02814 |
| Lu-FairCP | 2022 | AAAI-22 | Conformal prediction with equalized subgroup coverage | Dermatology photos | n.e. | Radiologist interviews identify CP use cases. Compares CP with epistemic uncertainty | Evidence that CP is already applied in medical imaging classification | https://arxiv.org/abs/2109.04392 |
| Dai-CRC-InstSeg | 2025 | ICIPCA 2025 (per arXiv) | CRC on Mask R-CNN / BlendMask, FNR/FDR control | n.e. on page | n.e. | Risk-controlled thresholds for medical instance segmentation | Shows CRC for medical segmentation is appearing | https://arxiv.org/abs/2504.04482 |

### 1d. XAI, faithfulness vs ground truth, shortcut learning

| Short name | Year | Venue | Method | Dataset | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|
| Selvaraju-GradCAM | 2017 (arXiv 2016) | ICCV'17; IJCV (2019 rev.) | Grad-CAM | n.e. | n.e. | Gradient-weighted class activation maps | Default XAI method | https://arxiv.org/abs/1610.02391 |
| Zhang-ExcitationBP | 2016 | ECCV 2016 (oral, short version) | Excitation backprop | COCO, VOC07, ImageNet, Flickr30k | n.e. | Top-down attention. The Pointing Game is commonly attributed to this paper (*not confirmed on page*) | Source to cite for a hit-rate / pointing-game metric against the tumor mask | https://arxiv.org/abs/1608.00507 |
| Wang-ScoreCAM | 2020 | CVPR 2020 Workshop | Score-CAM | n.e. | n.e. | Gradient-free CAM. Energy-based pointing game is commonly attributed here (*not confirmed on page*) | Source for "energy inside mask" metric | https://arxiv.org/abs/1910.01279 |
| Arun-SaliencyTrust | 2020/21 | arXiv (submitted to Radiology: AI; publication *not confirmed*) | 8 saliency methods × 4 trust criteria | 2 public radiology datasets (CXR) | AUPRC and SSIM used. All 8 methods fail at least one criterion | Saliency is often less trustworthy than baselines for localizing abnormalities | Methodological template: localization vs GT plus randomization tests | https://arxiv.org/abs/2008.02766 |
| Saporta-CXRSaliency | 2022 | Nature Machine Intelligence 4(10):867–878 | 7 saliency methods vs human benchmark | Chest X-ray, multilabel | Grad-CAM best but all worse than human; worst on small, complex lesions; confidence correlates with localization | Large-scale benchmark of saliency vs expert segmentation | Strong template and citation for quantifying Grad-CAM vs tumor masks | https://research.regionh.dk/en/publications/benchmarking-saliency-methods-for-chest-x-ray-interpretation/ |
| Jin-MSFI | 2022 | AAAI 2022 | MSFI metric; 16 heatmap methods; clinician study | Multi-modal MRI (BraTS per third-party summary; *dataset not shown on page*) | 16 methods fail clinical requirements | Mask-based, modality-aware faithfulness metric | Brain-tumor-adjacent precedent for saliency-vs-mask evaluation | https://arxiv.org/abs/2203.06487 |
| Filvantorkaman-Fusion | 2025 | arXiv preprint | MobileNetV2+DenseNet121 ensemble, Grad-CAM++, rule overlay | **Figshare (same dataset), 5-fold** | Acc 91.7%; Grad-CAM++ vs expert tumor regions Dice up to 0.88, IoU up to 0.78; radiologist Likert 4.4 / 4.0 (5 radiologists) | Grad-CAM evaluated against tumor regions on this dataset | **Direct evidence that "XAI vs GT mask on Figshare" already exists (max values only, annotation source unclear)** | https://arxiv.org/abs/2508.06891 |
| Srinivas-XAIBT | 2026 (Jan) | Frontiers in AI 8 (DOI 10.3389/frai.2025.1700214) | Custom CNN + TL, Grad-CAM, LIME | Kaggle 4-class | Best acc 95.86% (internal inconsistencies noted) | LIME shows misclassified cases attribute outside the tumor (qualitative) | Qualitative "shortcut" evidence in brain-tumor classifiers, with no quantitative mask metric | https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1700214/full |
| Asare-BTBenchmark | 2026 (Nov) | arXiv (under review, Sci. Reports) | 6 CNN/ViT/foundation models; calibration; decision analysis; external test | 7,216 public 4-class images; BDNeuro-MRI external (1,675) | Acc 94.5–95.6% → 90.3–91.7% after removing duplicates and shared patients; 36% near-duplicate rate; external 77.6–95.0% | Attribution maps "localized expert-annotated tumors poorly" yet passed the randomization test | **Very recent evidence that attribution-vs-tumor evaluation is being done and found poor** | https://arxiv.org/abs/2611.11793 |
| Geirhos-Shortcut | 2020 | Nature Machine Intelligence (DOI 10.1038/s42256-020-00257-z) | Perspective | — | — | Shortcut learning definition and taxonomy | Theory citation for the "context vs tumor" analysis | https://arxiv.org/abs/2004.07780 |
| Cheng-TRA | 2015 | PLOS ONE 10(10):e0140381 | Tumor-region augmentation (dilation) + ring partition; BoW/GLCM/IH | **Figshare (dataset paper)** | Tumor-only → augmented ROI: BoW 83.54→88.19%, GLCM 78.18→84.75%, IH 71.39→82.31%; with partition up to 91.28% | Peritumoral context helps tumor-type classification | **Explains Mediscan's finding that the tight ROI (0.74) < full image (0.80)**, and supports a "dilated ROI" variant | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381 |

### 1e. Human-in-the-loop and clinician trust

| Short name | Year | Venue | Method | Dataset | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|
| Hart-UG-HITL | 2026 (Oct) | arXiv preprint | TTA voxel uncertainty + topological filtering → simulated clinician refinement | 362 OOD glioma cases | WT Dice 0.891→0.914; WT HD95 5.82→4.76 mm; median workload 11.3% of volume (simulated oracle) | Uncertainty tells the clinician where to correct | Uncertainty-guided HITL for brain tumors is active (3D glioma) | https://arxiv.org/abs/2610.01452 |
| Zhu-MedUHIP | 2024 | arXiv preprint | Uncertainty-aware multi-hypothesis seg + iterative clinician edits | REFUGE2, LIDC-IDRI, QUBIQ | n.e. | Fewer interactions than prior interactive models | Generic HITL precedent | https://arxiv.org/abs/2408.01620 |
| Ketabi-RadXAI | 2026 (Apr) | arXiv preprint (cs.HC) | Radiologist questionnaire | — | n.e. (number of radiologists not on page) | Radiologists' requirements for explainable ML. Existing XAI reflects developers' views | Supports a small clinician-facing evaluation (e.g., a supervisor-assisted questionnaire) | https://arxiv.org/abs/2604.11700 |

**Count: 38 verified papers** (12 + 8 + 4 + 11 + 3).

---

## 2. "Is it already common?" verdicts

### E13 — uncertainty-gated routing/deferral in a seg→cls cascade
**Verdict: the components are common, but the exact combination appears fresh.** Nothing found uses segmentation uncertainty to choose between ROI and full-image classification, plus radiologist referral, for brain-tumor type. This was searched six ways; see [UNVERIFIED] for the limits of the search.
- Segmentation uncertainty for brain tumors is common: Mehta-QU-BraTS, Wang-TTA, Kaliaperumal-UNetDegrade, Hart-UG-HITL.
- Classifier-uncertainty deferral for brain-tumor type now exists: Sharma-BTTriage (Jul 2026) reports MC dropout, an entropy threshold, ECE, risk–coverage and AURC on 4-class brain MRI. **E13 must not be pitched as "first selective prediction for brain tumor classification."**
- Uncertainty propagation through cascades exists: Mehta-CascadeUQ and Feiner-PipelineUQ. They use uncertainty as an input or a joint measure, not as a router between two classifier views.
- Image-level segmentation failure prediction exists: DeVries-SegQuality and Jungo-Reliability.
- Deferral to clinicians exists: Mozannar-L2D, Dvijotham-CoDoC, Sadib-ThyroidSelective.
- **Opening (inference):** "Segmentation-quality-aware view selection" (ROI vs dilated ROI vs full image) is a three-way decision, not a binary accept/defer. Combined with the measured fact that predicted-mask ROI collapses accuracy (0.46), this is a defensible, specific contribution. Frame it as *routing* plus *deferral*, and benchmark against the Sharma-style classifier-only MSP/entropy gate.

### Calibration reporting (ECE, temperature scaling)
**Verdict: established, and expected rather than novel.** Evidence: Guo-Calibration; Sharma-BTTriage (ECE after TS); Sadib-ThyroidSelective; Asare-BTBenchmark (calibration on brain tumor). Mediscan should report it, but it is not a contribution by itself.

### Grad-CAM XAI for brain-tumor classification
**Verdict: very common (near-ubiquitous qualitative use).** Evidence: Filvantorkaman-Fusion, Srinivas-XAIBT, and Asare-BTBenchmark all use attribution maps on brain-tumor MRI. Many more appeared in searches (see UNVERIFIED).

### XAI faithfulness vs GT tumor mask
**Verdict: no longer fresh in principle, but still under-done and done poorly.** Evidence:
- General medical precedents: Arun-SaliencyTrust and Saporta-CXRSaliency (chest X-ray), Jin-MSFI (multimodal MRI).
- On the **same Figshare dataset**: Filvantorkaman-Fusion reports Grad-CAM++ vs expert regions (Dice up to 0.88, maximum values only).
- Asare-BTBenchmark (Nov 2026) reports that attribution maps localized tumors poorly.
- Most brain-tumor XAI papers remain qualitative (Srinivas-XAIBT).
- **Opening (inference):** The remaining gap is linking faithfulness to the pipeline:
  - Does mask–saliency overlap predict classifier correctness?
  - Does the full-image classifier use peritumoral context, which would explain why the ROI view hurts (Cheng-TRA)?
  - Can saliency-in-mask serve as an additional gate signal?

  Done with patient-disjoint folds and distribution-level metrics (pointing game, energy-in-mask, energy in a dilated-ring band), this is a reasonable secondary contribution. As a headline it is weak.

### Conformal prediction
**Verdict: established in general medical imaging, and moderately fresh for 2D brain-tumor-type classification on this dataset (no brain-tumor-classification CP paper verified).** Evidence: Angelopoulos-GentleCP, Angelopoulos-CRC, Lu-FairCP, Dai-CRC-InstSeg. CP sets are a cheap companion to E13, giving a coverage guarantee for the referral rule **(inference)**.

### HITL / interactive correction / clinician trust
**Verdict: active and growing (mostly 3D glioma and generic segmentation), with few brain-tumor-type trust studies verified.** Evidence: Hart-UG-HITL, Zhu-MedUHIP, Ketabi-RadXAI, Dvijotham-CoDoC. A full clinician study is out of scope for a student project. A small supervisor-assisted questionnaire on referred cases is feasible **(inference)**.

---

## 3. Gaps and openings for a 4 GB-GPU, 2D project (all inference)

1. **Three-way routing:** mask-confident → (dilated) ROI classifier; mask-uncertain → full-image classifier; both uncertain → refer. This is novel as a combination. It is cheap: 2D, with MC dropout or TTA at T≈10–20 forward passes.
2. **Dilated-ROI view (Cheng-TRA):** The 2015 dataset paper showed peritumoral context helps (+4.7 to +10.9 points). Tight ROI losing to full image is consistent with this. Add a dilated-ROI arm before concluding that "ROI is worse."
3. **Image-level gate signal design (Jungo-Reliability, DeVries-SegQuality):** Compare mean voxel entropy, entropy within the predicted mask, MC-sample Dice agreement, predicted-mask area, and classifier MSP. Report which best predicts *downstream classification error*, not segmentation Dice. This target differs from the existing seg-QC literature.
4. **XAI-as-diagnostic:** Use GT masks to measure where the full-image classifier looks (in-mask, peritumoral ring, elsewhere). Correlate this with correctness. It ties XAI to the routing story instead of being decorative.
5. **Leakage hygiene as a strength:** Asare-BTBenchmark found a 36% near-duplicate rate and accuracy drops on public 4-class sets. Mediscan's patient-disjoint Figshare folds are a methodological advantage, so state it explicitly.
6. **Conformal referral:** Split CP on calibration folds gives 3-class sets, and set size > 1 means refer. This offers a coverage guarantee at near-zero compute.

---

## 4. Recommended evaluation protocol for E13

All on the 5 patient-disjoint folds. Calibration (temperature) is fitted on a held-out part of training folds, never on test.

| Step | What | Source |
|---|---|---|
| 1 | Baselines: always-full, always-ROI(GT), always-ROI(pred), always-dilated-ROI(pred), oracle router (upper bound) | Cheng-TRA (dilated ROI); oracle is standard **(inference)** |
| 2 | Gate signals: MC dropout and TTA segmentation uncertainty (image-level aggregates); classifier MSP / entropy; 5-fold ensemble disagreement | Gal-MCDropout, Wang-TTA, Lakshminarayanan-DeepEns, Jaeger-FailureDetection (MSP baseline) |
| 3 | Calibration: ECE + Brier + reliability diagrams, before and after temperature scaling | Guo-Calibration; reported in this style by Sharma-BTTriage and Sadib-ThyroidSelective |
| 4 | Selective prediction: risk–coverage curve; AURC (and E-AURC); AUGRC; selective accuracy at fixed coverages (e.g., 95%, 90%, 80%); referral rate | Geifman-SelectiveCls; Geifman-BiasReduced (AURC acronym not confirmed on page); Traub-AUGRC; Sadib-ThyroidSelective (AURC+AUGRC); Sharma-BTTriage (95% coverage) |
| 5 | Failure-detection AUROC of each gate signal for *downstream classification error* | Jaeger-FailureDetection; Kaliaperumal-UNetDegrade (failure AUROC) |
| 6 | Decision-level reporting: don't treat calibration gains as decision gains | Maganti-SegDecision |
| 7 | Optional: conformal prediction sets (empirical coverage, mean set size) | Angelopoulos-GentleCP |
| 8 | Report per-class (glioma/meningioma/pituitary) and per-fold mean ± sd; patient-level aggregation | Jungo-Reliability (per-subject miscalibration warning) |

---

## 5. [UNVERIFIED]

Seen in search results but not opened, or opening failed. Do not cite without checking.

| Item | Reason |
|---|---|
| PeerJ CS "Interpretable four-class brain tumor MRI classification using a fine-tuned ResNet50" (peerj.com/articles/cs-3643) | HTTP 403. The search snippet says saliency validation was "not possible due to absence of voxel-level annotations" (Kaggle data) |
| Springer Discover AI "A modified dual-task DenseNet121 with Grad-CAM interpretability for brain tumor classification" (10.1007/s44163-026-02467-7) | Login redirect. The snippet says Grad-CAM was "judged subjectively" |
| "Clinicians' perspectives on the use of AI to triage MRI brain scans" (Eur J Radiol, S0720048X25000075) | HTTP 403. The snippet: 133 clinicians; 60% more confident with a heatmap |
| Research Square "Explainable, Uncertainty-Aware Ensemble Learning for Brain Tumor Classification ... Simulated Fairness Audit" (rs-11445762) | Not opened. The snippet reported 98.33% acc and ECE 0.0083 |
| Uncertainty-RIFA-Net (Springer chapter 10.1007/978-3-031-78398-2_21) | Not opened |
| BMC Med Inform Decis Mak "Interpreting deep learning models for glioma survival classification..." (s12911-023-02320-2) | Not opened. The snippet: Grad-CAM vs tumor Dice |
| Conformal Lesion Segmentation for 3D Medical Images (arXiv 2510.17897); Mossina et al., Conformal Semantic Image Segmentation (CVPRW 2024); Conformal confidence sets for biomedical segmentation (arXiv 2410.03406) | Not opened |
| Decision-aware evaluation protocol for uncertainty in medical segmentation (arXiv 2610.21616) | Not opened |
| BrainIAC (arXiv 2609.23026); Interactive DL interface for brain tumour segmentation (arXiv 2611.17983); VR interactive segmentation (Springer s10055-025-01284-0) | Not opened |
| Arun et al. publication in Radiology: AI | arXiv page says "submitted"; publication not confirmed |
| Saporta medRxiv preprint page | HTTP 403. Journal version verified via the regionh.dk portal instead |
| IEEE Xplore page for Mehta-CascadeUQ | Empty response. Verified via the OATML page and Crossref instead |
| Pointing Game definition in Zhang 2016; energy-based pointing game in Score-CAM; AURC/E-AURC definition in Geifman 2019 | Papers verified, but these specific body-text claims were not visible on the abstract pages |
| Traub-AUGRC NeurIPS 2024 acceptance | Not stated on the arXiv page; cited as arXiv |

**Search limits:** web search plus arXiv/publisher pages only. Scopus/Google Scholar were not queried, and paywalled full texts were not read. The E13 novelty verdict means "not found in six targeted searches," not "proven absent."
