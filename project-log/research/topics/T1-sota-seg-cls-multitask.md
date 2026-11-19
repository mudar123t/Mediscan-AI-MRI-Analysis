# T1: State of the art for segmentation and classification, and joint/multi-task/segmentation-guided classification

Review date: 2026-11-19. Every row below was opened with web tools during this task. Title, first author and year were checked against the URL listed. Metrics come only from the opened page (abstract or full text, as noted). "n/e" means not extracted. Split labels: **patient** (stated patient- or subject-disjoint), **image** (stated image- or record-level), **not stated**.

Several publisher sites (MDPI, Elsevier ScienceDirect, Nature, PubMed) blocked automated fetches with 403, cookie walls or login redirects. For those papers, verification used Europe PMC REST, Semantic Scholar API, Crossref API or PMC mirrors. The URL actually opened is the one listed.

## 1. Evidence table

### 1a. Figshare/Cheng dataset: classification and segmentation

| # | Short name | Year | Venue | Method | Dataset | Split | Key metrics (as reported) | Key idea | Relevance to Mediscan | Verified URL |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | Cheng2015-TRA | 2015 | PLOS ONE | Hand-crafted features (intensity histogram, GLCM, BoW) + SVM, on tumor ROI vs **dilated** ROI vs dilated + ring partition | Figshare/Cheng, 3,064 slices / 233 pts | **patient** (5-fold, patient-disjoint) | BoW acc: tumor only 83.54% → augmented ROI 88.19% → aug + ring 91.28%. Best dilation radius about 8 px; larger radii slightly worse | **Peritumoral tissue carries class information.** A tight ROI loses it | Directly explains our "ROI-from-GT hurts" result (0.74 vs 0.80 full image). It is the original dataset paper and the source of the patient-disjoint folds we use | https://journals.plos.org/plosone/doi?id=10.1371/journal.pone.0140381 |
| R2 | Swati2019-BlockFT | 2019 | Comput Med Imaging Graph | Pretrained deep CNN (VGG family) with block-wise fine-tuning | Figshare CE-MRI | 5-fold, **not stated** in abstract | Avg acc 94.82% | Block-wise transfer learning on a small dataset | Pretraining baseline. Usable on 4 GB | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.compmedimag.2019.05.001%20OR%20DOI:10.1016/j.compbiomed.2019.103345%20OR%20DOI:10.3390/app10061999&resultType=core&format=json |
| R3 | Deepak2019-GoogLeNet | 2019 | Comput Biol Med | Pretrained GoogLeNet features + SVM/KNN | Figshare | **patient** ("patient-level five-fold CV") | Mean acc 98% | Deep features + classical classifier. Also studies reduced training data | Strong patient-level reference number to compare against | same Europe PMC URL as R2 |
| R4 | Badza2020-CNN | 2020 | Applied Sciences | Small custom CNN | Figshare | **both**: record-wise and subject-wise 10-fold | Best = record-wise + augmentation 96.56%. Subject-wise value n/e (not on opened page) | Reports both split types side by side | Primary evidence that the split type changes the number on this dataset | https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/app10061999?fields=title,authors,year,venue,abstract |
| R5 | DiazPernas2021-Multiscale | 2021 | Healthcare (MDPI) | Multiscale CNN (3 spatial scales). Segmentation and classification from one model; no skull stripping | Figshare, 3,064 / 233 pts | **not stated** in abstract | Classification acc 0.973. Dice n/e | Early joint seg+cls on exactly our data | Prior art for "joint" on our dataset | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.3390/healthcare9020153&resultType=core&format=json |
| R6 | Gupta2021-MAGNet | 2021 | arXiv 2107.12321 (CBM journal version reported by a citing source, not verified) | Multi-task attention-guided U-Net encoder-decoder; depthwise-separable convs; cls + seg | Figshare | n/e | No numbers in abstract. Claims SOTA with fewest params (5.4M per search snippet, n/e on page) | **Shared encoder-decoder, 2 tasks, on Figshare** | Direct prior art for D12 | https://arxiv.org/abs/2107.12321 |
| R7 | Tummala2022-ViTEns | 2022 | Current Oncology | Ensemble of 4 ImageNet-pretrained ViTs (B/16, B/32, L/16, L/32) | Figshare, 3,064 / 233 pts | **not stated** (70:10:20 by slice counts, 2137/314/613) | Ensemble 98.7% at 384 px; best single ViT-L/32 98.21% | ViTs on Figshare | Shows the ViT SOTA claim. ViT-L does not fit our 4 GB budget for training | https://pmc.ncbi.nlm.nih.gov/articles/PMC9600395/ |
| R8 | Avazov2024-LightUNet | 2024 | Bioengineering (MDPI) | Lightweight U-Net + spatial attention blocks, BCE loss, 512×512 | Figshare (+ BraTS) | **patient** (80/20, all slices of a patient in one subset) | Dice 0.93 ± 0.005 | Lightweight attention U-Net | **Target Dice for our U-Net on a patient split** (we have 0.41). Lightweight enough for 4 GB | https://pmc.ncbi.nlm.nih.gov/articles/PMC11727338/ |
| R9 | Preetha2025-MSAUNet | 2025 | Scientific Reports | U-Net with ImageNet-pretrained EfficientNetB4 encoder, residual attention, attention-gated skips, multi-scale convs | Figshare, 3,064 / 233 pts | **not stated** (90:10 "stratified" by class, 2758/306 images) | Dice 0.9339 | Pretrained encoder for segmentation | Pretrained-encoder U-Net recipe. The split is probably image-level, so the Dice may be optimistic | https://pmc.ncbi.nlm.nih.gov/articles/PMC11929897/ |
| R10 | Ferreira2024-MTLvsST | 2024 | J. Health Informatics (CBIS 2024) | Shared encoder → FC classification branch + segmentation decoder, vs single-task classifier | SARTAJ + Figshare (masks only from Figshare) | 5-fold CV, patient-disjointness **not stated** | MTL 95.99 ± 0.70% vs ST 95.87 ± 1.01% acc. **No meaningful gain**, lower variance, far fewer params | Direct MTL-vs-ST ablation | **Strongest evidence that D12 alone gives no accuracy gain** | https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/1296 |
| R11 | Alqudah2019-CropCmp | 2019 (arXiv 2020) | IJATCSE 8(6) | CNN on cropped vs uncropped vs segmented-lesion images, several input sizes | 3,064 T1-CE (the Figshare set; name not on page) | **not stated** | Uncropped 99.0%, cropped 98.93%, **segmented lesion 97.62%** (lowest) | Compares input context directly | Same direction as our result: lesion-only input is worst. Its absolute numbers are probably image-split | https://arxiv.org/abs/2001.08844 |
| R12 | Zaman2025-IUTthesis | 2025 | BSc thesis, IUT (grey literature) | Attention Residual U-Net → **2-channel input (MRI + predicted mask)** → InceptionV3 | Figshare | **not stated** | Dice 0.80. 2-channel cls acc 93.23% (> raw-MRI baselines) | Mask-as-extra-channel | Shows D11 (mask channel) is already done, even at student level | https://repository.iutoic-dhaka.edu/handle/123456789/2645 |

### 1b. Multi-task and segmentation-guided work on other brain MRI data

| # | Short name | Year | Venue | Method | Dataset | Split | Key metrics (as reported) | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|---|---|
| R13 | Kordnoori2024-MTL | 2024 (online 2023) | Interdisciplinary Neurosurgery | Multi-task shared structure for seg + cls of supratentorial tumors | n/e | n/e | n/e (abstract not available on the opened metadata pages) | MTL seg+cls, brain | Further D12 prior art. Metadata only | https://api.crossref.org/works?query.bibliographic=Deep+multi-task+learning+structure+segmentation+classification+supratentorial+brain+tumors&rows=3 and https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.inat.2023.101931?fields=title,authors,year,venue,abstract |
| R14 | Lv2025-BrainTumNet | 2025 | Frontiers in Oncology | CNN enc-dec + adaptive masked Transformer + CBAM. Seg then T-InceptionNet cls (described both as "multi-task" and "two-stage cascade") | 485 private cases, 3 hospitals (glioma / metastasis / meningioma), 20 slices/case | **not stated** (case counts given; patient-level not confirmed) | DSC 0.91, acc 93.4%, AUC 0.96, external acc 0.92 | Unified seg→cls with external validation | Claims end-to-end beats 2-stage by +2.7% acc, but gives no table. Internal inconsistencies noted | https://pmc.ncbi.nlm.nih.gov/articles/PMC12129765/ |
| R15 | Benzorgat2025-MTEffV2 | 2025 | Brain Sciences (MDPI) | Shared EfficientNetV2S encoder + hierarchical hybrid attention. U-shaped seg branch + multi-depth cls branch | Kaggle 4-class seg dataset, 4,237 T1-CE images | claims "patient level" 70/10/20 (source of patient IDs unexplained) | Joint: Dice 92.25%, acc 99.53%. Seg-only Dice 91.13%. Cls-only acc 98.02% | MTL with ablation | MTL gave about +1.5 pt accuracy here, but on a likely image-level Kaggle set | https://pmc.ncbi.nlm.nih.gov/articles/PMC12838544/ |
| R16 | Pitarch2023-GliomaROI | 2023 | Cancers (MDPI) | CNN glioma grading: whole brain vs tumor ROI, with and without augmentation | BraTS2020 + EGD + REMBRANDT (1,125 samples) | 75/25 + 3-fold CV. Patient-level not explicit, but the paper argues for patient-level splits | ROI beats whole brain: e.g. LGG/HGG 0.883 vs 0.805 (with aug) | ROI **helps** for *grading* | Counter-evidence: ROI cropping helps in some tasks. Whether ROI helps depends on the task and on mask quality | https://pmc.ncbi.nlm.nih.gov/articles/PMC10341156/ |
| R17 | Fateh2025-BRISC | 2025 (Sci Data 2026) | arXiv 2506.14318 / Scientific Data 13:361 | New dataset: 6,000 T1-CE, radiologist masks, 4 classes incl. no-tumor | BRISC | n/e | Benchmarks mentioned, no numbers on page | Second public dataset with both masks and labels | **External test set** option for generalization | https://arxiv.org/abs/2506.14318 |

### 1c. Evaluation integrity (split and leakage)

| # | Short name | Year | Venue | Method | Dataset | Split | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|---|---|
| R18 | Yagis2021-Leakage | 2021 | Scientific Reports | Slice-level vs subject-level CV, three 2D CNNs | OASIS, ADNI, PPMI, local PD | both | Slice-level inflation of about 30% (OASIS), 29% (ADNI), 48% (PPMI), 55% (local). Random labels: about 96% slice-level vs about 50% subject-level | Slice-level splitting inflates accuracy massively on small data | Justifies our patient-disjoint protocol (semester-1 leakage) | https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41598-021-01681-w?fields=title,authors,year,venue,abstract |
| R19 | Vangala2026-Leak | 2026 | arXiv 2610.00421 (preprint) | 3-layer contamination audit (duplicate / patient / source-label). 9 architectures | Nickparvar (C1), SARTAJ (C2), Navoneel (C3); figshare traced inside C1 | audit | 28.8% of C1 test has a near-twin in train. 22.3% of C2 test byte-identical. 95.5% of traceable (201) C1 test images share a patient with train. Header metadata alone gives 0.959 balanced acc. Deduplication barely changes accuracy | "Stable scores ≠ benchmark integrity." It recovers 233 patients in the figshare data | Very recent and directly on our data family. It claims "every published split on this benchmark is image-level", which **R1/R3 contradict** (both patient-level). It never trains on a patient-disjoint split, which leaves room for us | https://arxiv.org/abs/2610.00421 and https://arxiv.org/html/2610.00421 |

### 1d. Foundational segmentation architectures and pretraining

| # | Short name | Year | Venue | Method | Dataset | Split | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|---|---|
| R20 | Ronneberger2015-UNet | 2015 | MICCAI (arXiv 1505.04597) | U-Net | EM, cell tracking | n/a | n/e | Enc-dec + skips + heavy augmentation for few labels | Our base segmenter | https://arxiv.org/abs/1505.04597 |
| R21 | Oktay2018-AttnUNet | 2018 | MIDL (arXiv 1804.03999) | Attention gates in U-Net skips | Abdominal CT | n/a | n/e | Learn where to look | Cheap upgrade (R8 and R9 use attention) | https://arxiv.org/abs/1804.03999 |
| R22 | Isensee2021-nnUNet | 2021 (online 2020) | Nature Methods | Self-configuring U-Net pipeline | 19 public competitions, 49 tasks (per arXiv page) | n/a | Outperforms most specialized pipelines (qualitative) | **Configuration and training recipe matter more than architecture** | Strong 2D baseline. Its Dice+CE loss and augmentation recipe are reusable on 4 GB with a small batch | https://arxiv.org/abs/1904.08128 and https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41592-020-01008-z?fields=title,authors,year,venue,abstract |
| R23 | Chen2021-TransUNet | 2021 | arXiv 2102.04306 | CNN + Transformer encoder, U-Net decoder | Synapse etc. | n/a | n/e | Global context via Transformer | Heavy for 4 GB. Cite as SOTA family only | https://arxiv.org/abs/2102.04306 |
| R24 | Cao2021-SwinUnet | 2021 | arXiv 2105.05537 | Pure Swin-Transformer U-shape | Synapse, ACDC | n/a | n/e | Shifted-window attention | Same as R23 | https://arxiv.org/abs/2105.05537 |
| R25 | Mei2022-RadImageNet | 2022 | Radiology: AI 4(5) | Radiology-image pretraining dataset and models | RadImageNet | n/a | n/e (abstract not on opened page) | Medical-domain pretraining vs ImageNet | Alternative pretrained init for the classifier or encoder | https://api.crossref.org/works/10.1148/ryai.210315 |

Count: **25 verified** entries. R12 is grey literature. R13 and R25 are metadata-only (abstract not seen).

## 2. "Is it already common?" verdicts

### D11: multi-input classifier (image + ROI + mask channels)
**Verdict: common. Not novel as a stand-alone contribution.**
- Mask-as-extra-channel on this exact dataset is already done in student work (R12: MRI + predicted mask, 2-channel, 93.23%).
- Context-vs-lesion input comparisons are old (R1 2015, R11 2019). Both show that lesion-only input is worst and that some surrounding context helps.
- What D11 is still good for: as an engineering fix for our 0.46 ROI-from-predicted-mask failure, it is well motivated by R1 and R11. It should be reported as a baseline or ablation, not as the novelty.

### D12: multi-task shared-encoder seg + cls
**Verdict: very common. Not novel.**
- On Figshare specifically: R5 (2021), R6 MAG-Net (2021), R10 (2024), and R12 builds a seg→cls variant.
- On other brain data: R13, R14, R15. A 2026 Sci Rep multi-task paper also exists (see UNVERIFIED).
- Evidence for gains is weak or mixed. R10 found no accuracy gain (95.99 vs 95.87), only lower variance and fewer parameters. R15 found about +1.5 pt on a probably leaky Kaggle split. R14 claims +2.7% with no table.
- D12 could be a solid *engineering* backbone, especially because one shared encoder fits 4 GB better than two networks. It will not count as a contribution.

### ROI-based classification (crop or mask to the tumor, then classify)
**Verdict: common and old. The literature is split on whether it helps.**
- It hurts or loses information on Figshare type classification: R1 (tight ROI 83.5% vs dilated 88.2%) and R11 (segmented lesion lowest).
- It helps for glioma grading with good masks and multi-sequence data: R16.
- Our own data point (full 0.80 > GT-ROI 0.74 > predicted-ROI 0.46) agrees with R1 and R11. **A tight mask removes peritumoral context, and a bad mask compounds the error.**

### Segmentation-guided / attention-guided classification
**Verdict: common in the form "use the mask or attention to help classify". No verified paper found that makes the guidance conditional on segmentation quality or uncertainty.**
- Attention-guided multi-task: R6. Seg-feature-to-cls cascades: R12, R14.
- In every verified paper the pipeline always trusts the mask. None measures how classification degrades as segmentation quality drops (predicted vs GT mask), and none routes cases based on seg confidence. This search was limited (about 20 queries), so this is an absence of evidence, not proof of novelty. T-topics on uncertainty should confirm it.

## 3. Gaps / openings (my inference, not established by the papers)

1. **Seg-quality-aware fallback (E13) looks like the freshest angle.** All seg→cls and MTL papers (R5, R6, R10, R12, R14, R15) assume the mask is usable. Our own measured collapse from 0.74 (GT ROI) to 0.46 (predicted ROI) is the exact failure mode that a gate would address. Framing: "when does segmentation help classification, and can the system know when it doesn't?" Report accuracy-vs-coverage and referral curves on patient-disjoint folds.
2. **Rigorous patient-disjoint re-benchmark on Figshare.** R19 (2026) audits leakage but never retrains on patient-disjoint splits. R4 shows the record-wise number is higher. R18 shows inflation can be 30 to 55% on brain MRI. We already have the official patient-disjoint cvind folds. A clean "leaky vs patient-disjoint" table for a seg+cls pipeline is cheap and useful context (not the main novelty).
3. **Context-margin study for 2-stage pipelines.** R1 found 8 px dilation optimal for hand-crafted features (2015). Nobody we verified re-measured the optimal context margin, or dilation as a function of mask uncertainty, for CNNs fed predicted masks. This is a small, 4 GB-friendly ablation that could feed E13: dilate more when the mask is uncertain.
4. **External generalization on BRISC (R17).** It is T1-CE with masks and the same 3 tumor classes plus no-tumor. Testing a Figshare-trained pipeline on BRISC, with a check for overlap since BRISC is compiled from public sets, gives an out-of-distribution test that most Figshare papers lack.
5. **Training recipe for 4 GB.** Use a pretrained lightweight encoder (EfficientNet/ResNet-18/34 class, as in R9, R2, R3), attention gates (R8, R21), Dice+CE loss and nnU-Net-style augmentation (R22), mixed precision and 256 px inputs. R8 reaches Dice 0.93 on a patient split with a lightweight U-Net, so our 0.41 is an underfitting and recipe problem, not a capacity limit. Reaching about 0.85 or more Dice is a prerequisite before any gating claim is credible.
6. ViT ensembles (R7) and TransUNet/Swin-Unet (R23, R24) are not needed. They are heavy for 4 GB and their gains on Figshare are reported on unclear splits.

## 4. Training strategies that matter on small 2D datasets (from verified rows)
- **Transfer learning / pretrained encoders:** R2 (block-wise fine-tuning), R3 (GoogLeNet features, patient-level 98%), R9 (EfficientNetB4 encoder for seg). RadImageNet (R25) is a medical-domain alternative to ImageNet init.
- **Augmentation:** R4 got its best result with augmentation (record-wise). R16 shows augmentation gains for both whole-brain and ROI inputs. R20 (U-Net) was designed around heavy augmentation for few labels.
- **Loss / pipeline configuration:** R22 (nnU-Net) shows that a well-configured standard U-Net beats most bespoke architectures. R8 shows that attention plus a light U-Net is enough for about 0.93 Dice.
- **Context:** R1, the dilated ROI, is effectively a "context augmentation".
- **Evaluation:** patient-disjoint CV (R1, R3, R4, R8, R18).

## 5. [UNVERIFIED]
| Item | Reason |
|---|---|
| Badža & Barjaktarović 2020 subject-wise accuracy "88.48%" | Seen only in a search-engine summary (quoting R19's related-work table). Not on the page I opened (S2 abstract shows only 96.56% record-wise). Check the full text before citing. |
| MAG-Net journal version, Comput Biol Med 134:104459 (2021) | Reported by a citing source in search results. Not opened. |
| Singh, Prabha, Abdulla 2026, "Multi-task deep learning framework for brain tumor classification and segmentation using ensemble feature extraction", Sci Rep, DOI 10.1038/s41598-026-75592-z | Title, authors and year seen only in a Crossref search listing. The Nature page redirected to login. No abstract or numbers. |
| Rhanoui et al. 2024 (MDPI), multi-task seg+cls across cancers incl. brain (about 90% brain cls) | Search snippet only. Not opened. |
| ResViT self-supervised (arXiv 2411.12874), 98.53% on Figshare | Search snippet only. |
| PBVit, IEEE Access 2025, 95.8% Figshare | Search snippet only. |
| JMIR AI 2025 ViT-B/16 cross-domain Figshare study | Search snippet only. |
| Alnowami et al. (Cardiff ORCA), whole-image rather than segmented-tumor training | Search snippet only. |
| Pacal 2026, OBM Neurobiology, ViT families | Opened and verified, but it uses a Mendeley dataset, not Figshare. Dropped as off-topic. It does show that authors increasingly disclose "image-level benchmark only". |
| Figshare dataset record itself (10.6084/m9.figshare.1512427) | figshare.com returned 403 and Crossref returned 404 for that DOI. Dataset facts are taken from the project's own verified data and from R1/R7/R9 pages. |
