# T3 — Medical foundation models, generative/synthetic data, edge deployment, report generation

Literature review for Mediscan AI (U-Net segmentation, then a CNN that classifies glioma, meningioma or pituitary tumour, on the Figshare dataset by Cheng et al.).
Date: 2026-11-19. Hardware assumed: an RTX 3050 laptop GPU with 4 GB of VRAM, with no cloud.

**Verification rule.** Every row in the tables below was opened in this session at the URL shown. I confirmed the title, first author and year there. A number appears only if it was on the page I opened. Where a number did not appear there, the cell says "not extracted". Anything labelled **(inference)** is my own estimate, not a published figure.

---

## 1. Verified items

### 1a. Segmentation foundation models (SAM family)

| Short name | Year | Venue | Method | Data | Size/compute (as reported) | Key metrics | Key idea | Relevance + feasibility on 4 GB | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| SAM (Kirillov) | 2023 | arXiv 2304.02643 (ICCV 2023, not checked) | Promptable segmentation: ViT image encoder plus prompt encoder and mask decoder | SA-1B: about 11M images and about 1B masks | not extracted | not extracted | Zero-shot segmentation from point or box prompts | This is the base model. Zero-shot inference with a box prompt taken from our U-Net should fit in 4 GB with the smallest (ViT-B) checkpoint (inference) | https://arxiv.org/abs/2304.02643 |
| MedSAM (Ma) | 2023/2024 | Nature Communications 15, 654 (2024) | SAM fine-tuned on medical image–mask pairs, with box prompts | 1,570,263 image–mask pairs, 10 modalities, more than 30 cancer types | Training used 20 A100 80GB GPUs (5 nodes × 4), per the GitHub README | 86 internal and 60 external validation tasks; per-task numbers not extracted | A medical SAM that takes box prompts | High relevance. Box-prompted inference on 2D slices should work on 4 GB (inference). Full fine-tuning on 4 GB is not feasible. Fine-tuning only the decoder may be (inference) | https://arxiv.org/abs/2304.12306 ; https://github.com/bowang-lab/MedSAM |
| LiteMedSAM (MedSAM repo branch) | 2024 | GitHub (bowang-lab) | TinyViT encoder distilled from MedSAM's image embeddings, then the whole pipeline fine-tuned | MedSAM data | Training needs "about 10GB GPU memory with a batch size of 4". README news says "10x faster than MedSAM" | not extracted | Lightweight MedSAM built for fast training and inference | **This is the best fit for 4 GB.** Batch size 1–2 with mixed precision should bring fine-tuning under 4 GB (inference, untested). CPU inference is plausible | https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM |
| Efficient MedSAMs challenge (Ma) | 2024 | arXiv 2412.16085 (summary of the CVPR 2024 "MedSAM on Laptop" challenge) | Challenge to find lightweight medical SAMs | Data from more than 20 institutions, 9 modalities | Parameters and speed not extracted from the abstract | not extracted | Shows that **on-laptop medical SAM is already an established benchmark track** | Evidence that running "SAM on a laptop" is not new by itself | https://arxiv.org/abs/2412.16085 |
| SAM QAT (Lu) | 2024 | arXiv 2412.11186 (challenge entry) | Quantisation-aware training of a medical SAM, deployed with OpenVINO | Challenge data | Speed-up not extracted | not extracted | INT8 medical SAM for CPU inference | A recipe for CPU or edge deployment of the second-opinion segmenter | https://arxiv.org/abs/2412.11186 |
| SAM-Med2D (Cheng J.) | 2023 | arXiv 2308.16184 | SAM fine-tuned on a large 2D medical corpus | About 4.6M images and 19.7M masks; tested on 9 MICCAI 2023 datasets | not extracted | not extracted | 2D-specific medical SAM | Relevant because our data are 2D slices. VRAM not extracted. Inference likely fits with ViT-B (inference) | https://arxiv.org/abs/2308.16184 |
| SAM 2 (Ravi) | 2024 | arXiv 2408.00714 | Streaming-memory SAM for images and video | SA-V video dataset (size not extracted) | About 6× faster than SAM on images | 3× fewer interactions on video | Memory-based propagation | Not needed for 2D slices. It could propagate masks across neighbouring slices only if we had volumes, which Figshare does not give us | https://arxiv.org/abs/2408.00714 |
| MedSAM2 (Ma) | 2025 | arXiv 2504.03600 | SAM 2 fine-tuned for 3D medical images and video | More than 455k 3D image–mask pairs and 76k video frames | not extracted | More than 85% reduction in annotation cost (user study) | 3D and video medical SAM | Low fit for our 2D-only data | https://arxiv.org/abs/2504.03600 |
| MobileSAM (Zhang C.) | 2023 | arXiv 2306.14289 | Distils SAM's image encoder into a small encoder | SA-1B | More than 60× smaller than SAM; about 10 ms per image on one GPU (8 ms encoder + 4 ms decoder) | 5× faster and 7× smaller than FastSAM | Decoupled distillation | Fits easily on 4 GB or CPU. Not medical, so it would need medical fine-tuning | https://arxiv.org/abs/2306.14289 |
| SAMed (Zhang K.) | 2023 | arXiv 2304.13785 | SAM with LoRA-style fine-tuning that updates only a small fraction of parameters | Synapse multi-organ CT | Fraction updated not extracted | DSC 81.88, HD 20.64 on Synapse | Parameter-efficient adaptation of SAM | The template for LoRA fine-tuning on 4 GB with batch 1 and fp16 (inference) | https://arxiv.org/abs/2304.13785 |
| BiomedParse (Zhao) | 2024 | arXiv 2405.12971; Nat Methods 22, 166–176 (2025) | Text-prompted joint segmentation, detection and recognition | More than 6M image–mask–text triples, 9 modalities, 82 object types | not extracted | State-of-the-art claimed (numbers not extracted) | Segments from a **text** prompt, with no box needed | Interesting as a third opinion. VRAM not extracted. The licence was reported as non-commercial in search snippets, which I did not verify | https://arxiv.org/abs/2405.12971 |

### 1b. SAM on brain tumour MRI, and SAM with uncertainty

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance + feasibility | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| SAM on brain-tumour MRI (Ali) | 2024 | Scientific Reports | Zero-shot SAM with point and box prompts | TCGA, BraTS | — | Box prompt: Dice 0.68 (TCGA) and 0.56 (BraTS); IoU 0.89 and 0.65 | Box prompts work best. Results vary with tumour size and shape | Direct baseline for using a "SAM second opinion" on brain MRI | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41598-024-72342-x&resultType=core&format=json (Nature page blocked; DOI 10.1038/s41598-024-72342-x) |
| Brain-tumour SAM benchmark (Ludwig) | 2025 | arXiv 2506.20786 | Zero-shot SAM, SAM 2, MedSAM and SAM-Med3D versus nnU-Net at different prompt qualities | BraTS 2023 adult glioma and paediatric data | — | SAM up to Dice 0.894 and SAM 2 up to 0.893 with *very accurate* boxes, above nnU-Net. Weak with realistic prompts | **Prompt quality is the bottleneck** | Key evidence: SAM's value depends on box quality. Our U-Net boxes will be imperfect | https://arxiv.org/abs/2506.20786 |
| UR-SAM (Zhang Y.) | 2023 | arXiv 2311.10529 | Perturbs auto-generated box prompts, uses the spread of the resulting masks as uncertainty, then rectifies the segmentation. No training | Two 3D datasets, 35 organs | Training-free | Dice up to +10.7% and +13.8% | **Uncertainty from prompt perturbation** | Very close to our "uncertainty-aware" idea. It is inference-only, so it fits 4 GB. It means prompt-jitter uncertainty is *not* new | https://arxiv.org/abs/2311.10529 |
| MedSAM-U (Zhou) | 2024 | arXiv 2409.00924 | Multi-prompt adapter plus uncertainty-guided automatic prompting on MedSAM | 5 modality datasets | Adapter training (compute not extracted) | +1.7% to +20.5% over MedSAM | Uncertainty-guided prompts | Same area as UR-SAM. Confirms the idea is being actively worked on | https://arxiv.org/abs/2409.00924 |

### 1c. Vision-language and general medical models

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance + feasibility | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| BiomedCLIP (Zhang S.) | 2023 | arXiv 2303.00915 | CLIP-style contrastive image–text pre-training | PMC-15M: 15M pairs from 4.4M articles | not extracted | State of the art on retrieval, classification and VQA (numbers not extracted) | Biomedical image–text embedding | Could provide a frozen feature extractor or zero-shot text-prompt classification. A ViT-B-sized model should run on 4 GB for inference and linear probing (inference) | https://arxiv.org/abs/2303.00915 |
| LLaVA-Med (Li C.) | 2023 | arXiv 2306.00890 | Vision-language assistant trained by curriculum on PMC figure captions and GPT-4 instructions | PMC (size not extracted) | 8 × A100, under 15 hours | not extracted | Biomedical multimodal chat | Training is out of reach locally. Inference on a 7B-class model needs 4-bit quantisation and is borderline on 4 GB (inference) | https://arxiv.org/abs/2306.00890 |
| MedGemma (Sellergren) | 2025 | arXiv 2507.05201; journal ref Nature Medicine 32 (2026) | Gemma 3 medical variants plus the MedSigLIP encoder | not extracted | **4B and 27B** parameters | +2.6–10% on medical multimodal QA; +15.5–18.1% on chest X-ray classification | Open medical VLM that can run locally | **The 4B model is the most realistic local LLM for report writing.** In 4-bit it needs roughly 3 GB, or it can run on CPU (inference, untested) | https://arxiv.org/abs/2507.05201 |

### 1d. Synthetic data (GAN and diffusion)

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance + feasibility | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| Med-DDPM (Dorjsembe) | 2023/2024 | IEEE JBHI (DOI 10.1109/JBHI.2024.3385504) | 3D diffusion model conditioned on a semantic mask | Not extracted | not extracted | Segmentation Dice: synthetic only 0.6207, real 0.6531, real + synthetic 0.6675 | Mask-conditioned synthesis | Shows a **small** gain from adding synthetic data. 3D, so not directly usable. A 2D version is feasible to *train* only in the cloud (inference) | https://arxiv.org/abs/2305.18453 |
| CycleGAN on Figshare (Yapici) | 2023 | Computers, Materials & Continua 74(3) | CycleGAN augmentation | **Figshare brain dataset** | not extracted | Best accuracy without → with synthetic data: glioma 0.9633→0.9968, meningioma 0.9569→0.9920, pituitary 0.9904→0.9952 | GAN augmentation for Figshare classification | Shows **GAN augmentation on Figshare is already published**. The study split was not checked, so patient-level leakage is possible | https://www.sciopen.com/article/10.32604/cmc.2023.035584 |
| Pix2Pix augmentation (Onakpojeruo) | 2024 | Brain Communications 6(6) fcae372 | Pix2Pix GAN generating 10,000 synthetic images, then a CNN classifier | Kaggle four-class dataset (3,264 slices), not Figshare | not extracted | CDCNN 72% on original data and 86% on synthetic data. Different test sets, so this is not a controlled comparison | GAN augmentation | Example of a weak evaluation design to avoid | https://pmc.ncbi.nlm.nih.gov/articles/PMC11528519/ |
| DCGAN negative result (Khan) | 2026 | arXiv 2609.28508 | Swin-T trained on real data versus real + 500 DCGAN images per class | 7,200 images, four classes | — | 96% accuracy in both cases; AUC 0.987 → 0.982; FID 209–314 | **Synthetic augmentation did not help** | Important counter-evidence. Generative augmentation is no reliable win | https://arxiv.org/abs/2609.28508 |

### 1e. Lightweight and edge models

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance + feasibility | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| Deep Fusion plus 8-bit quantisation (Islam) | 2024 | arXiv 2406.19690 | Fusion of ResNet152V2 and VGG16 features, then 8-bit quantisation | **Figshare** and Kaggle | 2.8M trainable parameters; 289.45 MB → 73.881 MB after quantisation | 98.36% on Figshare | Edge-oriented quantised classifier | Shows quantised Figshare classifiers are already published | https://arxiv.org/abs/2406.19690 |
| Quantised MobileNetV2 (Kanneti) | 2026 | arXiv 2605.19207 | Float16 post-training quantisation with TFLite on MobileNetV2 | Unnamed four-class brain MRI dataset | 35.34 MB → 5.76 MB | 82.37% (quantised) versus 82.20% (FP32) | Low-resource deployment | Shows the size and accuracy trade-off is trivial. Little novelty left here | https://arxiv.org/abs/2605.19207 |

### 1f. Report generation and hallucination

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance + feasibility | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| BTReport (Heras Rivera) | 2026 | MIDL 2026 (accepted), arXiv 2602.16006 | Features extracted deterministically from images, then an LLM used **only** to structure and format the text | BraTS plus a BTReport-BraTS synthetic report set; compared with clinical reports | LLM not named on the page | Numbers not extracted. Features predict survival and IDH status | Separate measurement from writing to cut hallucination | **Closest prior work to our "structured report from mask, class and measurements" idea.** It is 3D multiparametric. Ours would be 2D and single-sequence | https://arxiv.org/abs/2602.16006 |
| Brain3D (Barone) | 2026 | arXiv 2602.22098 | 2D medical encoder inflated to 3D, aligned to a causal LLM in three stages including LoRA | 468 BraTS subjects | not extracted | Pathology F1 0.951 versus 0.413 for a 2D baseline | End-to-end VLM report generation | Training needs more than 4 GB (inference). Shows end-to-end brain MRI report VLMs exist | https://arxiv.org/abs/2602.22098 |
| ReXTrust (Hardy) | 2024 | arXiv 2412.15264 | Finding-level hallucination detector built on VLM hidden states | MIMIC-CXR subset | not extracted | AUROC 0.8751 on all findings, 0.8963 on clinically significant findings | Hallucination detection is now its own research problem | Supports the risk argument. Chest X-ray only | https://arxiv.org/abs/2412.15264 |

### 1g. Dataset reference

| Short name | Year | Venue | Method | Data | Size/compute | Key metrics | Key idea | Relevance | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| Figshare dataset paper (Cheng) | 2015 | PLOS ONE | Tumour-region augmentation and ring partition, with BoW, GLCM and histogram features | 3,064 T1-CE slices, 233 patients (708 meningioma, 1,426 glioma, 930 pituitary) | — | Best 91.28% (BoW + ring partition) | Using the tumour region and its surroundings helps | Our dataset. Its "tumour region augmentation" idea matches our ROI-classification result: 0.46 with the bare ROI, which suggests context matters | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381 |

**Verified count: 27 items** (26 papers plus the LiteMedSAM GitHub branch).

---

## 2. "Is it already common?" verdicts

| Angle | Verdict | Evidence rows |
|---|---|---|
| SAM or MedSAM zero-shot on brain tumour MRI | **Common.** Several evaluations exist, mostly on BraTS | Ali 2024, Ludwig 2025, MedSAM |
| SAM or MedSAM specifically on the *Figshare* 2D dataset | **Not common in peer-reviewed work, as far as I found.** One unverified student project (BrainHack) fine-tunes the SAM decoder on it | (see UNVERIFIED) |
| Lightweight or laptop medical SAM | **Common and benchmarked.** The CVPR 2024 challenge covered it, and LiteMedSAM, MobileSAM and QAT already exist | Efficient MedSAMs, LiteMedSAM, MobileSAM, Lu 2024 |
| Uncertainty from SAM prompt perturbation | **Already published** in general medical imaging (organs). Not found for brain tumour 2D or for a 2-stage seg→cls pipeline | UR-SAM, MedSAM-U |
| Using SAM versus U-Net disagreement as an uncertainty and triage signal for a downstream classifier | **No paper found** (absence of evidence after limited search, not proof) | — |
| Vision-language models (BiomedCLIP, MedGemma) for brain-tumour type classification | Models exist and are general. Brain-tumour-specific use on Figshare was **not checked** | BiomedCLIP, MedGemma |
| GAN augmentation for Figshare classification | **Common.** Mixed evidence, sometimes with weak evaluation | Yapici 2023, Onakpojeruo 2024, Khan 2026 |
| Mask-conditioned diffusion for brain tumour MRI | **Established in 3D (BraTS)**. Less so for 2D T1-CE Figshare | Med-DDPM |
| Edge or quantised brain tumour classifiers | **Very common.** Many papers on MobileNet, quantisation and TFLite | Islam 2024, Kanneti 2026 |
| Brain-tumour MRI report generation | **Emerging (2026).** A few papers, all on 3D BraTS | BTReport, Brain3D |
| Hallucination risk in LLM reports | Well documented, mostly for chest X-ray | ReXTrust |

---

## 3. Feasibility and cost on local 4 GB versus cloud (all **inference**, not measured)

| Task | Local, 4 GB RTX 3050 | Cloud needed? | Effort estimate |
|---|---|---|---|
| MedSAM or SAM-Med2D (ViT-B) inference with U-Net box prompts on 3,064 slices | Likely fine in fp16 at batch 1. Expect roughly seconds per slice, so under an hour for the whole set | No | 1–3 days to integrate |
| LiteMedSAM or MobileSAM inference | Easy, and CPU works too | No | 1 day |
| LiteMedSAM fine-tuning on Figshare | README says 10 GB at batch 4. Batch 1 plus AMP and gradient accumulation might fit in 4 GB, but this is unconfirmed | Optional: free Colab or Kaggle T4 (16 GB) removes the risk | 3–5 days |
| LoRA or decoder-only fine-tuning of the MedSAM ViT-B | Possible with a frozen encoder and pre-computed embeddings (store the encoder outputs once, then train only the decoder) | No | 3–5 days |
| UR-SAM-style prompt-jitter uncertainty (N perturbed boxes per slice) | Inference only, about N× the cost of single inference. Pre-computing the image embedding once makes the perturbations cheap, because only the decoder is re-run | No | 2–3 days |
| BiomedCLIP features with linear probe or zero-shot classification | Fine | No | 1–2 days |
| 2D GAN (DCGAN or pix2pix) at 256² | Feasible locally but slow, roughly 10–30 hours | Optional | 1 week |
| 2D mask-conditioned diffusion (DDPM or ControlNet-style) at 256² | Training is impractical locally: slow, and the batch size is tiny | **Yes.** Roughly 20–60 T4 GPU-hours on Kaggle (30 h/week free) or Colab | 2–3 weeks; gain uncertain (Khan 2026) |
| ONNX or INT8 export of U-Net plus CNN, with CPU latency benchmark | Easy | No | 1–2 days. Low novelty |
| Template-based structured report (no LLM) from mask area, location, class and confidence | Trivial | No | 1–2 days |
| Local LLM writing (MedGemma-4B or similar at 4-bit) | Borderline on GPU; on CPU, roughly 5–20 s per report | No | 3–5 days |
| API LLM (cloud) for the text | — | Yes; small cost per report, plus privacy concerns | 1–2 days |

---

## 4. Gaps and openings (inference)

1. **A second opinion that also says how sure it is.** Feed U-Net-derived boxes to a frozen MedSAM or LiteMedSAM. Use the *disagreement* (Dice between U-Net and SAM, plus the spread of SAM masks under box jitter, as in UR-SAM) as a per-case uncertainty score. Then (a) flag cases for review, (b) gate whether the classifier sees the ROI or the full image, and (c) report risk–coverage curves. UR-SAM and MedSAM-U do prompt-jitter uncertainty, but I found none that uses **cross-model disagreement to gate a downstream tumour-type classifier**, or that does it on Figshare 2D. This fits in 4 GB.
2. **Prompt-quality analysis on our own data.** Ludwig 2025 shows SAM's results collapse without accurate boxes. Measuring how SAM's Dice changes as U-Net box error grows, on Figshare, would be a modest but clean contribution.
3. **Fixing the 0.46 ROI-only classifier.** Cheng 2015 already showed that tumour-surrounding context helps. Using a SAM-refined mask plus a context ring, and ablating the ring width, is cheap and well grounded.
4. **A "grounded report" in the BTReport style for 2D Figshare.** Compute measurements deterministically (area, centroid or hemisphere, class probabilities, uncertainty flag), then use a template or a small local LLM only for wording. Add a verification step that checks that every number in the text matches the computed values. That gives a simple hallucination check. BTReport is 3D BraTS, so a 2D single-sequence version with uncertainty statements in the report appears open.
5. **Avoid as a headline:** "on-device" or quantisation alone (very common), and GAN augmentation for Figshare classification (common, mixed results). Diffusion augmentation needs the cloud and may not help.
6. **Patient-level splits.** Several augmentation papers report very high Figshare accuracy without visible patient-level splitting. Strict patient-level evaluation is a methodological plus.

---

## 5. [UNVERIFIED] (seen only in search snippets; not opened, or the page was blocked)

- BrainHack School 2024 project "Brain Tumor Segmentation via SAM-based fine-tuning on structural MRI images" (Li-Xuan Alex Peng). It reportedly fine-tunes the SAM decoder on the Figshare data. I did not open it, and it is not peer-reviewed.
- LiteMedSAM encoder about 5.7M parameters versus about 89.7M for MedSAM. This comes from a search summary citing a table in arXiv 2412.11186, which I did not see on the abstract page I opened.
- NeuroVision-Lite (MobileNetV3, ONNX export; FGCU or IISA proceedings). Not opened.
- MobileNetV2 deployed on Jetson TX2 (IJCTE vol. 18). Not opened.
- Conditional GAN plus Swin Transformer, 99.54% on Figshare (Frontiers in Medicine 2025, doi 10.3389/fmed.2025.1635796). Not opened.
- Truong et al., mask-conditioned 3D diffusion for multi-contrast brain tumour MRI (SPIE Medical Imaging 2024). Not opened.
- LDM-seg (latent diffusion with ControlNet for brain tumour labels, ISBI 2025). Not opened.
- BrainGPT (Nature Communications 2025, 3D brain **CT** report generation, 18,885 pairs). Not opened.
- GPT-4 interpreting Japanese brain tumour MRI reports (2024/2025). Not opened.
- Probabilistic SAM (arXiv 2509.05809, Ward & Imran 2025). Not opened.
- Clinical Contrastive Decoding (CCD) for hallucination in radiology MLLMs (2025). Not opened.
- Artsi et al. 2025, systematic review of LLMs in radiology reporting (medRxiv 10.1101/2025.03.18.25324193). Returned 403 when fetched.
- BiomedParse licence (cc-by-nc-sa-4.0), from a Hugging Face snippet. Not opened.
- The Nature, PubMed and doi.org pages for Ali 2024 and BiomedParse were blocked by redirects, cookies or authorisation. Both were verified through alternative pages (Europe PMC REST and arXiv).

## 6. Tool notes
- nature.com redirected to a login page, and PubMed required cookies. Workarounds were the Europe PMC REST API and arXiv abstracts.
- medRxiv returned 403.
- Abstract pages often omit parameter counts and VRAM figures, so many cells say "not extracted". Checking the full PDFs would fill them in.
