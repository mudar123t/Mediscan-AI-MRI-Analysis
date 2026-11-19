# MedSAM and LiteMedSAM — Ma J. et al.

- **Verified URLs (opened 2026-11-19):**
  - https://arxiv.org/abs/2304.12306 — "Segment Anything in Medical Images", Jun Ma, 2023; journal ref Nature Communications 15, 654 (2024)
  - https://github.com/bowang-lab/MedSAM (README)
  - https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM (branch README)
  - https://arxiv.org/abs/2412.16085 — "Efficient MedSAMs: Segment Anything in Medical Images on Laptop", Jun Ma, 2024

## Facts as reported
- MedSAM was trained on 1,570,263 image–mask pairs (10 modalities, more than 30 cancer types) and validated on 86 internal and 60 external tasks.
- The MedSAM README says training used 5 A100 nodes × 4 GPUs (80 GB). It also provides `train_one_gpu.py`. Inference VRAM is not stated.
- LiteMedSAM (released 2024-01-15) replaces the ViT encoder with **TinyViT**. Training happens in two stages: first distil TinyViT to reproduce MedSAM's embeddings, then fine-tune the whole pipeline. The README says it is "10x faster than MedSAM" and that "the training pipeline requires about 10GB GPU memory with a batch size of 4."
- The CVPR 2024 "MedSAM on Laptop" challenge summary (2412.16085) covers data from more than 20 institutions and 9 modalities. Per-team speeds were not on the abstract page.
- A parameter count for the LiteMedSAM encoder (about 5.7M versus 89.7M) was seen only in a search snippet, so it is **UNVERIFIED**.

## Use in Mediscan AI (inference)
1. **Box-prompted second opinion.** Take the U-Net's predicted mask, turn it into a bounding box with a few pixels of padding, and run LiteMedSAM or MedSAM. Compare the two masks. Their Dice and the spread under box jitter (UR-SAM style) become an uncertainty score.
2. **Refinement.** Our U-Net Dice is 0.41, which suggests underfitting. A SAM mask from a rough box may still be better. Ludwig 2025 warns that SAM's quality falls sharply when boxes are poor, so measure the effect of box error explicitly.
3. **Fine-tuning on Figshare.** LiteMedSAM is the realistic candidate for 4 GB: batch 1–2, AMP, and optionally a frozen encoder with cached embeddings. Fallback: one Kaggle or Colab T4 session of roughly 5–15 GPU-hours (estimate).
4. **Leakage caveat.** The training corpus of MedSAM or LiteMedSAM may include public brain MRI datasets. Whether the Figshare data are in it was **not checked**, and a zero-shot claim on Figshare must address this.
