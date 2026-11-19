# UR-SAM — Zhang Y. et al., 2023 (arXiv 2311.10529)

- **Verified URL:** https://arxiv.org/abs/2311.10529 (opened 2026-11-19)
- **Title:** Enhancing the Reliability of Segment Anything Model for Auto-Prompting Medical Image Segmentation with Uncertainty Rectification
- **First author:** Yichi Zhang. First submitted 2023-11-17; v3 March 2024.

## What it does (from the abstract)
- Prompts come automatically from a localisation framework, much as our U-Net could supply boxes.
- A **prompt augmentation module** creates a set of perturbed prompts. The spread of SAM's outputs across them serves as a segmentation-uncertainty estimate.
- An **uncertainty-based rectification module** uses that uncertainty distribution to improve the final mask.
- It needs **no additional training or fine-tuning**.
- Dice improves by up to 10.7% and 13.8% on two public 3D datasets covering 35 organs.

## Why it matters for Mediscan AI
- This is the closest prior work to the "foundation model as a second opinion, with uncertainty" idea. Prompt-jitter uncertainty on its own is therefore **not novel**.
- What UR-SAM does *not* do, as far as the abstract says (inference):
  - brain tumour MRI, or the Figshare 2D T1-CE data
  - a *cross-model* disagreement signal between U-Net and SAM
  - a downstream tumour-type classifier whose input or confidence is gated by that uncertainty
  - risk–coverage or selective-prediction evaluation for classification
- **Feasibility on 4 GB (inference):** inference only. Compute the image embedding once, then re-run only the lightweight mask decoder for N jittered boxes. Cheap.

## Related work (verified)
- MedSAM-U (Zhou 2024, arXiv 2409.00924): adds uncertainty-guided multi-prompt adaptation to MedSAM; reports +1.7–20.5% over MedSAM.
- Ludwig 2025 (arXiv 2506.20786): SAM on BraTS only beats nnU-Net when given very accurate boxes. Prompt quality is the bottleneck, which is exactly what uncertainty should capture.
