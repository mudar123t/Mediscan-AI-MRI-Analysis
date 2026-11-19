# BTReport — Heras Rivera J. E. et al., 2026 (arXiv 2602.16006, MIDL 2026)

- **Verified URL:** https://arxiv.org/abs/2602.16006 (opened 2026-11-19)
- **Title:** BTReport: A Framework for Brain Tumor Radiology Report Generation with Clinically Relevant Features
- **First author:** Juampablo E. Heras Rivera. Submitted 2026-02-17. The arXiv page says it was accepted to MIDL 2026.

## What it does (from the abstract)
- It is an open-source framework. Image features are extracted **deterministically**, and an LLM is used **only for syntactic structuring and narrative formatting**, not for interpreting the image.
- It releases BTReport-BraTS, which adds synthetic reports generated with BTReport to BraTS imaging.
- The extracted features predict survival and IDH status. The generated reports align better with reference clinical reports than baselines do. No numbers appear in the abstract.
- The authors claim the reports are interpretable and less prone to hallucination. The abstract gives no quantitative hallucination evaluation.
- The specific LLM was not named on the abstract page.

## Why it matters for Mediscan AI
- It confirms that the "structured report from mask, class and measurements, with an LLM only for wording" pattern was **published in 2026 for 3D multiparametric BraTS**. So the idea is no longer new in principle.
- Possible remaining openings (inference):
  - 2D, single-sequence (T1-CE) Figshare setting, with tumour-type classification included
  - uncertainty-aware wording, for example hedging language or a "review recommended" flag driven by U-Net/SAM disagreement
  - an automatic **number-consistency check** that compares the generated text with the computed measurements, a cheap hallucination guard
- **Feasibility on 4 GB (inference):** a template-only version is trivial. Small local LLMs such as MedGemma-4B (arXiv 2507.05201; 4B and 27B sizes) at 4-bit, or CPU inference, are plausible. An API is easier but raises privacy and cost questions.

## Related work (verified)
- Brain3D (Barone 2026, arXiv 2602.22098): an end-to-end 3D VLM trained on 468 BraTS subjects; pathology F1 0.951.
- ReXTrust (Hardy 2024, arXiv 2412.15264): a finding-level hallucination detector (AUROC 0.8751; chest X-ray).
