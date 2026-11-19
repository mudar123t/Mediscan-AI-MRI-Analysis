# Session 2026-11-19 — Phase 4: state-of-the-art research

## Goals
Survey the SOTA (2023–2026 + foundations) for brain-tumor MRI segmentation and classification. Check whether each novelty angle is already common. Find datasets and evaluation standards. Verify the semester-1 references. Write `research-summary.md`.

## Actions
- Recorded Kareem's Phase 3 decisions as ADR-0005.
- Launched 4 background subagents with identical strict verification rules: list a paper only if its link was opened, record metrics only if seen on the page, and put the rest in an [UNVERIFIED] section. Each agent wrote its own files (no overlap).
- Read all four topic reports in full. Ran an independent fetch to spot-check the 4 most consequential claims.
- Merged the BibTeX files and wrote the literature matrix, the research summary and the idea-matrix update.

## Findings (details in `research/research-summary.md`)
- D11 (mask channel) and D12 (multi-task) are common. D12 gives no reliable gain (Ferreira 2024: 95.99 vs 95.87 %).
- Classifier-only uncertainty referral for brain-tumor type exists (Sharma 2026, spot-checked).
- No verified paper conditions classification on segmentation quality or uncertainty → R1 (segmentation-quality-aware routing) is the leading candidate.
- Cheng 2015 (the dataset paper): a dilated ROI beats a tight ROI. This explains our EXP-000 result.
- Patient-split references: Dice 0.93 (Avazov 2024); acc 91.28 % (Cheng 2015) and 98 % (Deepak 2019).
- Figshare dataset licence is CC BY 4.0. External test: BDNeuro-MRI. Kaggle merges contain Figshare images.
- Semester-1 references: 0/12 correct as cited.
- Tool limits: several publishers blocked fetches (Nature, MDPI, Elsevier, PubMed, figshare HTML). The agents used Europe PMC, Crossref, arXiv, PMC and the Figshare API instead.

## Outcomes
Phase 4 deliverables committed. Waiting at the Phase 4 STOP gate.
