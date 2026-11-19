# HANDOFF — current state

**Last updated:** 2026-11-19 (Phase 5: options proposed, waiting for Kareem's choice)
**Active phase:** **Phase 5 (idea selection)**. Ideas scored in [idea-matrix § Phase 5](ideas/idea-matrix.md), and 4 options written in [ideas/phase5-options.md](ideas/phase5-options.md): A safe (integrity + 'when does seg help'), **B recommended core (R1 seg-quality-aware routing)**, C ambitious (B + MedSAM second opinion + XAI diagnostic + grounded report; recommended as 'B first'), D product-centric. STOP: waiting for Kareem's choice → then ADR-0006.
**Branch:** `research/phase5-selection`. Baseline tag: `v1.0-semester1` = `390d7a0`.

## Where things stand
- Phase 1 audit: [00-project-understanding.md](00-project-understanding.md).
- Phase 3 idea analysis: [ideas/idea-matrix.md](ideas/idea-matrix.md). Now includes the Phase 4 verdicts and research ideas R1–R7.
- **Phase 4:** [research/research-summary.md](research/research-summary.md) is the main read. The [literature matrix](research/literature-matrix.md) holds the core table and the spot-check log. [references.bib](research/references.bib) has 116 verified entries. Topic files are `research/topics/T1–T4`; 13 paper notes are in `research/papers/`.
- **Phase 4 headline findings:**
  - D11 and D12 are common.
  - Classifier-only referral for brain tumors exists (Sharma 2026).
  - **Segmentation-quality-aware routing (R1) was not found** and is the leading candidate.
  - Keep the Figshare dataset (CC BY 4.0); use BDNeuro-MRI as the external test.
  - 0/12 semester-1 references are correct as cited.
- Semester-1 numbers: [EXP-000](experiments/EXP-000-semester1-checkpoint-eval.md) (clean test: Dice 0.411; acc 0.80 / 0.74 / 0.46).

## Key facts every agent must know
- **Raw data:** `dataset/dataset/` (git-ignored). Contents: `data/1..3064.mat`, `cvind.mat` (patient-disjoint folds). 233 patients, PIDs, binary masks. Licence CC BY 4.0: cite DOI 10.6084/m9.figshare.1512427 + Cheng 2015.
- The semester-1 evaluation is invalid (contaminated directories + patient leakage).
- The local PyTorch is **CPU-only**. The GPU is an RTX 3050 Laptop with **4 GB**.
- Rule: no code, model or training changes until Phase 6 is approved.

## Next steps
1. Kareem chooses an option (or asks questions) → write ADR-0006 and update HANDOFF.
2. Phase 6: detailed implementation plan (milestones, files, experiment design, metrics, baseline comparison, risks) → STOP for approval → Milestone 0 (leak-free strong baseline) → contribution.

## Open questions / blockers
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding.
