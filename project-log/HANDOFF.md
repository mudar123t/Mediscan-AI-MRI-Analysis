# HANDOFF — current state

**Last updated:** 2026-11-19 (end of Phase 4, before Kareem's approval)
**Active phase:** Phase 4 done → waiting at the STOP gate (present the research summary). Next: **Phase 5 (idea selection)** after approval.
**Branch:** `research/phase4-sota` (fast-forward into `main` after commit). Baseline tag: `v1.0-semester1` = `390d7a0`.

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
1. Kareem reviews the Phase 4 summary.
2. Phase 5: score all ideas (future.txt + R1–R7) on N/F/T/M/D/Th. Propose 3–5 options (at least one ambitious and one safe). Discuss. Record the choice as an ADR.
3. Phase 6: plan → groundwork milestone (new leak-free baseline) → contribution.

## Open questions / blockers
- Semester-1 ref [12]: T4 thinks the real source is Abdusalomov et al. 2023 (*Cancers* 15:4172). Kareem should confirm before we replace it.
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding.
