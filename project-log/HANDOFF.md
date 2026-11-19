# HANDOFF — current state

**Last updated:** 2026-11-19 (Phase 6 plan drafted, waiting for approval)
**Active phase:** **Phase 6, step 1: implementation plan.** Kareem chose **Option C, built as "B first"**; extras are decided later ([ADR-0006](decisions/ADR-0006-semester2-contribution.md)). The plan is in [plan-phase6.md](plan-phase6.md). **STOP: waiting for approval of the plan and answers to Q1–Q5.** No code has been written yet.
**Branch:** `research/phase6-plan`.

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
- Rule: no code, model or training changes until the Phase 6 plan is approved.
- Environment [V 2026-11-19]: driver CUDA 13.3, 15.2 GB RAM, 12 threads, **only 37 GB free disk** (the 47 GB pituitary dataset is infeasible).

## Next steps
1. Kareem approves the plan and answers Q1–Q5 (legacy move, deps + `.venv` CUDA torch, 256², Gradio, overall approval).
2. M0.1 environment and structure → M0.2 data pipeline → M0.3 segmentation → M0.4 classification → **STOP** (report vs baseline).
3. M1 (R3, uncertainty, router, conformal, shift/external) → STOP → M2 demo → STOP → decide extras.

## Open questions / blockers
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding · **ADR-0006 Option C, "B first" (R1 routing core; extras decided later)**.
