# HANDOFF — current state

**Last updated:** 2026-11-19 (M0.2 done)
**Active phase:** **Phase 6 execution, M0.2 done → next M0.3 (segmentation, EXP-002).** Plan [plan-phase6.md](plan-phase6.md) approved ([ADR-0007](decisions/ADR-0007-phase6-plan-approval.md)).
**Branch:** `feature/m0-data-pipeline`.

## Milestone status
- **M0.1 done** (`7c3a36b..5471d94`): `.venv` torch 2.9.1+cu128, GPU verified; pinned deps; `legacy/semester1/`; pyproject, configs, repro utils, tests. [environment.md](environment.md).
- **M0.2 done** ([EXP-001](experiments/EXP-001-data-pipeline.md)): `scripts/prepare_figshare.py` → `data/processed/figshare_256/` (images.npy, masks.npy uint8 256², metadata.csv, report.json; 384 MB, git-ignored) + tracked `data/splits/figshare_fold{0..4}.json` (test k, val (k+1) mod 5, train rest). 3,064 slices / 233 patients, 0 patients across folds, 0 empty masks, border-IoU median 0.970. 19 tests pass.
- How to rebuild the data: `.venv/Scripts/python scripts/prepare_figshare.py [--overwrite]` (~90 s).
- Facts learned: semester-1 images were transposed; the dataset has **no pixel spacing** (HD95 in pixels); slices mix axial/coronal/sagittal views (no view field).
- README authors section removed (`3923a2a`, Kareem's request).

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
- System Python has CPU-only torch; semester-2 work uses `.venv` (CUDA build). GPU: RTX 3050 Laptop, **4 GB**.
- Environment [V 2026-11-19]: driver CUDA 13.3, 15.2 GB RAM, 12 threads, **only 37 GB free disk** (the 47 GB pituitary dataset is infeasible).

## Next steps
1. M0.3 segmentation → M0.4 classification → **STOP** (report vs baseline).
2. M1 (R3, uncertainty, router, conformal, shift/external) → STOP → M2 demo → STOP → decide extras.

## Open questions / blockers
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding · ADR-0006 Option C, "B first" · **ADR-0007 Phase 6 plan approved**.
