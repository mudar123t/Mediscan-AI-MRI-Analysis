# HANDOFF — current state

**Last updated:** 2026-11-19 (M0.1 done)
**Active phase:** **Phase 6 execution, M0.1 (environment & repo structure).** Plan [plan-phase6.md](plan-phase6.md) approved ([ADR-0007](decisions/ADR-0007-phase6-plan-approval.md): legacy move yes, `.venv` + CUDA torch yes, 256² yes, Gradio, plan yes).
**Branch:** `chore/m0-env-structure`.

## M0.1 status: done (commit on `chore/m0-env-structure`)
- `.venv` (Python 3.12.2) with torch 2.9.1+cu128; GPU verified (RTX 3050, cap 8.6). Pinned `requirements.txt` + `requirements-lock.txt`. Details: [environment.md](environment.md).
- Semester-1 code in `legacy/semester1/` (git mv). New: `pyproject.toml` (`pip install -e . --no-deps`), `configs/base.yaml`, `src/mediscan/utils/repro.py`, `tests/` (6 pass incl. GPU).
- GPU probe: semester-1 UNet b8 256² AMP 0.102 s/step, 863 MiB.
- Open: README "Authors" section vs solo-presentation rule (asked Kareem).

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
1. M0.2 data pipeline (`feature/m0-data-pipeline`, EXP-001) → M0.3 segmentation → M0.4 classification → **STOP** (report vs baseline).
3. M1 (R3, uncertainty, router, conformal, shift/external) → STOP → M2 demo → STOP → decide extras.

## Open questions / blockers
- README "Authors" section: keep / remove / change? (solo-presentation rule)
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding · ADR-0006 Option C, "B first" · **ADR-0007 Phase 6 plan approved**.
