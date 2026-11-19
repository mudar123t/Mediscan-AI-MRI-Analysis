# HANDOFF — current state

**Last updated:** 2026-11-19 21:30 (M0.3 done, EXP-002 logged)
**Active phase:** **Phase 6 execution, M0.3 done → next M0.4 (classification + views, EXP-003), then STOP.** Plan [plan-phase6.md](plan-phase6.md) approved ([ADR-0007](decisions/ADR-0007-phase6-plan-approval.md)).
**Branch:** `exp/m0-segmentation` (commits `abc5efd`, `6145871`, `813903a` + docs commit; **not pushed yet: waiting for Kareem's OK**).

## Milestone status
- **M0.1 done** (`7c3a36b..5471d94`): `.venv` torch 2.9.1+cu128, GPU verified; pinned deps; `legacy/semester1/`; pyproject, configs, repro utils, tests. [environment.md](environment.md).
- **M0.2 done** ([EXP-001](experiments/EXP-001-data-pipeline.md)): `scripts/prepare_figshare.py` → `data/processed/figshare_256/` (images.npy, masks.npy uint8 256², metadata.csv, report.json; 384 MB, git-ignored) + tracked `data/splits/figshare_fold{0..4}.json` (test k, val (k+1) mod 5, train rest). 3,064 slices / 233 patients, 0 patients across folds, 0 empty masks, border-IoU median 0.970. 19 tests pass.
- How to rebuild the data: `.venv/Scripts/python scripts/prepare_figshare.py [--overwrite]` (~90 s).
- Facts learned: semester-1 images were transposed; the dataset has **no pixel spacing** (HD95 in pixels); slices mix axial/coronal/sagittal views (no view field).
- README authors section removed (`3923a2a`, Kareem's request).
- **M0.3 done ([EXP-002](experiments/EXP-002-segmentation.md)):** `src/mediscan/models/seg.py` (`unet_s1` semester-1 UNet; `unet_r34` smp U-Net ResNet-34 ImageNet + Dropout2d 0.2 after each decoder block), `metrics/seg.py` (Dice, IoU, HD95 px, NSD@2px, empty-prediction rule, LCC), `stats.py` (patient bootstrap), `scripts/train_seg.py`, `scripts/eval_seg.py`, configs `exp002{a,b,c}`. 28 tests pass.
  - Probe (fold 0, 2 epochs, r34): 26 s/epoch, 987 MiB VRAM, val Dice 0.57. Probe outputs in `runs/probe_r34/` (not part of results).
  - Ran `bash runs/run_exp002.sh` (a → b → c, 5 folds each, 6.1 h). Pooled 5-fold test, 3,064 slices / 233 patients:
    - a (semester-1 UNet, old recipe; **fair baseline**): Dice 0.508 [0.475, 0.541], HD95 56.8 px.
    - b (same UNet, new recipe): Dice 0.759; Δ vs a +0.250 [0.228, 0.273].
    - **c (ResNet-34 U-Net): Dice 0.786 [0.766, 0.806], HD95 14.4 px**; Δ vs b +0.027 [0.019, 0.037]. Glioma 0.685 / meningioma 0.893 / pituitary 0.858.
  - Planned target 0.85–0.93 **not reached** (reasons only inferred). Arm c is the candidate M1 segmenter; its weights and OOF predictions are in `runs/exp002c_unet_r34/fold*/`.

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
1. Kareem: OK to push `exp/m0-segmentation` and fast-forward `main`? Then M0.4 classification + views (EXP-003, branch `exp/m0-classification`) → **STOP** (report vs baseline; decide whether to tune the segmenter).
2. M1 (R3, uncertainty, router, conformal, shift/external) → STOP → M2 demo → STOP → decide extras.

## Open questions / blockers
- Segmentation below the planned target (0.786 vs 0.85–0.93): accept for M1, or 1–2 validation-only tuning runs (extra seeds, 384², loss, threshold)? To be decided at the M0.4 STOP.
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding · ADR-0006 Option C, "B first" · **ADR-0007 Phase 6 plan approved**.
