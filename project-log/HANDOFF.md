# HANDOFF — current state

**Last updated:** 2026-11-20 17:10 (M1 complete; **at the M1.4 STOP gate**, waiting for Kareem)
**Active phase:** **Phase 6 execution — M1 done; STOP after M1.4 (H-B1..H-B3 report given 2026-11-20, waiting for Kareem's decision).** Kareem (2026-11-20): push yes; keep EXP-002c segmenter (no tuning); start M1 with the seed run first; do M1.4 as needed without asking. Next STOP: after M1.4 (report H-B1..H-B3). Kareem wants to be told before every long run (with time estimate). Plan [plan-phase6.md](plan-phase6.md) approved ([ADR-0007](decisions/ADR-0007-phase6-plan-approval.md)).
**Branch:** `exp/m1-uncertainty` (M1 work, from `main` `1361e81`; contains `exp/m1-r3-seg-helps`; not pushed). `main` = `origin/main` = `1361e81` (M0.4 pushed).

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

- **M0.4 done ([EXP-003](experiments/EXP-003-classification.md)):** `src/mediscan/views.py` (full, masked_roi = semester-1 style, context_crop(margin) square bbox + margin resized to 256; empty pred mask → crop falls back to full image), `models/cls.py` (semester-1 SimpleCNN; timm ResNet-18 / EfficientNet-B0 ImageNet, in_chans=1), `metrics/cls.py`, `calibration.py` (temperature scaling, val only), `stats.confusion_bootstrap` (paired patient bootstrap), `utils/optim.py` (moved from train_seg, unchanged behaviour). `scripts/train_cls.py`, `scripts/eval_cls.py`, 8 configs `configs/exp003*.yaml`. 37 tests pass.
  - Arms (5 folds each): a SimpleCNN old recipe (10 ep, bs 8, Adam 1e-3, no aug, best val acc) full image; ResNet-18 full / masked / crop 0, 8, 16, 32 px; EfficientNet-B0 full. New recipe: ≤60 ep, bs 32, AdamW 3e-4, cosine, aug, AMP, patience 15, select on val balanced accuracy.
  - Train/select on GT-mask views; test (and val, for temperature fitting) also with EXP-002c predicted masks (thr 0.5 + largest component).
  - Probes (fold 0, 2 epochs, `runs/probe_exp003*`, not results): ResNet-18 ~9 s/epoch, 753 MiB; EffNet-B0 ~10 s/epoch, 1,898 MiB; SimpleCNN ~4 s/epoch.
  - Ran 2026-11-20 00:01–04:08 (4.1 h, 40 runs). Pooled 5-fold test (balanced accuracy):
    - SimpleCNN old recipe (**fair baseline**): 0.733 [0.703, 0.765] (acc 0.761).
    - ResNet-18 full 0.928; EffNet-B0 full 0.931 (equal).
    - Context rises with margin (GT masks): masked 0.870, crop0 0.881, crop8 0.928, crop16 0.939, **crop32 0.954**.
    - Predicted masks cost 0.6–2.1 points: **crop32 pred 0.941 [0.916, 0.961]** (acc 0.948 > Cheng 91.28 %).
    - Crop32 vs full: GT +0.026 (p 0.005); pred +0.013 [−0.005, +0.029] (n.s.) → motivates R1.
    - Temperature scaling lowers ECE everywhere (crop32 pred 0.033 → 0.005).

- **M1 done (2026-11-20)**, branch `exp/m1-uncertainty`, not pushed. Logs: [EXP-004](experiments/EXP-004-mask-quality.md), [EXP-005](experiments/EXP-005-uncertainty.md), [EXP-006/007](experiments/EXP-006-router.md), [EXP-008](experiments/EXP-008-shift.md), [EXP-009](experiments/EXP-009-external.md); session [2026-11-20-m1-routing-shift-external](sessions/2026-11-20-m1-routing-shift-external.md).
  - EXP-004: crop32 tolerant to eroded/dilated/shifted masks (0.933 at Dice 0.3), broken by false-positive blobs (0.636 at Dice 0.9); real Dice < 0.3 (224 slices, 172 zero overlap): crop32 −0.248 vs full.
  - EXP-005: `ens:disagree` best seg-failure detector (AUROC 0.899); no seg score predicts classifier error or adds to MSP (0.909).
  - EXP-006/007 (frozen score `ens:disagree`): router 0.935 vs crop32 0.941 (n.s.); AURC vs full −0.0115 [−0.0327, −0.0006]; τ1 unstable (val crop share 0.00–0.94); oracle 0.974. CP marginal coverage ≈ 1−α; meningioma 0.75 at α 0.1.
  - EXP-008: router − crop32 grows with noise/blur severity (+0.072 / +0.145 at s3); router − full +0.05..+0.09 under gamma; best worst-case (0.607) and mean AURC (0.072) over 13 conditions. CP coverage collapses under shift, referral rises.
  - EXP-009: BDNeuro 1,682/5,154 duplicates of Figshare (33 % of its test split); clean 3,030: crop32 0.931, router 0.922, full 0.904; router outcome set by each fold's τ1.
  - **Hypotheses:** H-B1 not supported; H-B2 supported vs full-image MSP gate only; H-B3 partly supported (synthetic noise/blur yes; external no).
  - Run notes: seed run crashed once (transient CUDA error), resumed; `runs/run_exp005_seeds.sh` now `set -eo pipefail`; seeded training not bit-exact. Preliminary det+tta outputs in `runs/*_prelim_tta/`.

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
1. **Wait for Kareem at the M1.4 STOP.** Then commit the M1 logs/outputs (if not done) and show what will be pushed before pushing `exp/m1-uncertainty`.
2. Options to discuss (not decided): (a) accept M1 as is and go to M2 demo; (b) stabilise τ1 (e.g. fit on pooled val of all other folds, or a fixed Dice-based rule) — needs Kareem's OK as it changes the frozen protocol; (c) class-conditional (Mondrian) CP for the meningioma gap; (d) CIs for the worst-case/mean-over-conditions summary in EXP-008.
3. Then M2 demo → STOP → decide extras.

## Open questions / blockers
- Segmentation below target (0.786): Kareem decided 2026-11-20 to keep EXP-002c (no tuning).
- M1.4 STOP: which of options (a)–(d) above, if any?
- Ref [12] resolved: Kareem does not know the source → replace it with Abdusalomov et al. 2023 (thesis-map).
- Not yet read: the Cheng 2015 correction notice (e0144479), the BraTS licence/size details, and the Badža 2020 subject-wise number.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local · ADR-0003 git workflow · ADR-0004 logging + hooks · ADR-0005 drop B6, groundwork vs contribution, time not binding · ADR-0006 Option C, "B first" · **ADR-0007 Phase 6 plan approved**.
