# CHANGELOG

Newest first. Each entry: date, what changed, why, commits.

## 2026-11-21 — M3: R2 (EXP-011) results
- Run 00:58–05:50 finished without failure or retry; EXP-008 Dice check passed (60/60). `scripts/eval_exp011.py` (parts ABC, D) → `experiments/EXP-011-medsam-second-opinion/results.json`. Negative: MedSAM (U-Net box) Dice 0.593 vs 0.785; seg-failure AUROC 0.717 vs ens 0.899; router(sam) ties router(ens) in distribution, worse worst case under shift (−0.0125 [−0.022, −0.003]); external mixed. Logged in EXP-011, results-summary, thesis-map, presentation-bank, session log. Committed locally on `research/m3-r2-medsam`; not pushed.

## 2026-11-21 — M3: R2 (EXP-011) started overnight
- Branch `research/m3-r2-medsam`. `a397cc2` pre-specifies EXP-011 (3 MedSAM signals, choice on val, frozen EXP-006 router rule, shift/external). `4604c1f` adds `src/mediscan/medsam.py` (4 tests), `scripts/run_exp011.py`, `scripts/run_exp011_retry.sh` (auto-retry), pins transformers 5.23.0 / tokenizers 0.23.3 / regex 2026.9.29 (newest regex was 1 day old). `scripts/eval_exp011.py` written. 78 tests pass. Run started 00:58, ETA ~05:30.

## 2026-11-20 — M3: R4 (EXP-010) results
- GPU run 18:57–19:02 (logit check passed); analysis + figures in `experiments/EXP-010-xai-diagnostic/`. Descriptive: crop focuses attribution on the lesion (~70 % vs 22 %). Negative: attribution is not a useful error or routing signal (router(cam) −0.008 vs ens router). Logged in results-summary, thesis-map, presentation-bank, session log. Merged into `main`.

## 2026-11-20 — M3: R4 (EXP-010) started
- Kareem chose R4 now, R2 tonight, R5 skipped. Branch `research/m3-r4-xai`; `e28ae5a` pre-specifies EXP-010 before any code. Code: `src/mediscan/xai.py` (Grad-CAM, back-projection, region fractions; 4 tests), `scripts/run_exp010.py`, `scripts/eval_exp010.py`. GPU run started ~18:57.

## 2026-11-20 — MIT licence; GPU crash diagnosis
- Kareem chose MIT with copyright "mudar123t": `LICENSE`, `pyproject.toml` `license = "MIT"`, README License section (weights also MIT; Figshare CC BY 4.0 attribution; not for clinical use). ADR-0009 updated.
- Morning crash explained from Windows logs: AMD iGPU driver TDR storm, then the display moved to the NVIDIA GPU, which killed the CUDA context. No hardware errors. Overnight runs will use auto-retry (HANDOFF key facts).
- Kareem: do R4 now, then stop; R2 will run tonight.

## 2026-11-20 — M2 approved; demo weights released
- Kareem approved the demo ("looks good"; screenshots saved to `images/demo_ui_*.png`) and chose option (a) ([ADR-0009](decisions/ADR-0009-m2-demo-weights-release.md)).
- `00eeaf6` demo loads only the crop32/full classifiers; README download instructions. `e2e062f` screenshots. `main` fast-forwarded to `e2e062f` and pushed.
- GitHub Release `demo-weights-fold3-v1` with `mediscan-demo-weights-fold3.zip` (355.6 MB, SHA-256 da2c6928…). Fresh-clone check passed.

## 2026-11-20 — M2 local demo (STOP: demo review)
- Branch `feature/m2-demo`: Gradio 6.32.0 pinned; `pipeline.FoldModels` refactor (output identical); `mediscan/demo.py`; `scripts/export_demo_params.py`, `configs/demo.yaml`, `app/params_fold3.json`; `app/demo_app.py`; `tests/test_demo.py` (70 tests pass); README demo section + figure; environment.md.
- Fold 3 used (fold 0's τ1 = −∞ would hide routing). Demo reproduces EXP-006 decisions on test-fold examples.

## 2026-11-20 — EXP-006b results
- (b) pre-specified pooled-val τ1 does not stabilise the router (flat objective); reported as a negative result, frozen results stay primary.
- Pushed `exp/m1-tau1-robustness` and fast-forwarded `main` to `7c7754d`.
- (d) robustness summary CIs: frozen router worst case and mean AURC significantly better than both fixed views.
- Updated EXP-008 log (CI pointer), results-summary, thesis-map, presentation-bank, session log, HANDOFF.

## 2026-11-20 — EXP-006b started (ADR-0008 (b)+(d))
- Branch `exp/m1-tau1-robustness`. `f547997` pre-specifies the pooled-val τ1 rule and the robustness-summary CIs before running; `9ced2ce` adds `policy_eval.pooled_val_tau1`, `fit_params(tau1=)`, `scripts/eval_exp006b.py` and 2 tests.
- Running since 17:31 (CPU). Early: pooled-val τ1 does not stabilise the threshold (fold 4 flips to always-full).

## 2026-11-20 — M1.4 gate decision
- Kareem chose (b) stabilise τ1 + (d) CIs for the EXP-008 summary, not run yet ([ADR-0008](decisions/ADR-0008-m1-gate-decisions.md)). Branch `exp/m1-uncertainty` pushed, then fast-forwarded into `main` and pushed (plan §1: merge when green; Kareem: "do what is better").

## 2026-11-20 — M1 complete (EXP-008/009 final), STOP at M1.4
- EXP-008/009 inference re-run with det+ens (16:07–16:46) for the frozen score `ens:disagree`; final analysis in `experiments/EXP-008-shift/`, `EXP-009-external/`.
- Added `EXP-008-shift.md`, `EXP-009-external.md`, session log `sessions/2026-11-20-m1-routing-shift-external.md`; updated results-summary, thesis-map, presentation-bank.
- H-B1 not supported; H-B2 vs full only; H-B3 partly (synthetic noise/blur).

## 2026-11-20 — Final EXP-005/006 with seed ensemble; M1 logs
- Seed run finished 16:04; ensemble scores computed. EXP-006 frozen score is now `ens:disagree`; conclusions unchanged (router ≈ crop32; better AURC than full). EXP-008/009 re-run with det+ens started 16:07.
- Added `experiments/EXP-004-mask-quality.md`, `EXP-005-uncertainty.md`, `EXP-006-router.md` (with EXP-007).

## 2026-11-20 — Seed run crash and resume
- Seed 2 fold 2 crashed (CUDA unknown error, transient); run script now uses `set -eo pipefail` so failures are not masked by the log filter; resumed 14:02, ETA ~15:45–16:30. Noted that seeded training is not bit-exact on this GPU.

## 2026-11-20 — EXP-008/009 inference done
- EXP-009 external (3,030 clean BDNeuro × 5 fold model sets) and EXP-008 shift (13 conditions × 5 folds) pipeline outputs in `runs/` (git-ignored); preliminary analysis running. Seed-run finish re-estimated to ~14:30–15:00.
- Preliminary EXP-008/009 analysis: router protects against the crop collapsing under noise/blur s3 (+0.073 / +0.142 bal. acc vs crop32) and against the full view collapsing under gamma (+0.089 vs full at s2); on the external set crop32 is best (0.931) and the router is in between (0.922).

## 2026-11-20 — EXP-006 preliminary; EXP-008/009 runs started
- Preliminary EXP-006/007 (no ensemble): router ≈ crop32 in-distribution (Δ bal. acc −0.006, n.s.), better AURC than full image; oracle headroom 0.974.
- `cfe8429` exp: `scripts/eval_shift_external.py` (frozen policies on shift/external).
- Long run started ~10:35: EXP-009 external + EXP-008 shift pipeline (det+tta), ~1.5 h.

## 2026-11-20 — EXP-004 results; EXP-005 analysis fix; EXP-006 criterion
- EXP-004 inference + analysis done (`experiments/EXP-004-mask-quality/`: controlled.csv, predicted_by_dice.csv, results.json, figure).
- EXP-005 det/mc/tta scores done; analysis: per-fold regression pooling artefact found and fixed (−log(1−MSP) features, mean within-fold AUROC) in `scripts/eval_exp005.py` and `policy_eval._combined_features`.
- EXP-006 score-selection criterion changed to val AUROC for "crop wrong, full right" (reason in HANDOFF); preliminary EXP-006 running.

## 2026-11-20 — M0.4 pushed; M1 started
- Pushed `exp/m0-classification`; `main` fast-forwarded to `1361e81` (Kareem approved). Kareem: keep the EXP-002c segmenter, start M1, do M1.4 without asking.
- Branch `exp/m1-uncertainty` (+ `exp/m1-r3-seg-helps`): `b7ac0ee` seed configs; `8fab45b` EXP-004 degradation study; EXP-005 uncertainty sampling/scores; `3f6187e` selective metrics, router, conformal, BDNeuro prep + duplicate audit; `9e8b014` policy evaluation, corruptions, end-to-end pipeline.
- BDNeuro-MRI downloaded (112 MB, SHA-256 verified): 1,682 / 5,154 tumor images are Figshare duplicates → only 3,030 clean images used.
- Long runs started: segmenter seeds 1–2 (~4.6 h+), EXP-004 and EXP-005 inference.

## 2026-11-20 — M0.4 done: EXP-003 results logged (STOP gate)
- EXP-003 (40 runs, 4.1 h, 5-fold patient-disjoint, 3,064 slices): fair semester-1 baseline SimpleCNN bal. acc 0.733 [0.703, 0.765]; ResNet-18 full 0.928; best automatic pipeline ResNet-18 crop 32 px + predicted masks 0.941 [0.916, 0.961] (acc 0.948); masked ROI (semester-1 view) −0.079 vs full; crop 32 beats full with GT masks (+0.026) but not with predicted masks (+0.013, n.s.).
- New: `experiments/EXP-003-classification.md` + `results.json`; session log `sessions/2026-11-20-m0-4-classification.md`. Updated results-summary, thesis-map, presentation-bank (beats 10–12), environment (classifier epoch times), HANDOFF.
- *Why:* close M0.4 and report at the STOP gate.

## 2026-11-20 — M0.4 code; EXP-003 training started
- `18e6fe1` feat: classifier views (full, masked ROI, context crop), cls model factory (SimpleCNN, ResNet-18, EfficientNet-B0), cls metrics (bal. acc, macro-F1, MCC, AUROC, ECE, Brier, NLL), temperature scaling, paired confusion-matrix patient bootstrap; optimizer/schedule helpers moved to `utils/optim.py` (train_seg behaviour unchanged). 37 tests pass.
- `7cf0f70` exp: `scripts/train_cls.py`, `scripts/eval_cls.py`, 8 EXP-003 configs (SimpleCNN old recipe; ResNet-18 × 6 views; EfficientNet-B0 full).
- Training of 40 runs started 2026-11-20 00:02 (estimate 3.5–5.5 h from probes).
- *Why:* M0.4 of the approved plan, as approved by Kareem (no additions).

## 2026-11-19 — M0.3 pushed
- Pushed `exp/m0-segmentation` (`abc5efd..cf69bb2`) and fast-forwarded `main` `9c30667..cf69bb2` (Kareem approved). M0.4 on hold until Kareem says go.

## 2026-11-19 — M0.3 done: EXP-002 results logged
- EXP-002 (5-fold, patient-disjoint, 3,064 slices): a old recipe Dice 0.508 [0.475, 0.541] (fair semester-1 baseline); b new recipe 0.759 (+0.250); c ResNet-34 U-Net 0.786 [0.766, 0.806], HD95 14.4 px (+0.027 vs b). Planned target 0.85–0.93 not reached.
- New: `experiments/EXP-002-segmentation.md` + `results.json`, session log `sessions/2026-11-19-m0-3-segmentation.md`. Updated results-summary, thesis-map, presentation-bank (beats 8–9), environment (measured epoch times), HANDOFF.
- *Why:* close M0.3 with the numbers on file before M0.4.

## 2026-11-19 — M0.3 segmentation code; EXP-002 training started
- `abc5efd` feat: seg model factory (semester-1 UNet; ResNet-34 U-Net with decoder dropout), seg metrics (Dice/IoU/HD95 px/NSD), patient bootstrap.
- `6145871` exp: `scripts/train_seg.py` + configs EXP-002a (old recipe: 5 ep, bs 4, Adam 1e-3, no aug, fp32), 002b (same UNet, new recipe: ≤100 ep, cosine, AdamW, aug, AMP, early stop 15), 002c (pretrained ResNet-34 U-Net, lr 3e-4).
- Eval script `scripts/eval_seg.py`; vectorised `mean_ci`.
- *Why:* M0.3 of the plan; the fair semester-1 baseline and the improved segmenter for M1.

## 2026-11-19 — M0.2 data pipeline (EXP-001)
- `3923a2a` docs: README authors section removed (Kareem's request).
- New `src/mediscan/data/{figshare,splits,datasets,augment}.py`, `scripts/prepare_figshare.py`, `configs/data_figshare.yaml`, `tests/test_data_pipeline.py` (13 tests), tracked `data/splits/figshare_fold{0..4}.json`.
- EXP-001: 3,064 slices / 233 patients, patient-disjoint splits, orientation verified against tumorBorder (median IoU 0.970). Found: semester-1 images were transposed; no pixel spacing in the dataset → plan §2.2 HD95 in pixels.
- results-summary, thesis-map, presentation-bank updated.
- *Why:* M0.2 of the approved plan; every later experiment reads only these split files.

## 2026-11-19 — Phase 6 plan approved; M0.1 environment and structure done
- ADR-0007: Q1–Q5 approved (legacy move, `.venv` + CUDA torch, 256², Gradio, plan).
- `git mv` of semester-1 scripts, `conv.py`, `notebooks/` → `legacy/semester1/` (history kept); `03_check_pilot_data,py` → `.py`.
- New: `pyproject.toml`, `configs/base.yaml`, `src/mediscan/utils/repro.py`, package `__init__.py` files, `tests/` (pytest), `legacy/semester1/README.md`. README: Figshare dataset citation + setup. `.gitignore`: `/data/processed/`.
- `.venv` with torch 2.9.1+cu128 (GPU verified); pinned `requirements.txt` + `requirements-lock.txt`; `project-log/environment.md` (versions + GPU probe 0.102 s/step, 863 MiB). CLAUDE.md commands/folder map updated. pytest: 6 passed.
- *Why:* M0.1 of the approved plan (reproducible environment and structure before the data pipeline).

## 2026-11-19 — Phase 5 decision + Phase 6 plan draft (no code changes)
- **ADR-0006:** Kareem chose Option C, built as "B first" (R1 segmentation-quality-aware routing + R3 + R7, evaluated with the R6 protocol). Extras R4/R2/R5 are decided later.
- `plan-phase6.md`: milestones M0.1–M2, files and modules to create, experiment design EXP-001–009 with hypotheses, router definition, compute/disk budget, dependencies, risks, definition of done, and questions Q1–Q5.
- Environment facts checked: driver CUDA 13.3, 15.2 GB RAM, 37 GB free disk.
- `university-requirements.md` R1/R3 updated with how they are addressed.

## 2026-11-19 — Phase 5: idea scoring and options (no code changes)
- `ideas/idea-matrix.md`: numeric scores (N/F/T/M/D/Th, 1–5) for all ideas. R1 leads (26/30), followed by R3 (24), P0 and R7 (23), and R2 and R4 (22).
- `ideas/phase5-options.md`: Milestone 0 (common groundwork) plus 4 options with pitch, hypotheses, evaluation, risks and effort. Options: A safe, B recommended core, C ambitious, D product.
- Ref [12] decision: replace with Abdusalomov et al. 2023 (Kareem does not know the original source).
- *Why:* Phase 5 of the plan. The choice is pending, and an ADR will follow it.

## 2026-11-19 — Phase 4: state-of-the-art research (no code changes)
- 4 parallel research subagents (approved): T1 SOTA seg/cls + multi-task (25 verified), T2 uncertainty/XAI/HITL (38), T3 foundation/generative/edge/reports (27), T4 datasets/evaluation/semester-1 reference check (33 bib). Output in `research/topics/`, plus 13 notes in `research/papers/`.
- Main agent spot-checked 4 key sources via an independent fetch (Sharma 2026, Vangala 2026, Figshare API licence, Cheng 2015). All confirmed; caveats logged in `literature-matrix.md`.
- `research/references.bib`: merged 124 → **116** unique entries (de-duplicated by key and DOI).
- `research/literature-matrix.md` (core table + index), `research/research-summary.md` (takeaways, trends, datasets, evaluation standard, candidates R1–R7).
- `ideas/idea-matrix.md`: Phase 4 novelty verdicts (D11 and D12 common; E13 partly pre-empted; R1 not found) and research ideas R1–R7.
- ADR-0005 (drop B6; groundwork vs contribution; time not binding). Thesis map and presentation bank updated.
- *Why:* Phase 4 of the plan; the novelty claims must rest on verified literature.

## 2026-11-19 — Phase 3: future.txt analysis (no code changes)
- `ideas/idea-matrix.md`: all 20 `future.txt` items (E1–E6, A1–A3, B4–B7, C8–C9, D10–D12, E13–E14) plus the audit prerequisite P0, each restated with feasibility, effort, novelty, risks and a recommendation. *Why:* Phase 3 of the plan.
- Verified the re-downloaded raw data in `dataset/dataset/` (3,064 `.mat`, 233 patients, PID present, binary masks). **Patient leakage confirmed:** 43/50 clean test images come from training patients. `cvind.mat` folds are patient-disjoint. Updated audit §8.2, EXP-000 and the presentation bank.
- ADR-0002 accepted (docs local); ADR-0001 updated (checkpoints local only, no Release).
- Branch `research/phase3-future-analysis`; commit listed in the next entry's push.

## 2026-11-19 — Phase 2: repo cleanup, memory and logging setup
- **History rewrite (approved, ADR-0001):** fast-forwarded to `3337b1a`; created the backup bundle `../mediscan-backup-2026-11-19/mediscan-original-history.bundle`; removed `dataset/`, `data/pilot/{processed,roi_gt,roi_pred}/`, `outputs/` and `__pycache__` from all commits; force-pushed with a lease (`3337b1a...390d7a0`). Repo pack 225 MB → 210 KB. *Why:* the dataset and checkpoints were public by mistake (the ignore file was inactive).
- **Tag** `v1.0-semester1` → `390d7a0` (pushed).
- `d7d2330` chore: `.gitignore.txt` → `.gitignore`, new rules (data, weights, docs, caches, secrets, Claude state).
- `6bca2df` chore: `.claude/settings.json` plus the SessionStart and Stop hooks (ADR-0004).
- `fdf6251` docs: `CLAUDE.md`.
- `b7ee178` docs: Phase 1 audit, ADR-0001..0004, project-log skeleton (research, ideas, thesis map, presentation bank, university requirements).
- `82e56d0` exp: EXP-000, inference-only re-evaluation of the semester-1 checkpoints. *Why:* no semester-1 metrics were recorded anywhere.
- This entry, HANDOFF and the session log: docs commit on `chore/phase2-repo-setup`, merged to `main`.

## 2026-11-19 — Phase 1: audit (read-only)
- Added `project-log/00-project-understanding.md`. Main findings: contaminated val/test directories, image-level split, no recorded results, wrong dataset citation, CPU-only PyTorch.
