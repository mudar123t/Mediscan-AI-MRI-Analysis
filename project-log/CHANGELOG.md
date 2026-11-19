# CHANGELOG

Newest first. Each entry: date, what changed, why, commits.

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
