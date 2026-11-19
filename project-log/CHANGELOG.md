# CHANGELOG

Newest first. Each entry: date, what changed, why, commits.

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
