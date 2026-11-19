# CHANGELOG

Newest first. Each entry: date, what changed, why, commits.

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
