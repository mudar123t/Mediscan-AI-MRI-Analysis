# CHANGELOG

Newest first. Each entry: date, what changed, why, commits.

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
