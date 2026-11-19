# Session 2026-11-19 — Phase 1 audit and Phase 2 setup

## Goals
1. Read every project file and document; audit git and GitHub state (Phase 1).
2. Baseline tag, repo hygiene, CLAUDE.md, logging system, hooks (Phase 2).

## Actions
- Read all code (`src/`, `scripts/`, `conv.py`), configs, README, `future.txt` and all 10 files in `docs/` (PDFs via pdftotext, DOCX via python-docx, `.doc` via strings).
- Checked the data: split lists vs directories, file mtimes, class counts, mask values, `cvind.mat` folds, nearest-neighbour NCC between test and train.
- Kareem answered the 7 audit questions (recorded in `00-project-understanding.md §10`).
- Fast-forward → tag → bundle backup → filter-repo in a scratch clone → mixed reset → force-push with lease → retag (ADR-0001).
- Wrote `.gitignore`, hooks, `CLAUDE.md`, ADR-0001..0004 and the log skeleton.
- Ran EXP-000 (CPU inference, 69 s).

## Findings
- 60/110 files in `val/` and `test/` are stale training images from earlier runs of step 02; all checkpoints were trained or evaluated on them.
- Strong signs of patient-level leakage in the clean lists (19/50 test images have a same-fold neighbouring index in train).
- EXP-000: on clean data, the predicted-ROI pipeline (0.46) is far below the full-image classifier (0.80); the U-Net Dice is 0.41.
- The local PyTorch is CPU-only; the GPU has 4 GB.

## Outcomes
- Phase 1 report and Phase 2 setup committed and pushed (see CHANGELOG).
- Hooks tested by piping synthetic stdin through the exact `settings.json` commands. All 8 scenarios passed: SessionStart prints HANDOFF and writes state; Stop returns 0 when only logs changed, 2 for a new non-log file, 0 when `stop_hook_active=true`, 2 when only HANDOFF was updated, 0 when both logs were updated, 2 for a file committed during the session and edited after the logs, and 0 on malformed stdin. The hooks become active in the next session.
- Waiting for Kareem's approval to start Phase 3.
