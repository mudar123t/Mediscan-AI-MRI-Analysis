# Session 2026-11-19 — Phase 6 M0.1: environment and repo structure

## Goals
Record the plan approval; build a reproducible environment (CUDA PyTorch) and the semester-2 repo structure.

## Actions
- ADR-0007: Kareem approved Q1–Q5 (Q4 "the better one" → Gradio). Plan status → approved.
- Branch `chore/m0-env-structure`.
- `git mv` semester-1 scripts, `conv.py`, `notebooks/` → `legacy/semester1/` (+ README); fixed the `03_check_pilot_data,py` file name.
- `.venv` (Python 3.12.2) with torch 2.9.1+cu128; remaining packages installed; `pip check` clean.
- Added `pyproject.toml` (editable install), pinned `requirements.txt`, `requirements-lock.txt`, `configs/base.yaml`, `src/mediscan/utils/repro.py`, package `__init__.py` files, `tests/` (6 tests).
- README: dataset section corrected (it said Kaggle) → Figshare DOI, CC BY 4.0, Cheng 2015 citation, setup steps. CLAUDE.md commands/folder map updated.
- `project-log/environment.md` with versions and a GPU probe.

## Findings
- `torch.cuda.is_available()` = True on the RTX 3050 Laptop (capability 8.6, CUDA runtime 12.8) [V].
- GPU probe: semester-1 UNet, batch 8, 256², AMP: 0.102 s/step, 863 MiB peak VRAM [V] → 4 GB is enough for this model; the larger encoder is measured in EXP-002.
- The README "Authors" section (two students + supervisor) conflicts with the solo-presentation rule → question to Kareem (not changed).

## Outcomes
M0.1 complete pending commit/push approval display. Next: M0.2 data pipeline on `feature/m0-data-pipeline`.
