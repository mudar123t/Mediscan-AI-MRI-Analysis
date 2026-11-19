# Session 2026-11-19 — Phase 6 M0.2: data pipeline

## Goals
Remove the README authors section (Kareem's request). Build the leak-free data pipeline and run EXP-001.

## Actions
- `3923a2a` docs: removed the README "Authors" section (Kareem, 2026-11-19). Names remain in older commits; history was not rewritten.
- Branch `feature/m0-data-pipeline`.
- New: `src/mediscan/data/figshare.py` (reader, transpose, border rasterisation, normalisation, resize), `splits.py` (fold rule + `LeakageError`), `datasets.py` (in-RAM seg/cls datasets from split rows), `augment.py` (albumentations; `"none"` for the old recipe), `scripts/prepare_figshare.py`, `configs/data_figshare.yaml`, `tests/test_data_pipeline.py`.
- Ran the conversion → `data/processed/figshare_256/` (384 MB, ignored) and `data/splits/figshare_fold{0..4}.json` (tracked).
- EXP-001 log, sample figure, report.

## Findings
See EXP-001: 3,064/233, 0 cross-fold patients, 0 empty masks, border IoU median 0.970. Semester-1 images were transposed. The dataset has no pixel spacing (HD95 in pixels; plan updated). A unit test caught a float64 promotion bug before any training.

## Outcomes
M0.2 complete; 19 tests pass. Next: M0.3 segmentation (`exp/m0-segmentation`, EXP-002).
