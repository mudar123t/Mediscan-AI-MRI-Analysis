# Semester-1 code (frozen)

This folder holds the semester-1 pipeline scripts as they were at tag `v1.0-semester1`, moved here with `git mv` in Milestone M0.1 (2026-11-19) so their history is kept. They are **not** used by the semester-2 pipeline.

- `conv.py`: `.mat` → PNG converter.
- `scripts/01..10_*.py`: pilot data preparation, U-Net training, ROI building, classifier training and evaluation. `03_check_pilot_data.py` was renamed from `03_check_pilot_data,py` (stray comma).
- `notebooks/01_dataset_explore.ipynb`: empty (0 bytes) in the original repository.

Known issues (see `project-log/00-project-understanding.md` and `project-log/experiments/EXP-000-semester1-checkpoint-eval.md`):
- Hardcoded absolute paths (`C:\Users\shawa\Desktop\final project`).
- Image-level split with patient leakage; val/test directories contaminated with training images.
- `conv.py` resizes masks with an interpolating resampler.

The models they use (`mediscan.segmentation.unet.UNet`, `mediscan.classification.cnn.SimpleCNN`) stay in `src/mediscan/` because semester 2 retrains them under the new protocol as the fair baseline.
