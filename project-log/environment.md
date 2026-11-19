# Environment record

Updated at every milestone. Each run also writes its own `environment.json` (via `mediscan.utils.repro.create_run_dir`).

## M0.1 — 2026-11-19 [V: `environment_info()` output and `pip freeze`]
| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| CPU / threads / RAM | AMD Ryzen 5 6800H (AMD64 Family 25 Model 68) / 12 / 15.2 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4,095 MiB, compute capability 8.6, driver 610.60 (CUDA UMD 13.3) |
| Python | 3.12.2 in `.venv` (created with `py -3.12`; the `python` command is the Microsoft Store 3.12.10, not used) |
| PyTorch | 2.9.1+cu128, torchvision 0.24.1+cu128, CUDA runtime 12.8, cuDNN 9.10.02 (91002) |
| Key packages | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, segmentation-models-pytorch 0.6.0, timm 1.0.31, albumentations 2.0.8, opencv-python-headless 5.1.0.95, h5py 3.16.0, ImageHash 4.3.2, pytest 9.2.0 |
| Pins | `requirements.txt` (direct), `requirements-lock.txt` (full freeze) |
| Tests | `pytest`: 6 passed (incl. GPU AMP forward pass), 35.9 s |

### GPU probe (rough, not an experiment)
Semester-1 `UNet` (7.76 M params), random tensors, batch 8, 1×256×256, Adam, AMP fp16, cuDNN non-deterministic, 20 timed steps after 5 warm-up steps:
- **0.102 s/step**, peak allocated VRAM **863 MiB**.
- [I] At about 1,840 training slices per fold (3 of 5 folds), one epoch would be about 230 steps ≈ 23 s of GPU compute, before data loading. Real epoch times are measured in EXP-002.
