# EXP-001 — Data pipeline sanity and leakage check (M0.2)

- **Date:** 2026-11-19
- **Branch / code:** `feature/m0-data-pipeline`; `scripts/prepare_figshare.py`, `src/mediscan/data/{figshare,splits,datasets,augment}.py`
- **Config:** `configs/data_figshare.yaml` (copied to `data/processed/figshare_256/config.yaml`)
- **Hardware / runtime:** CPU only; full conversion **90.2 s** (first run 136 s, cold file cache)
- **Raw report:** [EXP-001-data-pipeline/report.json](EXP-001-data-pipeline/report.json) (includes the environment record)
- **Figure:** [EXP-001-data-pipeline/samples.png](EXP-001-data-pipeline/samples.png) (4 random slices per class, seed 0, mask contour in red)

## Question
Is the new conversion correct (orientation, masks, labels) and are the patient-wise splits leak-free?

## Method
1. Read all 3,064 `.mat` files with h5py; transpose `image` and `tumorMask` (HDF5 stores MATLAB arrays column-major).
2. **Orientation check:** rasterise the hand-drawn `tumorBorder` polygon (README: `[x1, y1, …]`, 1-based) at native resolution and compute IoU with `tumorMask`.
3. Intensity: per-image clipping to the 0.5–99.5 percentiles, scale to [0, 1], store uint8. Resize 512→256 with area averaging (image) and nearest neighbour (mask). The 15 native 256² slices are not resized.
4. Splits: `cvind.mat` folds (stored 1..5, used 0..4). Test fold k, val fold (k+1) mod 5, train = other 3. `assert_patient_disjoint` runs on every split, and pytest re-checks the tracked split files.

## Results [V: report.json]
| Check | Result |
|---|---|
| Slices / patients | **3,064 / 233** |
| Slices per class | glioma 1,426 · pituitary 930 · meningioma 708 (matches dataset README) |
| Patients per class | glioma 89 · meningioma 82 · pituitary 62; 0 patients with more than one class |
| Native sizes | 512² × 3,049; 256² × 15 |
| Masks | all binary; 0 empty at native resolution; **0 empty after resize** |
| Border-polygon IoU (orientation) | min **0.846**, 1st percentile 0.903, median **0.970**; 0 slices below 0.80 |
| Exact duplicate images (SHA-1 of raw pixels) | **0** |
| Patients in more than one fold | **0** |
| Mask area (fraction of 256² image) | median 1.28 %, IQR 0.65–2.32 %, min 0.10 %, max 9.7 % |

Fold composition (slices, patients; class slices glioma/meningioma/pituitary):
| Fold | Slices | Patients | G / M / P |
|---|---|---|---|
| 0 | 542 | 45 | 245 / 112 / 185 |
| 1 | 679 | 46 | 338 / 167 / 174 |
| 2 | 572 | 48 | 231 / 139 / 202 |
| 3 | 628 | 47 | 325 / 124 / 179 |
| 4 | 643 | 47 | 287 / 166 / 190 |

Split sizes (slices; patients) per outer test fold k:
| k | train | val | test |
|---|---|---|---|
| 0 | 1,843; 142 | 679; 46 | 542; 45 |
| 1 | 1,813; 139 | 572; 48 | 679; 46 |
| 2 | 1,864; 138 | 628; 47 | 572; 48 |
| 3 | 1,793; 139 | 643; 47 | 628; 47 |
| 4 | 1,879; 141 | 542; 45 | 643; 47 |

## Findings
1. **Orientation [V].** The polygon check alone matches either "transpose both" or "transpose neither with x/y swapped" (IoU ≈ 0.94–0.98 on 5 probe files for both). It cannot detect a global transpose. HDF5's column-major storage and the README's (x, y) convention together fix it: transpose. The sample figure confirms correct anatomy (coronal slices upright, sagittal slices facing left, axial slices with the eyes at the top).
2. **Semester-1 images were transposed [V: `legacy/semester1/conv.py` has no transpose].** Image and mask were flipped together, so training was internally consistent, but the anatomy was mirrored across the diagonal. It also resized masks with an interpolating resampler.
3. **Mixed views.** The dataset mixes axial, coronal and sagittal slices (visible in the samples). View is not a metadata field [V: `.mat` keys are PID, image, label, tumorBorder, tumorMask only].
4. **No pixel spacing in the files [V: same key list].** HD95 is therefore reported in **pixels** (at 256²), not mm. The plan is updated accordingly.
5. The tumor occupies a small part of the image (median 1.3 %). This is relevant for loss choice (Dice + BCE) and for the context-crop views.
6. Bug caught by a unit test before use: percentile values (float64) promoted the image to float64. Fixed; rebuilt set differs from the first build in 0.011 % of pixels by one grey level.

## Conclusion
The data pipeline is correct and patient-disjoint. The processed set (384 MB, git-ignored) and the 5 tracked split files are the only data source for M0.3 onward. Tests: `tests/test_data_pipeline.py` (13 tests; 19 in the suite, all passing).
