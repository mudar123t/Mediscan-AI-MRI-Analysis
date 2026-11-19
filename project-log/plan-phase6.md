# Phase 6 — Implementation Plan (Milestone 0 + Option B; extras as placeholders)

- **Date:** 2026-11-19
- **Status:** **APPROVED 2026-11-19** ([ADR-0007](decisions/ADR-0007-phase6-plan-approval.md)): Q1 yes, Q2 yes, Q3 yes (256²), Q4 Gradio, Q5 yes. Execution started with M0.1.
- **Decision basis:** [ADR-0006](decisions/ADR-0006-semester2-contribution.md), [phase5-options.md](ideas/phase5-options.md), [research-summary.md](research/research-summary.md) §4 (evaluation standard).
- **Labels:** [V] verified · [I] inference/estimate · [U] unknown (to be measured). All effort figures are [I] estimates.

## 0. Verified environment facts (2026-11-19)
| Item | Value | Source |
|---|---|---|
| GPU | RTX 3050 Laptop, 4,096 MiB, driver 610.60, CUDA UMD 13.3 | `nvidia-smi` [V] |
| CPU / RAM | 12 logical threads / 15.2 GB | `nproc`, CIM [V] |
| Disk C: free | **37 GB** (93 % used) | `df` [V] → large external datasets are constrained |
| Python / torch | 3.12.10 / 2.9.1+**cpu** | pip [V] |
| Installed | numpy, pandas, scipy, scikit-learn, opencv, h5py, matplotlib | [V] |
| Missing | pyyaml, albumentations, timm, segmentation-models-pytorch, imagehash, pytest, gradio | [V] |

---

## 1. Milestones overview
| # | Milestone | Content | Branch(es) | Effort [I] | Gate |
|---|---|---|---|---|---|
| **M0.1** | Environment & repo structure | CUDA torch, pinned deps, `pyproject.toml`, configs, tests skeleton, legacy scripts moved | `chore/m0-env-structure` | 1–2 d | — |
| **M0.2** | Data pipeline | `.mat` → processed 256² set + `metadata.csv` (PID, label, fold, area); patient-disjoint splits; leakage tests | `feature/m0-data-pipeline` | 2 d | — |
| **M0.3** | Segmentation baseline | Semester-1 U-Net retrained on new protocol (fair "old recipe") vs improved U-Net (pretrained encoder, dropout for MC); 5 folds | `exp/m0-segmentation` | 3–4 d + GPU | — |
| **M0.4** | Classification baseline + views | Pretrained ResNet-18/EffNet-B0; views: full, masked ROI, context crop; GT vs predicted masks; calibration | `exp/m0-classification` | 3 d + GPU | **STOP: report vs baseline** |
| **M1.1** | R3 study | Accuracy vs mask quality × context margin | `exp/m1-r3-seg-helps` | 3 d | — |
| **M1.2** | Uncertainty signals | MC dropout, TTA, seed ensemble → image-level scores; failure-detection AUROC | `exp/m1-uncertainty` | 3 d | — |
| **M1.3** | Router + conformal referral | Policies, baselines, risk–coverage, R7 | `feature/m1-router` | 3–4 d | — |
| **M1.4** | Shift & external | Synthetic corruptions; BDNeuro-MRI external test (after pHash audit) | `exp/m1-shift-external` | 3 d | **STOP: report H-B1..H-B3** |
| **M2** | Demo | Local web demo of the routing pipeline | `feature/m2-demo` | 3 d | **STOP: demo review** |
| M3 | Extras (R4 / R2 / R5) | **Decided with Kareem after M1** (ADR-0006) | — | — | decision |
| T | Thesis & presentation | Continuous: thesis-map, presentation-bank, figures, CLAIM checklist, rebuilt references | `docs/*` | ongoing | — |

Each milestone: small commits → update HANDOFF/CHANGELOG/EXP logs/thesis-map/presentation-bank → show the push → push → merge to `main` (fast-forward) when green.

---

## 2. Files to create / change

### 2.1 Repo structure (M0.1)
| Path | Action | Purpose |
|---|---|---|
| `legacy/semester1/` | **`git mv`** of `scripts/01..10_*.py`, `conv.py`, `notebooks/` (history preserved) | Freeze semester-1 code; keeps `v1.0-semester1` reproducible. *Needs approval (Q1)* |
| `src/mediscan/` (existing) | Keep `unet.py`, `cnn.py`, `losses.py` (used for the "old recipe" baseline); add new modules below | Library |
| `pyproject.toml` | new | `pip install -e .` (removes the `PYTHONPATH` hack) |
| `requirements.txt` | rewrite, **pinned** | + `requirements-lock.txt` (pip freeze) per milestone |
| `configs/*.yaml` | new | One YAML per experiment; copied into every run folder |
| `tests/` | new (pytest) | Data, split, metric and router unit tests |
| `runs/` | new, **git-ignored** | Per-run outputs (checkpoints, logs, predictions) |
| `README.md` | update | Semester-2 setup, dataset download instructions (Figshare DOI, CC BY 4.0 attribution) |

### 2.2 New library modules (`src/mediscan/`)
| Module | Content |
|---|---|
| `data/figshare.py` | Read `.mat` (h5py; handle the MATLAB transpose), PID, label, mask; min-max (or 1–99 percentile) normalisation; resize image (area) / mask (**nearest**); 15 native-256² images handled |
| `data/splits.py` | Folds from `cvind.mat`; outer test fold k, **val fold (k+1) mod 5**, train = other 3 (60/20/20 by patient); assert patient-disjointness |
| `data/datasets.py` | Seg + cls datasets reading `metadata.csv` and split files (never globbing directories) |
| `data/augment.py` | Albumentations: flips (L/R), small rotations/scale, elastic (light), brightness/contrast/gamma; identical geometric transforms for image and mask |
| `models/seg.py` | (a) semester-1 `UNet` (old recipe); (b) U-Net with ImageNet-pretrained ResNet-34 encoder (segmentation-models-pytorch) + decoder dropout for MC dropout |
| `models/cls.py` | ResNet-18 / EfficientNet-B0 (timm/torchvision, pretrained), 1→3-channel adaptation |
| `views.py` | Classifier input views: `full`, `masked_roi` (semester-1 style), `context_crop(margin)` (bounding box + margin, resized), `mask_channel` (D11 ablation) |
| `metrics/seg.py` | Dice, IoU, HD95 (mm), NSD (τ = 2 px [I]) with an explicit empty-mask rule; per image and per patient |
| `metrics/cls.py` | Balanced accuracy, macro-F1, MCC, AUROC (OvR), per-class recall, confusion matrix, ECE (15 bins), Brier |
| `metrics/selective.py` | Risk–coverage, AURC, E-AURC, AUGRC, accuracy@coverage, referral rate |
| `stats.py` | Patient-level bootstrap CIs (2,000 resamples), paired bootstrap of differences, McNemar |
| `uncertainty.py` | MC dropout (T=20), TTA (T=8 flips/rotations), seed-ensemble; image-level aggregates (mean entropy in/around mask, MC-Dice agreement, area CV, mean max-prob) |
| `calibration.py` | Temperature scaling (fit on val only) |
| `router.py` | Routing policies (see §3.4) + split conformal prediction (R7) |
| `train/*.py`, `eval/*.py` | Thin scripts: `train_seg`, `train_cls`, `predict`, `evaluate`, `route` — all config-driven, AMP, seeded, write `runs/<exp>/<fold>/` |

### 2.3 Logs
`project-log/experiments/EXP-001…EXP-010` (one per experiment below) + `results-summary.md`, `environment.md` (versions, GPU, timings), HANDOFF, CHANGELOG, thesis-map, presentation-bank after every meaningful change.

---

## 3. Experiment design

### 3.1 Protocol (applies to every experiment)
- **Data:** full Figshare set, 3,064 slices / 233 patients [V], 256×256.
- **Splits:** 5 outer folds from `cvind.mat` (patient-disjoint [V]). For test fold k: val = fold (k+1) mod 5 (early stopping, temperature, router thresholds, conformal calibration); train = the other 3. **Test folds are never used for any choice.**
- **Seeds:** fixed (e.g. 0, 1, 2 for the seed ensemble); `torch.backends.cudnn.deterministic` where feasible; seed stored in config.
- **Reporting:** out-of-fold predictions pooled over 5 folds plus per-fold mean ± sd; 95 % CIs by patient-level bootstrap; paired comparisons by patient-level paired bootstrap (and McNemar for classification); per-class metrics; confusion matrices. Metrics Reloaded + CLAIM 2024 (T4 §5).
- **Leakage guards (tests):** no PID in two of train/val/test; pHash near-duplicate check between external data and Figshare; outputs written to fresh run folders (fixes the semester-1 stale-file bug).
- **Fold-ensemble caveat:** the 5 fold models **must not** be ensembled on a test fold (4 of them trained on its patients). Use a **seed ensemble within each fold** instead [I — a design catch].

### 3.2 Milestone 0 experiments
| EXP | Question | Design | Metrics | Hypothesis / target |
|---|---|---|---|---|
| EXP-001 | Is the new data pipeline correct and leak-free? | Conversion stats, mask binary check, orientation check vs `tumorBorder`, class/patient counts per fold, timing | counts, assertions | 233 patients, 3,064 slices, 0 PID overlap |
| EXP-002 | How much does the recipe matter for segmentation? | (a) semester-1 UNet, old recipe (5 ep) on new protocol = **fair semester-1 baseline**; (b) same UNet, new recipe; (c) pretrained-encoder U-Net, new recipe | Dice, IoU, HD95, NSD (per image/patient), per class | target Dice ≈ 0.85–0.93 (Avazov 2024 [V]); measure epoch time [U] |
| EXP-003 | Classification baseline and views | SimpleCNN old recipe (fair baseline) vs ResNet-18/EffNet-B0; views full / masked ROI (GT, pred) / context crop (GT, pred, margins 0/8/16/32 px) | bal. acc, macro-F1, MCC, AUROC, ECE before/after TS | target ≥ Cheng 2015 91.28 % patient-wise [V]; reference Deepak 2019 98 % [V] |

**Comparison with semester 1:** EXP-000 values (clean50) are reported as the *historical* reference only (different subset, leaky split). The **statistically valid baseline** is the semester-1 architecture retrained on the new protocol (EXP-002a, EXP-003 SimpleCNN). This re-verifies semester 1 as requested.

### 3.3 Milestone 1 experiments (Option B)
| EXP | Question | Design | Metrics | Hypothesis |
|---|---|---|---|---|
| EXP-004 (R3) | When does segmentation help classification? | Classifier accuracy vs **mask quality** (GT masks degraded by erosion/dilation/shift/random blobs to controlled Dice bins 1.0→0.3, plus predicted masks) × **context** (masked ROI, crop margins 0/8/16/32 px, full) | bal. acc curves with CIs | Accuracy falls with mask quality; context margin softens the fall (Cheng 2015 [V] direction) |
| EXP-005 | Which uncertainty signal predicts failure? | MC dropout (T=20), TTA (T=8), seed ensemble (3) → image-level scores | AUROC for (a) seg failure (Dice < 0.7), (b) **downstream classification error**; correlation with Dice | ≥ 1 seg-uncertainty score predicts classification error better than chance and adds to MSP (Jaeger 2023 baseline [V]) |
| EXP-006 | Does routing help? (R1) | Policies: always-full, always-context-crop(pred), always-masked-ROI(pred), MSP gate (Sharma-style), seg-uncertainty router, combined router, oracle router; thresholds fit on val | bal. acc @100 %, risk–coverage, AURC, E-AURC, AUGRC, acc@95/90/80 %, referral rate | **H-B1** router ≥ best single view at 100 %; **H-B2** lower AURC/AUGRC than MSP gate |
| EXP-007 | Conformal referral (R7) | Split CP (val as calibration) on routed classifier, α ∈ {0.05, 0.1} | empirical coverage, mean set size, referral rate (set size > 1) | coverage ≈ 1−α |
| EXP-008 | Synthetic shift | Test-fold images corrupted (Gaussian noise, blur, contrast/gamma, bias field) at 3 severities | all of EXP-006 per severity | **H-B3** router benefit grows with severity |
| EXP-009 | External test | BDNeuro-MRI (glioma/meningioma/pituitary only), after pHash audit vs Figshare; trained fold models applied unchanged | bal. acc, AUROC, ECE, AURC; referral rate | performance drops; routing/referral gain larger than in-distribution (H-B3) |

### 3.4 Router definition (to be fixed in EXP-006, then frozen)
Per image: compute segmentation-uncertainty score *u* (best signal from EXP-005) and classifier confidence *c* for each view.
- *u* ≤ τ₁ → **context-crop view** (margin from EXP-004).
- *u* > τ₁ → **full-image view**.
- If the chosen view's calibrated confidence < τ₂, or the conformal set size > 1 → **refer**.
- τ₁ and τ₂ are tuned on val by minimising AURC (or risk at a target coverage). This yields a 3-way decision, with an oracle router as the upper bound. A learned router (small logistic model on [u, c_views]) is tried only as an ablation, to avoid overfitting.

### 3.5 Milestone 2 — demo
Local web app (**Gradio** recommended — *Q4*). Upload a slice → mask overlay + uncertainty map → routing decision with its reason → class + calibrated confidence → "Refer" flag. Runs on the laptop (CPU fallback).

---

## 4. Compute & data budget [I unless noted]
- Data in RAM: 3,064 × 256² uint8 ≈ 0.2 GB → fits in 15.2 GB [V RAM].
- VRAM: U-Net (ResNet-34 encoder) at 256², batch 8, AMP ≈ fits 4 GB [I]; batch 4 + gradient accumulation as a fallback. ResNet-18 classifier at 224–256² batch 32 AMP fits [I].
- Training runs: seg 3 variants × 5 folds (+2 extra seeds for the improved model) ≈ 25 runs; cls ≈ 5 views × 2 masks × 5 folds (+ seeds). **Epoch time [U] is measured in EXP-002**, and the epoch budget (early stopping, patience ~15) is then fixed.
- Disk: processed set < 1 GB; runs/checkpoints ≈ 5–10 GB if all checkpoints are kept → keep best-only. **37 GB free [V]**.
- **External datasets:** BDNeuro-MRI size [U] (check before download). **Pandit pituitary (47.2 GB [V]) is NOT feasible** with 37 GB free; BraTS-MEN size [U]. External *segmentation* tests are therefore optional/deferred.
- **Cloud:** not needed for M0–M2 [I].

## 5. Dependencies to add (needs approval, Q2)
`torch`/`torchvision` CUDA build (cu12x/cu13x wheel matching driver CUDA 13.3), `pyyaml`, `albumentations`, `segmentation-models-pytorch`, `timm`, `imagehash`, `pytest`, `gradio` (M2). All open-source, widely used. Versions pinned in `requirements.txt`; full `pip freeze` in `requirements-lock.txt`. HD95/NSD implemented with scipy distance transforms + unit tests (avoids the heavy MONAI dependency).

## 6. Risks and mitigations
| Risk | Likelihood [I] | Impact | Mitigation |
|---|---|---|---|
| CUDA torch install breaks the environment | low | medium | Install into a **new virtual env** (`.venv`, git-ignored); keep the current system Python untouched; verify `torch.cuda.is_available()` |
| VRAM OOM on 4 GB | medium | low | AMP, batch 4 + grad accumulation, 256², `channels_last` |
| Training too slow | medium | medium | Measure in EXP-002; early stopping; fewer seeds; cloud only as a last resort (stated with cost first) |
| Windows DataLoader worker issues | medium | low | `num_workers` 0–4 with `if __name__ == "__main__"` guards; preload data into RAM |
| Leakage regression | low | **high** | pytest PID-disjointness tests in CI-style pre-commit; split files are the only data source; fresh run dirs |
| Strong U-Net → gate rarely triggers in-distribution (H-B1/H-B2 small) | medium | medium | H-B3 shift/external experiments; report honestly; the R3 study stands on its own |
| Router overfits thresholds | medium | medium | Thresholds only on val folds; simple 2-threshold rule; learned router only as an ablation |
| Novelty pre-empted by later papers | low-medium | high (publication) | Scopus/Scholar re-check before writing; position as "seg-quality-aware view selection + referral" |
| BDNeuro-MRI unusable (size, format, overlap) | low-medium | medium | pHash audit first; fallback: Daffodil set [V T4]; synthetic shift (EXP-008) still tests H-B3 |
| Disk space (37 GB) | medium | medium | Best-only checkpoints; no 47 GB datasets; clean `runs/` of failed runs (with your OK) |
| Orientation/transposition errors from `.mat` | low | medium | EXP-001 checks mask vs `tumorBorder` polygon |

## 7. Definition of done per milestone
- **M0:** EXP-001–003 logged; leak tests pass; new baseline numbers with CIs in `results-summary.md`; comparison to EXP-000 and to literature targets; STOP report.
- **M1:** EXP-004–009 logged; H-B1..H-B3 each marked supported/not supported with evidence; figures for the presentation; STOP report.
- **M2:** demo runs locally from a fresh clone + README instructions; screenshots in the presentation bank; STOP.

## 8. Questions for approval
1. **Q1** Move the semester-1 scripts, `conv.py` and `notebooks/` into `legacy/semester1/` with `git mv` (history kept)? *(Recommended: yes)*
2. **Q2** Install the dependencies in §5 into a new `.venv` with CUDA PyTorch? *(Recommended: yes)*
3. **Q3** Image size 256×256 (VRAM-friendly; Figshare is 512² natively)? *(Recommended: yes; 512² only if time allows, as an ablation)*
4. **Q4** Demo framework: Gradio *(recommended, simplest for ML demos)* or Streamlit?
5. **Q5** Approve the plan as a whole (milestones, experiments, STOP gates at M0, M1, M2)?
