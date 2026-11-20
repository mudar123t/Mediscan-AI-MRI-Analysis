# Thesis Map

Maps log entries and findings to the semester-2 thesis sections. Semester-1 thesis (latest): `docs/final report1.pdf` (2026-01-08, local only).

| Thesis section | Source (log / file / commit) | Content to use | Status |
|---|---|---|---|
| 1 Introduction — motivation for the semester-2 contribution | `university-requirements.md` | Committee asked for a creative addition to a common pipeline | pending |
| 2 Literature review | `research/literature-matrix.md`, `research/research-summary.md`, `research/topics/T1–T4` | Verified SOTA (116 refs), trends, novelty landscape | ready |
| References (all chapters) | `research/references.bib`, `research/topics/T4 §6` | **Rebuild the reference list**: 0/12 semester-1 refs correct as cited; fix [2]–[10]; remove [11] (not found); **replace [12] with Abdusalomov et al. 2023, *Cancers* 15:4172, doi:10.3390/cancers15164172** (Kareem does not know the original source; verified by T4 via PMC10453020; its numbers match the claim) | ready |
| 3 Methodology — evaluation standard | `research-summary.md §4`, T4 §5 (Metrics Reloaded, CLAIM 2024) | Metrics, CIs, external test, CLAIM appendix | ready |
| 3 Methodology — datasets | T4 §1–3 | Figshare CC BY 4.0 citation; BDNeuro-MRI external test; why not Kaggle merges | ready |
| 3 Methodology — data | `00-project-understanding.md §3` | Correct dataset: Figshare / Cheng et al., 3,064 slices, 233 patients (fix wrong Kaggle citation) | ready |
| 3 Methodology — evaluation protocol | `00-project-understanding.md §8`, `EXP-000` | Why a patient-wise split is required; leakage audit method (stale files, CV-fold adjacency, NCC) | ready |
| 2/3 Discussion of `future.txt` roadmap | `ideas/idea-matrix.md` | Which planned improvements are hygiene vs contribution; why B6 was dropped | ready |
| 4 Experiments — baseline re-evaluation | `experiments/EXP-000-semester1-checkpoint-eval.md` | Contaminated vs clean metrics; CIs | ready |
| 3 Methodology — data pipeline | `experiments/EXP-001-data-pipeline.md`, `src/mediscan/data/` | Conversion (transpose, percentile normalisation, nearest-neighbour masks), orientation check vs tumorBorder (median IoU 0.970), fold composition tables, split rule (val = (k+1) mod 5) | ready |
| 6 Discussion — semester-1 preprocessing flaws | EXP-001 findings 2 | Semester-1 images were transposed and masks interpolated | ready |
| 4 Experiments — segmentation baseline | `experiments/EXP-002-segmentation.md`, `results.json` | Recipe vs encoder ablation (a/b/c), pooled 5-fold OOF Dice/IoU/HD95/NSD with patient CIs, paired tests, per class, LCC ablation, training cost | ready |
| 4 Experiments — classification baseline and views | `experiments/EXP-003-classification.md`, `results.json` | SimpleCNN old vs ResNet-18/EffNet-B0; views full / masked ROI / crop 0–32 px × GT vs predicted masks; paired patient bootstrap; temperature scaling | ready |
| 6 Discussion — why semester-1 seg-guided classification failed | EXP-003 conclusions 3–4 | Masked ROI −0.079 vs full; context rises with margin; predicted masks cost 1–2 points → motivates R1 | ready |
| 4 Experiments — when segmentation helps (R3) | `experiments/EXP-004-mask-quality.md`, `controlled.csv`, `predicted_by_dice.csv`, figure | Controlled mask degradation × context margin; real masks by Dice bin | ready |
| 4 Experiments — uncertainty signals | `experiments/EXP-005-uncertainty.md`, `auroc_table.csv` | MC/TTA/ensemble scores; seg-failure vs classification-error AUROC; "adds to MSP" (pooling fix noted) | ready |
| 4/5 Contribution — routing and conformal referral (R1, R7) | `experiments/EXP-006-router.md`, `risk_coverage.png` | Router vs fixed views, risk–coverage, oracle head-room, τ1 instability, class-conditional CP gap | ready |
| 4/6 Extras — explainability as a diagnostic (R4) | `experiments/EXP-010-xai-diagnostic.md`, `attribution_shares.png`, `gradcam_examples.png` | Where each classifier looks (crop ~70 % on tumour+border vs full ~22 %, 4.3× chance); no link to errors beyond MSP; weaker routing signal than ensemble disagreement; sellar-meningioma example | ready |
| 5 Results — robustness summary with CIs; τ1 stabilisation (post-hoc, pre-specified) | `experiments/EXP-006b-stable-tau1.md`, `results.json` | Worst-case/mean over 13 shifts with paired CIs; pooled-val τ1 failure and flat-objective explanation | ready |
| 5 Results — robustness to shift / external (H-B3) | `experiments/EXP-008-shift.md`, `EXP-009-external.md`, `accuracy_vs_severity.png` | Severity curves, worst-case summary, external test, BDNeuro duplicate audit | ready |
| 6 Discussion — negative results and limits | EXP-005/006 conclusions, EXP-009 limitation | Seg uncertainty ≠ classifier error; router value is robustness, not in-distribution gain; image-level CIs on BDNeuro | ready |
| 5 Results | `experiments/results-summary.md` | Master table | ongoing |
| 6 Discussion — limitations of semester 1 | `00-project-understanding.md §8`, `EXP-000` conclusions | Leakage inflated accuracy by 0.11 (full image); predicted-ROI pipeline ≈ majority class | ready |
| Appendix — reproducibility / engineering | `decisions/ADR-0001..0004`, `CHANGELOG.md` | Repo cleanup, workflow, logging, hooks | ready |
