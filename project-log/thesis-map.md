# Thesis Map

Maps log entries and findings to the semester-2 thesis sections. Semester-1 thesis (latest): `docs/final report1.pdf` (2026-01-08, local only).

| Thesis section | Source (log / file / commit) | Content to use | Status |
|---|---|---|---|
| 1 Introduction — motivation for the semester-2 contribution | `university-requirements.md` | Committee asked for a creative addition to a common pipeline | pending |
| 2 Literature review | `research/literature-matrix.md`, `research/research-summary.md` | Verified SOTA; replace placeholder references from semester 1 | Phase 4 |
| 3 Methodology — data | `00-project-understanding.md §3` | Correct dataset: Figshare / Cheng et al., 3,064 slices, 233 patients (fix wrong Kaggle citation) | ready |
| 3 Methodology — evaluation protocol | `00-project-understanding.md §8`, `EXP-000` | Why a patient-wise split is required; leakage audit method (stale files, CV-fold adjacency, NCC) | ready |
| 2/3 Discussion of `future.txt` roadmap | `ideas/idea-matrix.md` | Which planned improvements are hygiene vs contribution; why B6 was dropped | ready |
| 4 Experiments — baseline re-evaluation | `experiments/EXP-000-semester1-checkpoint-eval.md` | Contaminated vs clean metrics; CIs | ready |
| 5 Results | `experiments/results-summary.md` | Master table | ongoing |
| 6 Discussion — limitations of semester 1 | `00-project-understanding.md §8`, `EXP-000` conclusions | Leakage inflated accuracy by 0.11 (full image); predicted-ROI pipeline ≈ majority class | ready |
| Appendix — reproducibility / engineering | `decisions/ADR-0001..0004`, `CHANGELOG.md` | Repo cleanup, workflow, logging, hooks | ready |
