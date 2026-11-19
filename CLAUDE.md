# CLAUDE.md — Mediscan AI

**Start every session by reading `project-log/HANDOFF.md`** (a SessionStart hook prints it). Update HANDOFF.md and CHANGELOG.md after every change to code or plans; a Stop hook enforces this.

## Project
Graduation project (Software Engineering, 2 semesters). Brain-MRI pipeline: a U-Net segments the tumor (binary), then a CNN classifies the type (glioma / meningioma / pituitary). Semester 1 is done (tag `v1.0-semester1`). Semester 2 goal: add a **creative, novel, rigorously evaluated** contribution (the committee says the core idea is common; the supervisor may co-publish). Target grade AA.
- Data: Figshare brain tumor dataset (Cheng et al.), 3,064 T1-CE slices, 233 patients, binary masks and patient IDs. Raw `.mat` files in `dataset/dataset/data/`, patient-wise 5-fold indices in `dataset/dataset/cvind.mat`. Not in git.
- Known semester-1 issues: val/test contaminated by training images; image-level (not patient-wise) split; see `project-log/00-project-understanding.md` and `experiments/EXP-000`.

## Tech stack
Python 3.12 · PyTorch (currently a CPU-only build is installed; CUDA build pending) · OpenCV · NumPy · scikit-learn · h5py · matplotlib.
Hardware: Dell G15, RTX 3050 Laptop **4 GB VRAM**, Ryzen 5 6800H. Never assume cloud; if an idea needs Colab/Kaggle, say so and estimate cost/effort.

## Key commands (current state; scripts still hold hardcoded paths, to be fixed in Phase 6)
```bash
pip install -r requirements.txt h5py          # setup (requirements unpinned)
export PYTHONPATH=src                          # package is not installed
python conv.py                                 # .mat -> PNG (edit input/output paths)
python scripts/02_prepare_pilot_data.py        # pilot subset + splits
python scripts/05_train_unet_pilot.py          # train U-Net
python scripts/07_build_roi_dataset.py         # ROI sets (GT + predicted)
python scripts/09_train_classifier.py          # train 3 CNNs
python scripts/10_eval_classifier.py           # test accuracy + confusion matrix
python project-log/experiments/EXP-000-semester1-checkpoint-eval/eval_semester1.py  # audit eval
```

## Folder map
```
src/mediscan/{preprocessing,data,segmentation,classification}/   library code
scripts/01..10_*.py      numbered pipeline steps
conv.py                  raw .mat -> PNG converter
data/pilot/splits/       semester-1 split lists (tracked); images are ignored
dataset/, data/**, outputs/, docs/   local only (git-ignored)
images/                  README figures
project-log/             project memory (see below)
.claude/                 settings.json + hooks/ (tracked); state/ (ignored)
```

## Coding conventions
- English everywhere: code, comments, logs, docs.
- Type hints and docstrings; small pure functions in `src/mediscan`; scripts stay thin.
- No hardcoded absolute paths in new code; use configs (YAML/JSON) saved with every run.
- Reproducibility: fixed seeds, saved config, recorded environment (versions, GPU), split files used by the loaders.
- Report metrics with per-class values and confidence intervals; never report on images that were seen in training.

## Git workflow (ADR-0003)
- `main` stays stable. Work on `feature/*`, `research/*`, `exp/*`, `chore/*` branches.
- Conventional Commits: feat, fix, docs, exp, refactor, chore, test. Keep commits small and logical.
- **No AI co-author or attribution lines** in commits or PRs.
- Show what will be pushed before every push.
- GitHub repo is presented as a solo project: do not add contributor or team sections.

## NON-NEGOTIABLE RULES
1. **Do not assume.** Ask Kareem when something is unclear (AskUserQuestion). Label claims VERIFIED / INFERRED / UNKNOWN.
2. **Phases with hard stops.** At every STOP: summarize, list open questions, wait for explicit approval. Never skip a gate.
3. **No code, model or training-script changes until Phase 6 is approved.** Phases 1–5 only read, set up repo and logs, research and discuss.
4. **No fabrication.** Never invent papers, citations, metrics, dataset facts or results. Every logged paper needs a DOI / arXiv / publisher link that was actually opened; otherwise mark `[UNVERIFIED]` or drop it. Every number comes from a file, a run or a cited source.
5. **Git safety.** Never force-push, rewrite history, delete branches or delete files without asking. Show what will be pushed before each push.
6. **Log everything** (see below). The logs are the project's memory and the thesis raw material.
7. **Discuss, don't decide.** Present options with trade-offs and a recommendation; Kareem chooses.
8. Use subagents in parallel only where they clearly help (e.g. literature search by topic). Keep it moderate.

## Phases
1 Audit ✅ · 2 Repo/memory/logging setup ✅ · 3 future.txt analysis ✅ · 4 SOTA research · 5 Idea selection · 6 Implementation plan + execution. Current phase: see HANDOFF.md.

## Logging system (`project-log/`)
| File | Purpose |
|---|---|
| `HANDOFF.md` | ALWAYS current: state, active phase, last action, next steps, open questions, blockers, key decisions, timestamp |
| `CHANGELOG.md` | Dated entries: what changed, why, commit hashes |
| `sessions/YYYY-MM-DD-<topic>.md` | One per working session: goals, actions, findings, outcomes |
| `decisions/ADR-XXXX-<title>.md` | Context, options, decision, consequences, date |
| `research/` | `literature-matrix.md`, `papers/`, `references.bib`, `research-summary.md` |
| `experiments/` | `EXP-XXX-<name>.md` (hypothesis, config, split, seed, hardware, runtime, results, vs baseline, conclusion) + `results-summary.md` |
| `ideas/idea-matrix.md` | Every candidate improvement with scores |
| `thesis-map.md` | Log entry → thesis section |
| `presentation-bank.md` | Best results, figures, wow points |
| `university-requirements.md` | University requests and how each is addressed |

Logs must be factual, dated and specific (numbers, paths, commit hashes). No filler.
