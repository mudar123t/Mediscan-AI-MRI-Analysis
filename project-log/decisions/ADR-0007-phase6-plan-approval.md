# ADR-0007 — Phase 6 plan approved (Q1–Q5)

- **Date:** 2026-11-19
- **Status:** Accepted
- **Context:** [plan-phase6.md](../plan-phase6.md) asked five questions before any code was written (rule 3).

## Decisions (Kareem, 2026-11-19)
| Q | Question | Answer |
|---|---|---|
| Q1 | Move semester-1 scripts, `conv.py`, `notebooks/` to `legacy/semester1/` with `git mv` | **Yes** |
| Q2 | Install dependencies + CUDA PyTorch into a new `.venv` | **Yes** |
| Q3 | Work at 256×256 (512 only as an ablation) | **Yes** |
| Q4 | Demo framework | **"The better one"** → Gradio (the plan's recommendation: built for ML demos, image inputs/outputs in a few lines, runs locally) |
| Q5 | Approve the plan (milestones, experiments, STOP gates after M0, M1, M2) | **Yes** |

## Consequences
- Phase 6 execution starts with M0.1 on `chore/m0-env-structure`.
- The system Python stays untouched; all semester-2 work runs in `.venv` (git-ignored).
- STOP gates remain after M0 (M0.4), M1 (M1.4) and M2. Extras (R4/R2/R5) are decided after M1 (ADR-0006).
