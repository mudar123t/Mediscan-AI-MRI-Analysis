# Session 2026-11-19 — Phase 3: future.txt analysis

## Goals
Analyse every idea in `future.txt`: restate it, assess feasibility on our hardware and data, effort, real novelty, risks and dependencies. Log the results in `ideas/idea-matrix.md`. No implementation.

## Actions
- Recorded Kareem's Phase 2 answers: docs stay local; checkpoints local only; raw data is at `dataset/dataset/`.
- Read all 3,064 raw `.mat` files (read-only) to get facts that determine feasibility: PIDs, labels, mask type, image sizes, `cvind` fold purity. Rechecked the semester-1 split lists against the PIDs.
- Wrote the per-idea analysis and the qualitative summary matrix.

## Findings
- Masks are binary → B6 (edema/core/enhancing) is not feasible on this dataset [V].
- Patient leakage in the semester-1 clean lists is confirmed: 43/50 test and 41/50 val [V].
- `cvind.mat` gives a ready-made patient-wise 5-fold CV [V].
- Most `future.txt` items are standard practice [I]. E13, D12 and D11 are the research-flavoured candidates; their novelty is unverified until Phase 4.

## Outcomes
- `ideas/idea-matrix.md` (Phase 3 version). Logs updated (HANDOFF, CHANGELOG, audit, EXP-000, presentation bank, thesis map, ADR-0001/0002).
- Waiting at the Phase 3 STOP gate for discussion.
