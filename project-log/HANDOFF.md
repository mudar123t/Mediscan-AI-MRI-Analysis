# HANDOFF — current state

**Last updated:** 2026-11-19 (end of Phase 3, before Kareem's approval)
**Active phase:** Phase 3 done → waiting at the STOP gate (discussion). Next: **Phase 4 (SOTA research)** after approval.
**Branch:** `research/phase3-future-analysis` (merged to `main` by fast-forward). Baseline tag: `v1.0-semester1` = `390d7a0`.

## Where things stand
- Phase 1 audit: [00-project-understanding.md](00-project-understanding.md). Phase 2 setup: see the CHANGELOG.
- Phase 3: every `future.txt` idea is analysed in [ideas/idea-matrix.md](ideas/idea-matrix.md). Conclusion: ~80 % is baseline strengthening and evaluation hygiene (necessary, not novel). Research-flavoured candidates: **E13** uncertainty-gated pipeline (strongest), **D12** multi-task, **D11** context-preserving multi-input. **B6** (sub-region segmentation) should be dropped because the masks are binary.
- Semester-1 numbers: [EXP-000](experiments/EXP-000-semester1-checkpoint-eval.md). Clean test: Dice 0.411; acc full 0.800, GT-ROI 0.740, predicted-ROI 0.460. These are still optimistic, because 43/50 clean test images come from training patients.

## Key facts every agent must know
- **Raw data:** `dataset/dataset/` (git-ignored). Contents: `data/1.mat`…`3064.mat`, `cvind.mat`, `README.txt`. 3,064 slices, 233 patients (82 meningioma, 89 glioma, 62 pituitary), 1–38 slices per patient, PID present, binary masks, 15 images at 256² and the rest at 512². `cvind.mat` folds are patient-disjoint.
- The old converted PNGs are in `dataset/output/` (min-max uint8, interpolated masks); the pilot is in `data/pilot/`. Both are git-ignored.
- The semester-1 evaluation is invalid (contaminated directories + patient leakage).
- The local PyTorch is **CPU-only** (`2.9.1+cpu`). The GPU is an RTX 3050 Laptop with **4 GB**.
- `docs/` stays local (ADR-0002, accepted). Semester-1 checkpoints stay local only (no Release).
- Rule: no code, model or training changes until Phase 6 is approved.

## Next steps
1. Kareem discusses the Phase 3 analysis and approves.
2. Phase 4: SOTA research. Check the novelty of E13, D11 and D12 on this task and dataset. Cover the unexplored angles (XAI, foundation models, robustness, report generation, edge, HITL demo). Find datasets for external validation or replacement. Verify the Figshare licence and the semester-1 references. Outputs go to `research/` + `research-summary.md`.
3. Phase 5: score and select.

## Open questions / blockers
- None blocking. Phase 4 decides whether to use subagents for parallel literature search by topic (moderate parallelism, per rule 8).
- GitHub still reports the old repo size until its GC runs (no action).

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local (accepted) · ADR-0003 git workflow · ADR-0004 logging + hooks.
