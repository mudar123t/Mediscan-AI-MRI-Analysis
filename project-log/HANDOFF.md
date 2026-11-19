# HANDOFF — current state

**Last updated:** 2026-11-19 (end of Phase 2, before Kareem's approval)
**Active phase:** Phase 2 done → waiting at the STOP gate. Next: **Phase 3 (future.txt analysis)** after approval.
**Branch:** `main` (setup merged from `chore/phase2-repo-setup`). Baseline tag: `v1.0-semester1` = `390d7a0`.

## Where things stand
- Phase 1 audit: [00-project-understanding.md](00-project-understanding.md).
- Git history rewritten to remove data and weights (225 MB → 210 KB); backup bundle at `../mediscan-backup-2026-11-19/` ([ADR-0001](decisions/ADR-0001-remove-data-and-weights-from-git-history.md)).
- `.gitignore` active; `docs/` kept local ([ADR-0002](decisions/ADR-0002-keep-semester-docs-local.md), provisional).
- `CLAUDE.md`, logging system and hooks are in place ([ADR-0003](decisions/ADR-0003-git-workflow.md), [ADR-0004](decisions/ADR-0004-logging-system-and-hooks.md)).
- Semester-1 numbers recovered in [EXP-000](experiments/EXP-000-semester1-checkpoint-eval.md). Clean test: Dice 0.411; acc full 0.800, GT-ROI 0.740, predicted-ROI 0.460. The contaminated test inflated these to 0.909 / 0.873 / 0.709.

## Key facts every agent must know
- Dataset = Figshare / Cheng et al. (3,064 slices, 233 patients). The semester-1 reports cite the wrong (Kaggle) dataset.
- The semester-1 evaluation is invalid: 60/110 val and test files are training images, and the split is image-level, not patient-wise.
- The local PyTorch is **CPU-only** (`2.9.1+cpu`). The GPU is an RTX 3050 Laptop with **4 GB**.
- Rule: no code, model or training changes until Phase 6 is approved.

## Next steps
1. Kareem reviews Phase 2 (structure, hook test, commits) and approves.
2. Phase 3: analyse every idea in `future.txt` → `ideas/idea-matrix.md`. STOP.
3. Phase 4: SOTA research, including **alternative or extra datasets** (Kareem is open to a better one) and verifying the Figshare licence.

## Open questions / blockers
- ADR-0002: should `docs/` stay out of the public repo? (default: yes)
- Should the semester-1 checkpoints be attached to a GitHub Release for `v1.0-semester1`? (default: no, local only)
- Raw `.mat` files (with patient IDs): Kareem is re-downloading them; they are not yet in the folder.
- GitHub still reports the old repo size until its GC runs. No action needed unless we want GitHub Support to purge the cache.

## Key decisions
ADR-0001 history rewrite · ADR-0002 docs local (provisional) · ADR-0003 git workflow · ADR-0004 logging + hooks.
