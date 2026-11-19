# ADR-0003 — Git workflow for semester 2

- **Date:** 2026-11-19
- **Status:** Accepted (specified by Kareem in the project brief)

## Decision
- `main` stays stable. Work happens on `feature/<name>`, `research/<name>`, `exp/<name>` or `chore/<name>` branches and is merged into `main` by fast-forward or merge commit, never by rebase of published history.
- Use **Conventional Commits**: `feat`, `fix`, `docs`, `exp`, `refactor`, `chore`, `test`. Keep commits small and logical, with an imperative subject of 72 characters or fewer and a body that explains *why*.
- **No AI co-author or attribution lines** in commits or PRs (Kareem's instruction).
- Before every push, show what will be pushed (`git log origin/<branch>..HEAD --stat`). Never force-push, rewrite history, delete branches or delete files without explicit approval. The one approved exception is ADR-0001.
- Tags: `v1.0-semester1` = semester-1 baseline. Future milestones get `v2.x-<milestone>`.

## Consequences
History is easy to map onto the thesis (commit hashes are referenced in CHANGELOG and EXP logs), and `main` is always demo-able.
