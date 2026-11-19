# ADR-0001 — Remove data and model weights from the git history

- **Date:** 2026-11-19
- **Status:** Accepted (approved by Kareem, Phase 1 answer to Q2: "go with c, keep our work clean")

## Context
The ignore file was named `.gitignore.txt`, so git never applied it. The single semester-1 push to the **public** repo therefore included 8,622 data/artifact files: the converted Figshare dataset (`dataset/`, 6,129 files), the pilot images and ROIs (`data/pilot/{processed,roi_gt,roi_pred}/`), 4 checkpoints (`outputs/`, ~132 MB) and `__pycache__`. The repo was 225 MB, and the README claimed the dataset was not included. The dataset licence terms are still unverified.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| a. Leave as is | No risk | Public redistribution of the dataset; 225 MB clone; contradicts README |
| b. Stop tracking from now on | No history rewrite | Files stay in history forever; clone stays 225 MB |
| **c. Rewrite history** | Clean, 210 KB repo; no data published | Force-push; commit hashes change; GitHub may cache old objects for a while |

## Decision
Option c, done as follows:
1. Fast-forwarded local `main` to `origin/main` (`3337b1a`).
2. Backed up the full original history (all refs, including a tag on `3337b1a` that has the data) to `../mediscan-backup-2026-11-19/mediscan-original-history.bundle` (233,817,967 bytes, `git bundle verify` OK). This folder is outside the repo.
3. Ran `git filter-repo --invert-paths` in a scratch clone for: `dataset/`, `data/pilot/processed/`, `data/pilot/roi_gt/`, `data/pilot/roi_pred/`, `outputs/`, `*__pycache__*`. **Kept:** code, README, `future.txt`, `images/`, `data/pilot/splits/*.txt`.
4. Verified that `git diff 3337b1a <rewritten>` contains only deletions (8,622) and that all of them are in the paths above.
5. Ran `git reset --mixed` in the working folder, so all files on disk stayed untouched.
6. `git push --force-with-lease=main:3337b1a` → `3337b1a...390d7a0`. Annotated tag `v1.0-semester1` created on `390d7a0` and pushed.

Commit mapping (old → new): `2167d62→ef9c188`, `dbe1761→08ed11c`, `323890b→3fb677d`, `d9baf30→3596133`, `ea75e9e→51eb8b1`, `3337b1a→390d7a0`.

## Consequences
- The repo now holds no data. Datasets live outside git; download instructions will be documented (Phase 6).
- Semester-1 checkpoints stay on local disk only (`outputs/`, ignored) and in the backup bundle. Optional later step: attach them to a GitHub Release (not done; needs approval).
- Anyone with an old clone must re-clone. There were 0 forks at rewrite time.
- GitHub may serve old objects by SHA until its garbage collection runs. Contacting GitHub Support can speed this up if needed.
- The local `.git` still contains the old objects (reflog) until `git gc` runs. That's harmless, and the backup exists anyway.
