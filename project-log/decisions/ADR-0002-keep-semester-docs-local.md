# ADR-0002 — Keep `docs/` (semester reports and slides) out of the public repo for now

- **Date:** 2026-11-19
- **Status:** Accepted (Kareem confirmed on 2026-11-19: keep `docs/` local)

## Context
`docs/` (untracked) holds the semester-1 thesis versions, interim reports, slides and an unrelated course template. These documents contain **both students' full names and student ID numbers**. The repo is public. Kareem said that GitHub presents the project as a solo project and that no team sections should be added.

## Options considered
1. **Ignore `docs/` (keep local only)** — no personal data published; the logs reference the docs by filename.
2. Commit `docs/` — the complete record lives in git, but student IDs and the second author become public.
3. Commit redacted PDFs only — a middle ground that takes extra work.

## Decision
Option 1 for now: `/docs/` is in `.gitignore`. Kareem can reverse this at any time.

## Consequences
- The documents exist only on this machine. Kareem should keep his own backup (they were not part of the git bundle, because they were never committed).
- The thesis map and logs cite them by filename (e.g. `docs/final report1.pdf`).
