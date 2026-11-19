# ADR-0004 — Project logging system and Claude Code hooks

- **Date:** 2026-11-19
- **Status:** Accepted (specified by Kareem in the project brief)

## Context
Work spans many sessions and agents. The logs are the project's memory and the raw material for the thesis and the presentation, so they must not drift from the code.

## Decision
- `project-log/` layout: HANDOFF, CHANGELOG, sessions/, decisions/ (ADRs), research/, experiments/, ideas/, thesis-map, presentation-bank, university-requirements.
- Hooks (project-level, committed in `.claude/settings.json`, scripts in `.claude/hooks/`, Python, so they run on Windows without jq):
  - **SessionStart** → `session_start.py`: prints `HANDOFF.md` into context. It also records the session's start HEAD and time in `.claude/state/` (git-ignored).
  - **Stop** → `stop_check_logs.py`: when files outside `project-log/` changed (uncommitted, or committed since the session started), both `HANDOFF.md` and `CHANGELOG.md` must have an mtime at or after the newest change. If not, it exits 2 with the reason on stderr. It returns 0 immediately when `stop_hook_active` is true (loop guard), and it fails open on internal errors.

## Options considered for the Stop check
| Option | Verdict |
|---|---|
| "logs appear in `git status`" | Fails once logs are committed; ignores order |
| **mtime: logs newer than the newest non-log change** | Chosen: works for uncommitted and committed changes, and catches "code edited after logs" |
| content hash snapshots per session | Heavier; no benefit for this project |

## Consequences
- A branch switch that changes files counts as a change, so the logs need a touch afterwards. That's acceptable, since a branch switch is a state change worth logging.
- Hook config changes take effect only in a new session (or after reviewing them in `/hooks`).
