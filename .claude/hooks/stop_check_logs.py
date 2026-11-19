"""Stop hook: block stopping if project files changed but the logs were not updated.

Rule: when any file OUTSIDE project-log/ has changed (uncommitted in the working
tree, or committed during this session), both project-log/HANDOFF.md and
project-log/CHANGELOG.md must have been modified at or after the most recent of
those changes. Otherwise exit 2 with a message on stderr, which makes Claude
continue and update the logs.

Loop guard: if stdin JSON has stop_hook_active == true (Claude is already
continuing because of this hook), exit 0.
Fails open (exit 0) on any internal error so a broken hook never traps a session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = "project-log/"
REQUIRED_LOGS = ["project-log/HANDOFF.md", "project-log/CHANGELOG.md"]
EXEMPT_PREFIXES = (LOG_DIR, ".claude/state/")
TOLERANCE_S = 1.0


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
    return out.stdout


def working_tree_changes() -> set[str]:
    paths: set[str] = set()
    for line in git("status", "--porcelain", "-uall").splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:  # rename: take the new path
            path = path.split(" -> ", 1)[1]
        paths.add(path.strip().strip('"'))
    return paths


def session_committed_changes(session_id: str) -> tuple[set[str], float | None]:
    state = ROOT / ".claude" / "state" / f"session-{session_id}.json"
    if not state.exists():
        return set(), None
    data = json.loads(state.read_text(encoding="utf-8"))
    start_head, start_time = data.get("start_head"), data.get("start_time")
    if not start_head:
        return set(), start_time
    head = git("rev-parse", "HEAD").strip()
    if head == start_head:
        return set(), start_time
    names = git("diff", "--name-only", start_head, head).splitlines()
    return {n.strip() for n in names if n.strip()}, start_time


def mtime(rel: str) -> float | None:
    p = ROOT / rel
    return p.stat().st_mtime if p.exists() else None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    if payload.get("stop_hook_active"):
        return 0

    try:
        changed = working_tree_changes()
        committed, start_time = session_committed_changes(str(payload.get("session_id") or "unknown"))
        changed |= committed
        relevant = sorted(p for p in changed if not p.startswith(EXEMPT_PREFIXES))
        if not relevant:
            return 0

        times = [t for t in (mtime(p) for p in relevant) if t is not None]
        if not times and start_time is not None:
            times = [start_time]  # only deletions: compare against session start
        if not times:
            return 0
        latest_change = max(times)

        stale = [log for log in REQUIRED_LOGS if (mtime(log) or 0) + TOLERANCE_S < latest_change]
        if not stale:
            return 0

        shown = "\n".join(f"  - {p}" for p in relevant[:15])
        more = f"\n  ... and {len(relevant) - 15} more" if len(relevant) > 15 else ""
        sys.stderr.write(
            "LOG UPDATE REQUIRED before stopping.\n"
            f"Files outside project-log/ changed:\n{shown}{more}\n"
            f"Not updated since the latest change: {', '.join(stale)}\n"
            "Update project-log/HANDOFF.md (state, next steps, open questions, timestamp) and add a dated "
            "entry to project-log/CHANGELOG.md, then stop again.\n"
        )
        return 2
    except Exception as exc:  # fail open
        sys.stderr.write(f"stop_check_logs hook error (ignored): {exc}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
