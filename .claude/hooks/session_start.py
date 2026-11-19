"""SessionStart hook: print project-log/HANDOFF.md into the new session's context.

Also records the session's starting git HEAD and time in .claude/state/ so the
Stop hook can detect files committed during this session.
Never blocks: always exits 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "project-log" / "HANDOFF.md"
STATE_DIR = ROOT / ".claude" / "state"


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    session_id = str(payload.get("session_id") or "unknown")
    source = payload.get("source", "")

    # Record session start (only on a fresh start/clear, keep the original on resume/compact).
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file = STATE_DIR / f"session-{session_id}.json"
        if not state_file.exists() or source in ("startup", "clear"):
            state_file.write_text(
                json.dumps({"session_id": session_id, "start_head": git_head(), "start_time": time.time()}),
                encoding="utf-8",
            )
    except Exception:
        pass

    sys.stdout.reconfigure(encoding="utf-8")
    if HANDOFF.exists():
        print("=== project-log/HANDOFF.md (current project state; read before acting) ===")
        print(HANDOFF.read_text(encoding="utf-8"))
        print("=== end HANDOFF.md === Rule: update HANDOFF.md and CHANGELOG.md after every change to code or plans.")
    else:
        print("WARNING: project-log/HANDOFF.md not found. Create it before doing any work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
