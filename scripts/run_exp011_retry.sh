#!/usr/bin/env bash
# EXP-011 unattended runner: clean -> external -> shift, each resumable (finished units are skipped).
# Retries up to 6 times with a 90 s pause after a failure (e.g. a GPU driver reset, see HANDOFF key facts);
# stops at once if a pre-specified check fails (retrying cannot fix that).
# exit status is taken from python itself (PIPESTATUS), not from the log filter
PY=.venv/Scripts/python.exe
LOG=runs/exp011.log
for mode in clean external shift; do
  for attempt in 1 2 3 4 5 6; do
    echo "=== $mode attempt $attempt $(date -Iseconds)" >> "$LOG"
    $PY scripts/run_exp011.py --mode "$mode" 2>&1 | grep --line-buffered -v -i -E "warn|redirects|Loading weights" >> "$LOG"
    status=${PIPESTATUS[0]}
    if [ $status -eq 0 ]; then echo "=== $mode done $(date -Iseconds)" >> "$LOG"; break; fi
    if grep -q "pre-specified check" "$LOG"; then echo "=== STOP: pre-specified check failed" >> "$LOG"; exit 2; fi
    echo "=== $mode failed (status $status); retry in 90 s" >> "$LOG"
    sleep 90
    if [ $attempt -eq 6 ]; then echo "=== GIVE UP on $mode" >> "$LOG"; exit 1; fi
  done
done
echo "=== ALL DONE $(date -Iseconds)" >> "$LOG"
