#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_ROOT="$WORKSPACE/benchmark_results/drs_29_scene_phase1_k30_i90"
PHASE0_ROOT="$WORKSPACE/benchmark_results/drs_29_scene_phase0_k30_i90"
RUN_LOG="$WORKSPACE/benchmark_results/drs_29_scene_phase1_k30_i90_runner.log"
MONITOR_LOG="$OUTPUT_ROOT/hourly_monitor.log"
LOCK_FILE="$OUTPUT_ROOT/hourly_monitor_v3.lock"
SNAPSHOT_FILE="$OUTPUT_ROOT/hourly_snapshot.txt"
PID_FILE="$OUTPUT_ROOT/runner.pid"

mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

timestamp=$(date --iso-8601=seconds)
echo "===== $timestamp =====" >> "$MONITOR_LOG"
if timeout 120 "$SCRIPT_DIR/.venv/bin/python" \
    "$SCRIPT_DIR/analyze_drs_29_scene_phase1.py" \
    "$PHASE0_ROOT" "$OUTPUT_ROOT" \
    --report "$OUTPUT_ROOT/progress.md" \
    > "$SNAPSHOT_FILE" 2>&1; then
  grep -E 'completed:|baseline completed|^\| [^|]+ \|' \
    "$SNAPSHOT_FILE" >> "$MONITOR_LOG" || true
else
  echo "analyzer=failed-or-timed-out" >> "$MONITOR_LOG"
  tail -20 "$SNAPSHOT_FILE" >> "$MONITOR_LOG" 2>/dev/null || true
fi

completed_variants=$(grep -Ec '^\| [^|]+ \| 29/29 \|' "$SNAPSHOT_FILE" || true)
if [[ "$completed_variants" -eq 14 ]]; then
  echo "runner=complete variants=14/14" >> "$MONITOR_LOG"
  rm -f "$PID_FILE"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  runner_pid=$(<"$PID_FILE")
  if [[ "$runner_pid" =~ ^[0-9]+$ && -r "/proc/$runner_pid/cmdline" ]]; then
    runner_command=$(tr '\0' ' ' < "/proc/$runner_pid/cmdline")
    if [[ "$runner_command" == *"run_drs_29_scene_phase1.sh"* ]]; then
      echo "runner=active pid=$runner_pid" >> "$MONITOR_LOG"
      exit 0
    fi
  fi
fi

echo "runner=inactive; resuming" >> "$MONITOR_LOG"
nohup env \
  BUNDLE_PALM_REQUEST_PORT=18756 \
  BUNDLE_PALM_RESULT_PORT=18757 \
  OVERWRITE=0 \
  LIVE_OUTPUT=0 \
  DEBUG_OUTPUT=0 \
  "$SCRIPT_DIR/run_drs_29_scene_phase1.sh" \
  >> "$RUN_LOG" 2>&1 < /dev/null 9>&- &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$PID_FILE"
echo "runner_pid=$runner_pid" >> "$MONITOR_LOG"