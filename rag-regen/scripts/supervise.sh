#!/usr/bin/env bash
# Keep a long pipeline run alive across GPU evictions.
#
# gpu_wait.sh guards only the initial launch. Over a 15-hour run the dominant
# failure is a neighbour landing at 3am: the pipeline checkpoints correctly and
# exits 2, then sits idle until a human notices. This turns "the run died" into
# "the run paused" (doc 5 §9).
#
# Exit 2 is schedule.StageAborted -- queue.json is saved and resuming is
# correct. Any other non-zero code is a real failure and is NOT retried, or a
# deterministic crash would loop until the deadline.
#
# Usage:
#   ./scripts/supervise.sh [options] -- <command> [args...]
#
#   --max-retries N   give up after N evictions       (default 40)
#   --interval S      seconds between attempts        (default 120)
#   --deadline S      total wall-clock budget, 0=none (default 0)
#   --need GB         passed to gpu_wait.sh           (default 17)
#   --gpu N           GPU index to wait on and pin    (default 0)
#   --no-gpu-wait     run the command directly (tests)
#   --log FILE        tee output here
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_RETRIES=40
INTERVAL=120
DEADLINE=0
NEED_GB=17
GPU=0
USE_GPU_WAIT=1
LOG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --interval)    INTERVAL="$2"; shift 2 ;;
    --deadline)    DEADLINE="$2"; shift 2 ;;
    --need)        NEED_GB="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --no-gpu-wait) USE_GPU_WAIT=0; shift ;;
    --log)         LOG="$2"; shift 2 ;;
    --)            shift; break ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    # 64 is EX_USAGE. Deliberately not 2 -- this script defines 2 as
    # "the child was evicted, retry", and a mistyped flag must never be
    # mistaken for an eviction by anything wrapping us.
    *)             echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

[ $# -ge 1 ] || { echo "error: no command after --" >&2; exit 64; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-logs/supervise_${TS}.log}"
mkdir -p "$(dirname "$LOG")"

say() { printf '[supervise %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Preflight: gpu_wait.sh is a hard dependency of the production path. Fail
# honestly, before the retry loop, rather than let a missing-file 127 surface
# from inside attempt 1 with no explanation.
if [ "$USE_GPU_WAIT" -eq 1 ]; then
  if [ ! -f "$ROOT/scripts/gpu_wait.sh" ] || [ ! -x "$ROOT/scripts/gpu_wait.sh" ]; then
    say "missing dependency: scripts/gpu_wait.sh not found or not executable -- create it (see docs) or pass --no-gpu-wait to skip GPU acquisition"
    exit 1
  fi
fi

started=$(date +%s)
attempt=0

while :; do
  attempt=$((attempt + 1))
  say "attempt ${attempt}/${MAX_RETRIES}: $*"

  if [ "$USE_GPU_WAIT" -eq 1 ]; then
    ./scripts/gpu_wait.sh --need "$NEED_GB" --gpu "$GPU" --log "$LOG" -- "$@"
  else
    "$@"
  fi
  rc=$?

  if [ "$rc" -eq 0 ]; then
    say "command finished cleanly after ${attempt} attempt(s)"
    exit 0
  fi

  # Anything other than StageAborted is a bug, not an eviction. Retrying a
  # deterministic crash would burn the whole booked block.
  if [ "$rc" -ne 2 ]; then
    say "command exited ${rc} -- not an eviction, not retrying"
    exit "$rc"
  fi

  say "evicted (exit 2); queue.json is checkpointed"

  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    say "giving up: --max-retries ${MAX_RETRIES} reached"
    exit 3
  fi

  if [ "$DEADLINE" -gt 0 ]; then
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -ge "$DEADLINE" ]; then
      say "giving up: deadline ${DEADLINE}s reached after ${elapsed}s"
      exit 4
    fi
  fi

  [ "$INTERVAL" -gt 0 ] && sleep "$INTERVAL"
done
