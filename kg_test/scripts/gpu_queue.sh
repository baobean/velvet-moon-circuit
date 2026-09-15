#!/usr/bin/env bash
# GPU job queue: wait for sufficient free GPU memory on this shared card,
# then run a command. If the command fails with a CUDA OOM signature (the
# real, observed failure mode here -- another researcher's job can grow its
# footprint mid-run), wait for headroom again and retry, up to a bounded
# number of attempts. Any other failure is not retried.
#
# Usage:
#   gpu_queue.sh <min_free_mib> <log_file> [<max_attempts>] -- <command...>
#
# Appends everything (queue status + command stdout/stderr) to <log_file>,
# so the caller only has to tail one file. Exits with the command's exit
# code on success, or 1 if attempts are exhausted / min_free_mib is never
# reached within POLL_TIMEOUT_S.

set -uo pipefail

MIN_FREE_MIB="$1"; shift
LOG_FILE="$1"; shift
MAX_ATTEMPTS=3
if [[ "$1" =~ ^[0-9]+$ ]]; then
  MAX_ATTEMPTS="$1"; shift
fi
if [ "$1" != "--" ]; then
  echo "usage: gpu_queue.sh <min_free_mib> <log_file> [<max_attempts>] -- <command...>" >&2
  exit 2
fi
shift

POLL_INTERVAL_S=30
STATUS_EVERY_S=1800  # log a heartbeat every 30min so a long wait isn't silent

log() { echo "[gpu_queue] $(date -Iseconds) $*" | tee -a "$LOG_FILE"; }

wait_for_headroom() {
  # Waits indefinitely -- shared-GPU contention on this card has been
  # observed to last multiple hours (a training job's memory footprint
  # doesn't shrink until it finishes), and the entire point of this queue
  # is to wait as long as it takes rather than need a human/agent to keep
  # relaunching it. A bounded timeout defeats that purpose (confirmed: a 2h
  # cap gave up while the competing job was still running).
  local waited=0
  local last_logged=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ]; then
      log "free=${free}MiB >= ${MIN_FREE_MIB}MiB, proceeding"
      return 0
    fi
    if [ $((waited - last_logged)) -ge "$STATUS_EVERY_S" ]; then
      log "still waiting: free=${free:-unknown}MiB < ${MIN_FREE_MIB}MiB (waited ${waited}s so far)"
      last_logged=$waited
    fi
    sleep "$POLL_INTERVAL_S"
    waited=$((waited + POLL_INTERVAL_S))
  done
}

# Scans only the bytes THIS attempt appended to LOG_FILE (from
# $1=byte-offset-before-launch to current EOF), not a fixed-size tail of the
# whole (shared, ever-growing across attempts and driver steps) log file --
# a fixed tail was confirmed to miss a real OOM signature once it scrolled
# past pytest's trailing warnings/summary output.
is_oom_failure() {
  local offset="$1"
  tail -c "+$((offset + 1))" "$LOG_FILE" | grep -qiE "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory"
}

log "queued: $* (min_free=${MIN_FREE_MIB}MiB, max_attempts=${MAX_ATTEMPTS})"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  log "attempt ${attempt}/${MAX_ATTEMPTS}: waiting for headroom"
  if ! wait_for_headroom; then
    exit 1
  fi
  log "attempt ${attempt}/${MAX_ATTEMPTS}: launching: $*"
  pre_offset=$(wc -c < "$LOG_FILE")
  "$@" >> "$LOG_FILE" 2>&1
  code=$?
  log "attempt ${attempt}/${MAX_ATTEMPTS}: command exited with code ${code}"
  if [ "$code" -eq 0 ]; then
    exit 0
  fi
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ] && is_oom_failure "$pre_offset"; then
    log "OOM signature detected, will retry after headroom frees up"
    attempt=$((attempt + 1))
    continue
  fi
  log "non-OOM failure or attempts exhausted, giving up"
  exit "$code"
done
