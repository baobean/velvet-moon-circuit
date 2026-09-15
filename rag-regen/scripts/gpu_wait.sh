#!/usr/bin/env bash
# Block until the shared 4090 has enough free VRAM, then exec a command.
#
# Not a pipeline stage, so deliberately not a `run.sh` verb -- this wraps
# run.sh from the outside. See RUNBOOK §2.3: one card, shared, and the models
# do not co-fit (Qwen-7B ~16 GB, FLUX-nf4 ~12 GB with a hard check_vram floor).
#
# Usage:
#   ./scripts/gpu_wait.sh [options] -- <command> [args...]
#
#   --need GB       free VRAM required to launch      (default 17)
#   --stable N      consecutive passing polls needed  (default 3)
#   --interval S    seconds between polls             (default 60)
#   --timeout S     give up after S seconds, 0 = wait forever (default 0)
#   --gpu N         GPU index to watch and pin        (default 0)
#   --log FILE      tee the command's output here     (default logs/gpu_wait_<TS>.log)
#
# Example -- the inpaint smoke test, then nothing else:
#   ./scripts/gpu_wait.sh -- ./scripts/run.sh repair --case axolotl \
#       --mechanism inpaint --draft outputs/screen_latest/axolotl/draft.png
#
# --stable exists because a neighbour's job dips between iterations. One poll
# at 18 GB free can be the gap between two of their steps, and we would load
# 16 GB of Qwen straight into their next allocation. Three passing polls a
# minute apart means the card is actually idle, not merely exhaling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NEED_GB=17
STABLE=3
INTERVAL=60
TIMEOUT=0
GPU=0
LOG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --need)     NEED_GB="$2"; shift 2 ;;
    --stable)   STABLE="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --log)      LOG="$2"; shift 2 ;;
    --)         shift; break ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ $# -ge 1 ] || { echo "error: no command after --" >&2; exit 2; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-logs/gpu_wait_${TS}.log}"
mkdir -p "$(dirname "$LOG")"

# MiB, because that is what nvidia-smi speaks.
NEED_MIB=$(awk -v g="$NEED_GB" 'BEGIN { printf "%d", g * 1024 }')

say() { printf '[gpu-wait %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

free_mib() {
  nvidia-smi --id="$GPU" --query-gpu=memory.free \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r'
}

say "watching GPU $GPU for ${NEED_GB} GB free, ${STABLE}x${INTERVAL}s apart"
say "will then run: $*"
say "log: $LOG"

probe="$(free_mib)"
if ! [[ "$probe" =~ ^[0-9]+$ ]]; then
  say "cannot read nvidia-smi on GPU $GPU -- refusing to guess"
  exit 3
fi

started=$(date +%s)
hits=0

while :; do
  avail="$(free_mib)"
  if ! [[ "$avail" =~ ^[0-9]+$ ]]; then
    # A transient nvidia-smi failure is not evidence the card is free.
    say "nvidia-smi unreadable; resetting streak"
    hits=0
  elif [ "$avail" -ge "$NEED_MIB" ]; then
    hits=$((hits + 1))
    say "${avail} MiB free (need ${NEED_MIB}) -- ${hits}/${STABLE}"
    [ "$hits" -ge "$STABLE" ] && break
  else
    # Speak on EVERY poll, not just when a streak breaks. campaign.sh's
    # watchdog reads "the log stopped growing" as a hung stage and kills it,
    # so a gate that waits silently through a long neighbour job gets killed
    # for behaving correctly -- which is what happened to build_index on
    # 2026-08-07 against a job that held the card for nine hours.
    if [ "$hits" -gt 0 ]; then
      say "${avail} MiB free (need ${NEED_MIB}) -- streak broken, restarting"
    else
      say "${avail} MiB free (need ${NEED_MIB}) -- waiting"
    fi
    hits=0
  fi

  if [ "$TIMEOUT" -gt 0 ]; then
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
      say "timed out after ${elapsed}s without ${NEED_GB} GB free"
      exit 4
    fi
  fi

  sleep "$INTERVAL"
done

waited=$(( $(date +%s) - started ))
say "card is free after ${waited}s -- launching"
nvidia-smi --id="$GPU" | sed 's/^/[gpu-wait] /'

# nvidia-smi numbers GPUs by PCI bus order; CUDA's own default enumeration
# (FASTEST_FIRST) can rank them differently on a mixed card box -- e.g. an
# A5000 outranks an A4000 by CUDA's heuristic even if nvidia-smi calls the
# A4000 "GPU 0". Without forcing PCI_BUS_ID order, --gpu N picks CUDA's Nth
# device, not nvidia-smi's GPU N, and a job silently lands on the wrong
# physical card wearing the right index.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

# PIPESTATUS, not $?, or tee's success masks the command's failure.
set +e
{
  echo "=== $(date -Is) :: $* ==="
  "$@" 2>&1
} | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

say "command exited $rc (waited ${waited}s, log $LOG)"
exit "$rc"
