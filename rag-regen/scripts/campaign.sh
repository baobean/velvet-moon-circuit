#!/usr/bin/env bash
# Run the doc-5 common-concept campaign end to end, resumably.
#
# Every stage records a marker under logs/campaign/. Re-running this script
# skips completed stages and picks up at the first one without a marker, so
# it is safe to run again after a crash, a reboot, or an agent session that
# ran out of quota mid-campaign. Nothing here is destructive on re-entry.
#
# WHY THE PIPELINE RUNS PIN THEIR RUN DIRECTORY:
# run_pipeline.py exits 2 on eviction (schedule.StageAborted) and prints
# "resume with: --resume <dir>". supervise.sh retries by re-running the
# IDENTICAL command -- so a command without --resume calls trace.open_run
# again, mints a fresh outputs/<tag>_<TS>/, and starts the queue from zero.
# Over a 15-hour run on a contended card that discards hours of work on
# every eviction and may never finish. This driver creates the run directory
# once, stores it under logs/campaign/, and always passes --resume, which
# makes the supervised command genuinely idempotent.
#
# WHY THERE IS A WATCHDOG:
# The first campaign attempt hung for 3.5 hours at 0% CPU. A worker died of
# glibc heap corruption ("free(): invalid next size") and img2dataset's
# multiprocessing pool deadlocked waiting on results that never arrived. A
# deadlock logs nothing, so nothing noticed. Any stage that stops producing
# output for --stall seconds is now killed and retried. Silence is not
# success.
#
# Usage:
#   ./scripts/campaign.sh            # run/resume the campaign
#   ./scripts/campaign.sh status     # print what is done, running, next
#   ./scripts/campaign.sh redo <stage>   # clear one stage's marker
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

STATE="logs/campaign"
LOG="$STATE/campaign.log"
DATASET="configs/dataset_common.yaml"
OWNER="$STATE/OWNER"
mkdir -p "$STATE"

# Which card to wait on and pin. CAMPAIGN_GPU=auto picks the one with the most
# total memory, which is what you want on a mixed box: the pipeline stages ask
# for 17 GB, so on a 2x16GB + 1x24GB machine only the 24 GB card can EVER
# satisfy them. Defaulting to index 0 there would wait forever on a card that
# physically cannot fit the job.
pick_gpu() {
  local want="${CAMPAIGN_GPU:-0}"
  if [ "$want" = "auto" ]; then
    want="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits \
            2>/dev/null | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')"
    [ -n "$want" ] || { echo "cannot read nvidia-smi to pick a GPU" >&2; exit 3; }
  fi
  printf '%s' "$want"
}

STAGES=(select download manifest make_cases validate build_index screen dry_run real_run report)

say() { printf '[campaign %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# logs/campaign/ and outputs/ live on the shared NAS, so a second machine can
# see this campaign's markers and run directories. Two drivers would both
# claim the same stage, both write index.faiss, and interleave into one
# campaign.log. There is no cross-host PID to check, so ownership is a
# heartbeat: run_watched touches OWNER every 60s, and a file younger than
# STALE_AFTER means someone is alive out there.
STALE_AFTER=300
claim_campaign() {
  local host="$(hostname)" now age ohost opid
  now=$(date +%s)
  if [ -s "$OWNER" ]; then
    read -r ohost opid _ < "$OWNER"
    age=$(( now - $(stat -c%Y "$OWNER" 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$STALE_AFTER" ]; then
      if [ "$ohost" != "$host" ]; then
        say "REFUSING TO START: campaign is live on '$ohost' (pid $opid), heartbeat ${age}s ago."
        say "  logs/campaign/ and outputs/ are shared -- two drivers would corrupt both."
        say "  Stop it there first, or rm $OWNER if you are certain it is dead."
        exit 9
      fi
      if kill -0 "$opid" 2>/dev/null; then
        say "REFUSING TO START: already running here as pid $opid."
        exit 9
      fi
    fi
    [ "$age" -ge "$STALE_AFTER" ] && \
      say "taking over from '$ohost' pid $opid (stale ${age}s)"
  fi
  printf '%s %s %s\n' "$host" "$$" "$(date -Is)" > "$OWNER"
}

pin_run_dir() {
  #: `f` is assigned on its OWN line on purpose. bash expands every word of a
  #: `local` command BEFORE binding any of them, so `local key="$1"
  #: f="$STATE/${key}_dir"` expands ${key} while key is still unset -- and
  #: under `set -u` that kills the command substitution and hands the caller
  #: an empty string. See tests/test_campaign_run_dir.py.
  local key="$1" tag="$2"
  local f="$STATE/${key}_dir"
  if [ ! -s "$f" ]; then
    local d="outputs/${tag}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$d"
    printf '%s\n' "$d" > "$f"
    ln -sfn "$(basename "$d")" "outputs/${tag}_latest"
    #: To stderr. `say` tees to stdout, and this function's stdout IS its
    #: return value -- the caller reads it through $(...). Left on stdout the
    #: banner is prepended to the path on the one call that mints the
    #: directory, so the first run of a fresh campaign gets a run dir named
    #: "[campaign ...] pinned dry run directory: outputs/...". tee still
    #: writes the line to $LOG, so nothing is lost from the transcript.
    say "pinned $key run directory: $d" >&2
  fi
  cat "$f"
}

# This driver runs under `set -uo pipefail` with NO `-e`, deliberately:
# run_stage inspects exit codes itself. The cost is that a failed command
# substitution assigns an empty string and execution simply continues. On
# 2026-08-09 that turned `--resume "$DRY_DIR"` into `--resume ""`, which
# argparse resolves to the repo root -- both arms shared one directory, the
# inpaint arm found every case already terminal and produced nothing in 8
# seconds, and it overwrote the stitch run's run.json with its own mechanism.
# An empty run directory must stop the campaign at the first stage instead.
require_run_dir() {
  local what="$1" dir="$2"
  if [ -z "$dir" ]; then
    say "ABORT: $what run directory is empty -- pin_run_dir failed."
    say "  Running a stage now would target the repo root and silently"
    say "  produce a run that looks complete and means nothing."
    exit 3
  fi
  #: Not just non-empty: pin_run_dir returns its value on stdout, so ANY
  #: stray write to stdout inside it is appended to the path. That is not
  #: hypothetical either -- `say` tees to stdout, and the banner it prints
  #: when it mints the directory landed in the path until it was sent to
  #: stderr. A multi-line or non-outputs/ value means something is writing
  #: where the return value lives.
  case "$dir" in
    outputs/*) ;;
    *) say "ABORT: $what run directory is not under outputs/: '$dir'"
       say "  Something wrote to pin_run_dir's stdout; that is its return value."
       exit 3 ;;
  esac
  if [ "$(printf '%s' "$dir" | wc -l)" -ne 0 ] || [ ! -d "$dir" ]; then
    say "ABORT: $what run directory is not a single existing path: '$dir'"
    exit 3
  fi
}

# Run a command, killing it only if it makes no progress for $stall seconds.
# Returns the command's exit code, or 124 if the watchdog fired.
#
# Liveness is (log grew) OR (the stage's own progress probe changed). Log
# growth alone is NOT enough: img2dataset downloads for many minutes between
# tqdm writes, and an earlier version of this watchdog killed a perfectly
# healthy download that had simply gone quiet. A probe that measures the
# thing the stage actually produces is the only honest signal.
run_watched() {
  local name="$1" stall="$2" probe="$3"; shift 3
  "$@" >>"$LOG" 2>&1 &
  local pid=$! last_sig="" sig last_change now
  last_change=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    #: Ownership heartbeat -- see claim_campaign. Refreshed here
    #: because this is the one loop that runs for a stage's whole life.
    touch "$OWNER" 2>/dev/null || true
    sig="$(stat -c%s "$LOG" 2>/dev/null || echo 0)"
    [ -n "$probe" ] && sig="$sig:$(eval "$probe" 2>/dev/null || echo probe_error)"
    now=$(date +%s)
    if [ "$sig" != "$last_sig" ]; then last_sig="$sig"; last_change="$now"; fi
    if [ $(( now - last_change )) -ge "$stall" ]; then
      say "STALL: $name made no progress for ${stall}s (pid $pid) -- killing it"
      say "  last progress signature: $last_sig"
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

# tries > 1 is only safe for stages that resume rather than restart.
run_stage() {
  local name="$1" tries="$2" stall="$3" probe="$4"; shift 4
  if [ -f "$STATE/$name.done" ]; then
    say "SKIP $name (completed $(cat "$STATE/$name.done"))"
    return 0
  fi
  rm -f "$STATE/$name.failed"
  date '+%F %T' > "$STATE/$name.running"
  local rc=0 i
  for i in $(seq 1 "$tries"); do
    say "START $name (attempt $i/$tries, stall guard ${stall}s)"
    say "  \$ $*"
    run_watched "$name" "$stall" "$probe" "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      rm -f "$STATE/$name.running"
      date '+%F %T' > "$STATE/$name.done"
      say "DONE $name"
      return 0
    fi
    say "attempt $i/$tries of $name exited $rc"
    [ "$i" -lt "$tries" ] && say "retrying $name -- it resumes rather than restarts"
  done
  rm -f "$STATE/$name.running"
  printf '%s\n' "$rc" > "$STATE/$name.failed"
  say "FAIL $name (exit $rc after $tries attempt(s))"
  say "The campaign stops here. Investigate, then re-run ./scripts/campaign.sh"
  exit "$rc"
}

cmd_status() {
  printf 'Campaign state (%s)\n\n' "$ROOT/$STATE"
  local next=""
  for s in "${STAGES[@]}"; do
    if [ -f "$STATE/$s.done" ]; then
      printf '  [done]     %-12s %s\n' "$s" "$(cat "$STATE/$s.done")"
    elif [ -f "$STATE/$s.running" ]; then
      printf '  [RUNNING]  %-12s since %s\n' "$s" "$(cat "$STATE/$s.running")"
      [ -z "$next" ] && next="$s (in flight)"
    elif [ -f "$STATE/$s.failed" ]; then
      printf '  [FAILED]   %-12s exit %s\n' "$s" "$(cat "$STATE/$s.failed")"
      [ -z "$next" ] && next="$s"
    else
      printf '  [pending]  %-12s\n' "$s"
      [ -z "$next" ] && next="$s"
    fi
  done
  printf '\n'
  #: Exclude this very invocation and its parent, or `status` reports the
  #: driver as running purely because *it* matched the pattern.
  if pgrep -f "scripts/campaign.sh" 2>/dev/null \
       | grep -vx -e "$$" -e "$PPID" | grep -q .; then
    printf 'driver: RUNNING (pid %s)\n' \
      "$(pgrep -f 'scripts/campaign.sh' | grep -vx -e "$$" -e "$PPID" | head -1)"
  else
    printf 'driver: not running\n'
  fi
  [ -s "$STATE/dry_dir" ]  && printf 'dry-run dir:  %s\n' "$(cat "$STATE/dry_dir")"
  [ -s "$STATE/real_dir" ] && printf 'real-run dir: %s\n' "$(cat "$STATE/real_dir")"
  if [ -n "$next" ]; then
    printf '\nNext: %s   -- resume with ./scripts/campaign.sh\n' "$next"
  else
    printf '\nAll stages complete.\n'
  fi
}

case "${1:-run}" in
  status) cmd_status; exit 0 ;;
  redo)
    [ $# -ge 2 ] || { echo "usage: $0 redo <stage>" >&2; exit 64; }
    rm -f "$STATE/$2.done" "$STATE/$2.failed" "$STATE/$2.running"
    echo "cleared markers for stage: $2"
    exit 0 ;;
  run) ;;
  *) echo "unknown command: $1" >&2; exit 64 ;;
esac

claim_campaign
GPU="$(pick_gpu)"

say "=== campaign start (pid $$ on $(hostname), GPU $GPU) ==="
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.free,memory.total \
  --format=csv,noheader 2>&1 | sed 's/^/[campaign] gpu: /' | tee -a "$LOG"

# Progress probes: what each stage actually produces. Cheap enough to run
# once a minute, and truthful when the stage is working silently.
P_IMAGES="find data/laion100k/images -name '*.jpg' | wc -l"
P_REFS="find data/gt_refs -name '*.jpg' 2>/dev/null | wc -l"

# ---- CPU / network stages -------------------------------------------------
run_stage select     1 900  ""          ./scripts/run.sh fetch-corpus select

# Concurrency is deliberately below img2dataset's default. The first attempt
# ran 8 processes x 32 threads and a worker died of heap corruption, taking
# the pool with it. Fewer, calmer workers; incremental_mode means each retry
# skips the shards already on disk, so progress is monotonic across attempts.
run_stage download   6 1800 "$P_IMAGES" ./scripts/run.sh fetch-corpus download \
  --processes 4 --threads 16

run_stage manifest   1 1800 ""          ./scripts/run.sh fetch-corpus manifest
run_stage make_cases 2 1800 "$P_REFS"   ./scripts/run.sh make-cases
run_stage validate   1 600  ""          ./scripts/run.sh validate --dataset "$DATASET"

# ---- GPU stages -----------------------------------------------------------
# supervise blocks on gpu_wait until the card is genuinely idle, so a long
# quiet period here is expected -- hence the generous stall guards.
# --need 8, not 6: so400m loads fp32, so the weights alone are ~4.0 GB and the
# measured peak at batch 32 is ~5.8 GB (2026-08-06 OOM: 4.01 allocated + 0.47
# reserved + context, still short 384 MB). A 6 GB gate has no margin -- it let
# the stage launch beside a neighbour that then grew, and CUDA OOM exits 1,
# which supervise correctly refuses to retry. Costs no wall-clock: screen
# needs 13 GB immediately after.
run_stage build_index 3 5400 "" ./scripts/supervise.sh --gpu "$GPU" --need 8 --log "$STATE/build_index.log" \
  -- ./scripts/run.sh build-index

run_stage screen      3 5400 "" ./scripts/supervise.sh --gpu "$GPU" --need 13 --log "$STATE/screen.log" \
  -- ./scripts/run.sh screen --dataset "$DATASET"

DRY_DIR="$(pin_run_dir dry doc5dry)"
require_run_dir dry "$DRY_DIR"
#: The queue file is rewritten as each case resolves, so its mtime is the
#: truthful liveness signal for a pipeline that logs one line per case.
P_DRY="stat -c%Y '$DRY_DIR/queue.json' 2>/dev/null || echo 0"
run_stage dry_run     3 7200 "$P_DRY" ./scripts/supervise.sh --gpu "$GPU" --need 17 --log "$STATE/dry_run.log" \
  -- ./scripts/run.sh pipeline --dataset "$DATASET" \
     --arm full --mechanism stitch --resume "$DRY_DIR"

REAL_DIR="$(pin_run_dir real doc5)"
require_run_dir real "$REAL_DIR"
P_REAL="stat -c%Y '$REAL_DIR/queue.json' 2>/dev/null || echo 0"
run_stage real_run    3 7200 "$P_REAL" ./scripts/supervise.sh --gpu "$GPU" --need 17 --log "$STATE/real_run.log" \
  -- ./scripts/run.sh pipeline --dataset "$DATASET" \
     --arm full --mechanism inpaint --resume "$REAL_DIR"

run_stage report      1 1800 "" ./scripts/run.sh report --run "$REAL_DIR" --dataset "$DATASET"

say "=== campaign complete ==="
say "dry-run:  $DRY_DIR"
say "real-run: $REAL_DIR"
say "Read the dry-run's report before trusting the real one."
