#!/usr/bin/env bash
# Chains the two remaining GPU-bound Phase 1 steps through gpu_queue.sh, so
# checking on progress means tailing one log instead of re-launching each
# step by hand:
#   1. Re-run Task 9's end-to-end pipeline GPU test (last blocked by shared-
#      GPU contention, not a code bug -- see kg_test/graft/env.py and
#      refine.py's docstrings for the two real bugs already fixed).
#   2. On success, the first real Task 11 eval run: 6 species x
#      {ours,b0,b1,b2} against real held-out Treevill references.
#
# min_free_mib=7800: kg_build.py/verify.py/generate.py now unload each
# model right after its own phase (not just between generate/verify at the
# refine-loop level), so peak usage at any instant is one model at a time --
# ~5GB (vlm nf4 or reranker bf16) or ~7-8GB (SDXL+IP-Adapter), never the
# ~13-14GB sum. A prior OOM trace measured SDXL's true need at ~7.6GB
# (failed allocating the last 20MiB with 18.81MiB free after 7.57GB was
# already placed). 7800 leaves a thin but real margin above that measured
# floor. (Started at 15000, sized for the OLD all-models-resident peak;
# dropped to 9000 once phases were split; dropped again to 7800 after this
# card's free memory plateaued at 5.6GB for 1.5h straight under sustained
# contention from another researcher's job -- gpu_queue.sh's OOM-triggered
# retry covers the residual risk of this thinner margin.)
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
QUEUE=./scripts/gpu_queue.sh
LOG=logs/phase1_remaining.log
: > "$LOG"  # fresh log for this run

echo "[driver] $(date -Iseconds) step 1/2: Task 9 pipeline GPU test" | tee -a "$LOG"
if ! "$QUEUE" 7800 "$LOG" 3 -- "$PY" -m pytest tests/test_pipeline_gpu.py -v -m gpu; then
  echo "[driver] $(date -Iseconds) step 1/2 FAILED -- stopping before the eval run" | tee -a "$LOG"
  exit 1
fi
echo "[driver] $(date -Iseconds) step 1/2 PASSED" | tee -a "$LOG"

echo "[driver] $(date -Iseconds) step 2/2: Task 11 real eval run (6 species x ours,b0,b1,b2)" | tee -a "$LOG"
if ! "$QUEUE" 7800 "$LOG" 3 -- "$PY" -m graft.run_eval \
      --root data/treevill/rawdata2 \
      --n-species 6 \
      --methods ours,b0,b1,b2 \
      --config configs/pipeline_eval_run.yaml; then
  echo "[driver] $(date -Iseconds) step 2/2 FAILED" | tee -a "$LOG"
  exit 1
fi
echo "[driver] $(date -Iseconds) step 2/2 PASSED -- Phase 1 remaining work complete" | tee -a "$LOG"
