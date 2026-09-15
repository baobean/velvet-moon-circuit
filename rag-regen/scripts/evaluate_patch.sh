#!/usr/bin/env bash
# Gated evaluation for the inpainting/verifier repair. Safe to resume: model
# stages persist per-case JSON and pipeline stages persist queue.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCREEN="${SCREEN_RUN:-outputs/screen_20260726_233320}"
LABELS="$SCREEN/labels.csv"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"
PILOT_LIMIT="${PILOT_LIMIT:-6}"

wait_gpu() {
  local need="$1"
  shift
  ./scripts/gpu_wait.sh --need "$need" --stable 2 \
    --interval "$WAIT_INTERVAL" --timeout "$WAIT_TIMEOUT" -- "$@"
}

# Candidate semantic fix: references shown beside the candidate.
wait_gpu 8 ./scripts/run.sh score-b-ref \
  --labels "$LABELS" --dataset configs/dataset.yaml --arm oracle --max-refs 2 \
  --out "$SCREEN/stream_b_ref_oracle.json"
./scripts/run.sh verifier-report --run "$SCREEN" \
  --decisions "$SCREEN/stream_b_ref_oracle.json" \
  --stem verifier_ref_oracle_verdict_identity

# Candidate grounded fix: retrieved image prototypes, with raw margins saved
# for an offline delta sweep. It is not made live until this report calibrates
# a development threshold.
wait_gpu 8 ./scripts/run.sh score-a \
  --labels "$LABELS" --dataset configs/dataset.yaml --proto-arm retrieved \
  --out "$SCREEN/stream_a_retrieved.json"
./scripts/run.sh c1 --run "$SCREEN" \
  --stream-a "$SCREEN/stream_a_retrieved.json" \
  --stream-b "$SCREEN/stream_b.json" \
  --stem c1_retrieved_prompt

# Verifier-free causal arm: every case is edited once; no routing, early stop,
# best-of-N, or evaluation-metric selection.
wait_gpu 14 ./scripts/run.sh pipeline \
  --tag patch_no_verifier --arm oracle --mechanism inpaint --verifier none \
  --ref-prep crop --screen-run "$SCREEN" --limit "$PILOT_LIMIT"
PILOT_RUN="$(readlink -f outputs/patch_no_verifier_latest)"
wait_gpu 14 ./scripts/run.sh report --run "$PILOT_RUN" \
  --screen-run "$SCREEN" --dataset configs/dataset.yaml

printf '[evaluate-patch] complete: verifier=%s pilot=%s\n' "$SCREEN" "$PILOT_RUN"
