#!/usr/bin/env bash
# Run and score the fixed six-case causal inpainting pilot once GPU-gated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCREEN="${SCREEN_RUN:-outputs/screen_20260726_233320}"
LIMIT="${PILOT_LIMIT:-6}"
PILOT_RUN="${PILOT_RUN:-}"

PIPELINE_ARGS=(
  --arm oracle --mechanism inpaint --verifier none --ref-prep crop
  --screen-run "$SCREEN" --limit "$LIMIT"
)
if [ -n "$PILOT_RUN" ]; then
  PIPELINE_ARGS=(--resume "$PILOT_RUN" "${PIPELINE_ARGS[@]}")
else
  PIPELINE_ARGS=(--tag patch_no_verifier "${PIPELINE_ARGS[@]}")
fi

./scripts/run.sh pipeline "${PIPELINE_ARGS[@]}"

PILOT_RUN="${PILOT_RUN:-$(readlink -f outputs/patch_no_verifier_latest)}"
./scripts/run.sh report --run "$PILOT_RUN" \
  --screen-run "$SCREEN" --dataset configs/dataset.yaml

printf '[no-verifier-pilot] complete: %s\n' "$PILOT_RUN"
