#!/usr/bin/env bash
# Stage-1 MMKG verifier: one VLM-read batch + offline gate/report.
# See ragregen/mmkg/run_stage1.py for the full flow.
#
# Usage:
#   ./scripts/mmkg_stage1.sh [--out DIR] [extra run_stage1.py args...]
#
# Detached (mirrors GRAFT's VLM-load memory guard -- one GPU load, hours):
#   setsid ./scripts/mmkg_stage1.sh > outputs/mmkg_stage1/run.log 2>&1 < /dev/null &
set -euo pipefail

PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache
export PYTHONPATH="$ROOT"

OUT="outputs/mmkg_stage1"
mkdir -p "$OUT"

exec "$PY" -m ragregen.mmkg.run_stage1 --out "$OUT" "$@"
