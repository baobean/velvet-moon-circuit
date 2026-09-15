#!/usr/bin/env bash
# MMKG species store build: one encoder load (+VLM for iNat attrs), re-embed
# Treevill refs + iNat medoids, write the store + build summary.
# See ragregen/mmkg_store/build.py for the full flow.
#
# Usage:
#   ./scripts/mmkg_store_build.sh [--out DIR] [extra build.py args...]
#
# Detached (mirrors the Stage-1 VLM-load memory guard -- one GPU load, hours):
#   setsid bash scripts/mmkg_store_build.sh > outputs/mmkg_store_build.log 2>&1 < /dev/null &
set -euo pipefail

PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache
export PYTHONPATH="$ROOT"

exec "$PY" -m ragregen.mmkg_store.build "$@"
