#!/usr/bin/env bash
# Build the held-out part-geometry store for the 8 Stage-A concepts (Stage-A' prep).
set -u
PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT=data/treevill/rawdata2
OUT=outputs/mmkg
CFG=configs/pipeline_eval_run.yaml
CONCEPTS=("Ashok" "Ashore" "Avocado" "Bamboo" "Camphor Tree" "Egyptian lotus" "Hijol" "Nageshore")
cd /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test || exit 1
for c in "${CONCEPTS[@]}"; do
  if [ -f "$OUT/heldout_parts/$c.json" ]; then
    echo "[skip] $c (already built)"; continue
  fi
  echo "=== building held-out parts for: $c ==="
  "$PY" -m graft.worker_build_heldout_parts --root "$ROOT" --concept "$c" --out "$OUT" --config "$CFG"
  echo "[rc=$?] $c"
done
echo "ALL HELDOUT-STORE DONE"
