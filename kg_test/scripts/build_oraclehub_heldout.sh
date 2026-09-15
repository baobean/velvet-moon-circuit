#!/usr/bin/env bash
# Held-out part-geometry store for the OracleHub (build_P=0) experiment. k_build_refs=0 -> ALL
# unique images become held-out (target+scoring); no build reservation is needed because build_P=0
# uses zero own crops. Fresh dir outputs/mmkg_oh so it never collides with the Stage-A' stores.
set -u
PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT=data/treevill/rawdata2
OUT=outputs/mmkg_oh
CFG=configs/oraclehub.yaml
CONCEPTS=("Akashmoni" "Ashok" "Avocado" "Bahera" "Baro bottle brush" "Cannonball Tree" "Chaplash" \
"Crown Gardenia" "Golden Shower Tree" "Guava" "Haldu" "Haritaki" "Hijol" "Holudkrishnachura" \
"Jack Fruit" "Karanja" "Khejur" "Koinar" "Mango" "Marking Nut tree" "Mastwood" "Nageshore" \
"Palm" "Piliostigma" "Sisso")
cd /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test || exit 1
for c in "${CONCEPTS[@]}"; do
  if [ -f "$OUT/heldout_parts/$c.json" ]; then echo "[skip] $c"; continue; fi
  echo "=== held-out parts: $c ==="
  "$PY" -m graft.worker_build_heldout_parts --root "$ROOT" --concept "$c" --out "$OUT" --config "$CFG"
  echo "[rc=$?] $c"
done
echo "ALL ORACLEHUB HELDOUT DONE"
