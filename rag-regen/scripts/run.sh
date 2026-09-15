#!/usr/bin/env bash
# The operator's only entry point. See docs/RUNBOOK.md.
set -euo pipefail

PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache

usage() {
  cat <<'EOF'
Usage: ./scripts/run.sh <stage> [options]

  validate     preflight your configs          (no GPU, seconds)
  build-index  corpus -> FAISS index           (GPU, ~1h / 500k)
  fetch-corpus build the 100k LAION corpus     (no GPU, ~3h network)
  make-cases   ImageNet -> the 92-case set     (no GPU, ~10 min)
  supervise    keep a long run alive           (wraps pipeline)
  screen       THE GATE: draft + hand-label    (GPU, ~1h)
  bakeoff      FG-CLIP vs SigLIP, both roles   (GPU, ~1h)
  score-a      Stream A over labelled drafts  (GPU, ~5 min)
  score-b      Stream B over labelled drafts  (GPU, ~10 min)
  score-b-ref  reference-aware Stream B       (GPU, ~10 min)
  verifier-report verifier non-pass statistics (no GPU, seconds)
  hybrid-report frozen hybrid holdout report (no GPU, seconds)
  freeze-eval-manifest immutable identity/eligibility manifest (no GPU, seconds)
  hybrid-v2-develop leave-one-concept-out v2 development study (no GPU, seconds)
  retrieved-ref-score  retrieve + re-score reference_relevance (GPU, minutes)
  retrieved-detector-develop  detector-only LOCO on retrieved refs (no GPU, seconds)
  freeze-detector-policy  hash the complete frozen policy (no GPU, seconds)
  freeze-phase-b-holdout  immutable 24+24 Phase B holdout manifest (no GPU, seconds)
  phase-b-detector  frozen-policy detector gate, stops before generation (no GPU, seconds)
  phase-b-generate  conditional attempt_1 generation, detector-PASS only (GPU, minutes)
  phase-b-evaluate  end-to-end gates, thin-guard, frozen readiness (GPU, ~5 min)
  repair       repair ONE case, oracle arm      (GPU, ~1 min stitch)
  pipeline     THE LOOP: repair every case    (GPU, hours; resumable)
  reverify     re-grade a finished run's images (GPU, minutes; never
               writes to the source run)
  report       metrics over a finished run       (GPU, ~5 min)
  c1           fuse, sweep tau, report        (no GPU, seconds)

Run `validate` first. `screen` gates everything downstream.
EOF
}

[ $# -ge 1 ] || { usage; exit 1; }
STAGE="$1"; shift

case "$STAGE" in
  validate)    exec "$PY" scripts/validate_cli.py "$@" ;;
  build-index) exec "$PY" scripts/build_index.py "$@" ;;
  fetch-corpus) exec "$PY" scripts/fetch_corpus.py "$@" ;;
  make-cases)  exec "$PY" scripts/make_cases.py "$@" ;;
  supervise)   exec ./scripts/supervise.sh "$@" ;;
  screen)      exec "$PY" scripts/screen_premise.py "$@" ;;
  bakeoff)     exec "$PY" scripts/bakeoff.py "$@" ;;
  score-a)     exec "$PY" scripts/score_a.py "$@" ;;
  score-b)     exec "$PY" scripts/score_b.py "$@" ;;
  score-b-ref) exec "$PY" scripts/score_b_reference.py "$@" ;;
  verifier-report) exec "$PY" scripts/verifier_report.py "$@" ;;
  hybrid-report) exec "$PY" scripts/hybrid_verifier_report.py "$@" ;;
  freeze-eval-manifest) exec "$PY" scripts/freeze_eval_manifest.py "$@" ;;
  hybrid-v2-develop) exec "$PY" scripts/hybrid_v2_develop.py "$@" ;;
  score-b-candidates) exec "$PY" scripts/score_b_candidates.py "$@" ;;
  hybrid-v3-develop) exec "$PY" scripts/hybrid_v3_develop.py "$@" ;;
  retrieved-ref-score) exec "$PY" scripts/retrieved_reference_score.py "$@" ;;
  retrieved-detector-develop) exec "$PY" scripts/retrieved_detector_develop.py "$@" ;;
  freeze-detector-policy) exec "$PY" scripts/freeze_detector_policy.py "$@" ;;
  freeze-phase-b-holdout) exec "$PY" scripts/freeze_phase_b_holdout.py "$@" ;;
  phase-b-detector) exec "$PY" scripts/phase_b_detector.py "$@" ;;
  phase-b-generate) exec "$PY" scripts/phase_b_generate.py "$@" ;;
  phase-b-evaluate) exec "$PY" scripts/phase_b_evaluate.py "$@" ;;
  repair)      exec "$PY" scripts/repair_case.py "$@" ;;
  pipeline)    exec "$PY" scripts/run_pipeline.py "$@" ;;
  reverify)    exec "$PY" scripts/reverify.py "$@" ;;
  report)      exec "$PY" scripts/report.py "$@" ;;
  c1)          exec "$PY" scripts/c1_report.py "$@" ;;
  -h|--help)   usage ;;
  *)           echo "unknown stage: $STAGE" >&2; usage; exit 1 ;;
esac
