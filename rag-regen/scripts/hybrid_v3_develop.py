"""CPU-only v3 development study: v2 detector + semantic-gated selector.

Consumes the frozen manifest, the v2 artifacts (semantic/reranker/DINO), and the
new candidate-semantic verdicts, runs the leave-one-concept-out evaluation with
the v3 selector, and writes a development report. Freezes a policy only on a
numeric-PASS readiness.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import eval_manifest, hybrid_verifier_v3
from scripts.hybrid_v2_develop import _render, _verify_hash, _write_folds_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--candidate-semantic", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")

    manifest = json.loads(args.manifest.read_text())
    semantic = _verify_hash("semantic", args.semantic, manifest)
    reranker = _verify_hash("reranker", args.reranker, manifest)
    dino = _verify_hash("dino", args.dino, manifest)
    candidate_semantic = json.loads(args.candidate_semantic.read_text())

    result = hybrid_verifier_v3.develop(
        manifest, semantic, reranker, dino, candidate_semantic)
    result["sources"] = dict(manifest.get("sources", {}))
    result["sources"]["candidate_semantic"] = {
        "path": str(args.candidate_semantic),
        "sha256": eval_manifest.sha256_file(args.candidate_semantic)}
    result["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    out.mkdir(parents=True, exist_ok=True)
    (out / "development.json").write_text(json.dumps(result, indent=2))
    (out / "development.md").write_text(_render(result))
    _write_folds_csv(out / "folds.csv", result["folds"])

    print(_render(result), end="")
    print(f"readiness: {result['readiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
