#!/usr/bin/env python
"""Candidate semantic confirmation (v3 signal): run the plain semantic verifier
over every repair attempt image that already exists in a candidate bank.

The plain (non-reference) verifier is deliberate: every reference for a case is
the same identity, and each is either the held-out DINO reference (leakage) or an
edit reference the attempt was built from (circular). Scoring the attempt against
the concept prompt alone avoids both.

Loads Qwen2.5-VL only. No image generation, no DINO, no reranker.

Usage: ./scripts/run.sh score-b-candidates \\
  --dataset configs/dataset_verifier_holdout.yaml \\
  --candidate-run outputs/holdout_candidates_20260818_182938 \\
  --out outputs/hybrid_v2_development/candidate_semantic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, trace, vlm  # noqa: E402
from ragregen.verify.semantic import SemanticVerifier  # noqa: E402


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantize", default="nf4")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset = config.load_dataset(args.dataset)
    prompts = {case.id: case.prompt for case in dataset.cases}

    existing = (json.loads(args.out.read_text())
                if args.out.is_file() and not args.force else {})

    # (case_id, attempt_label, path) for every attempt image, skipping cached.
    todo = []
    for case in dataset.cases:
        case_dir = args.candidate_run / case.id
        for img in sorted(case_dir.glob("attempt_*.png")):
            label = img.stem
            if case.id in existing and label in existing[case.id]:
                continue
            todo.append((case.id, label, img))

    print(f"[score-b-candidates] {len(todo)} attempt images to score -> "
          f"{args.out}")
    if not todo:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run = trace.open_run("score_b_candidates", argv=sys.argv, args=vars(args))

    from PIL import Image
    verifier = SemanticVerifier(
        vlm.QwenVLM(device=args.device, quantize=args.quantize))

    n_degenerate = 0
    for i, (cid, label, img) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {cid}/{label}", flush=True)
        image = Image.open(img).convert("RGB")
        v = verifier.judge(image, prompts[cid])
        n_degenerate += bool(v.degenerate)
        existing.setdefault(cid, {})[label] = {
            "ok": v.ok,
            "degenerate": v.degenerate,
            "raw": v.raw,
            "issues": [{"concept": iss.concept, "problem": iss.problem}
                       for iss in v.issues],
        }
        _write(args.out, existing)

    run.finish("ok", {"n_scored": len(todo), "n_degenerate": n_degenerate})
    env.reclaim_gpu()
    print(f"[score-b-candidates] wrote {sum(len(v) for v in existing.values())} "
          f"attempt verdicts, {n_degenerate} degenerate -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
