#!/usr/bin/env python
"""Render absolute and pairwise reports from a reranker score artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import reranker_smoke  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="qwen_reranker")
    ap.add_argument("--pairwise-dino", type=Path, default=None)
    args = ap.parse_args(argv)

    artifact = json.loads(args.scores.read_text())
    if args.pairwise_dino:
        dino = json.loads(args.pairwise_dino.read_text())
        result = reranker_smoke.evaluate_pairwise(artifact, dino)
        report = reranker_smoke.render_pairwise(result)
    else:
        result = reranker_smoke.evaluate_absolute(artifact)
        report = reranker_smoke.render_absolute(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.stem}.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / f"{args.stem}.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
