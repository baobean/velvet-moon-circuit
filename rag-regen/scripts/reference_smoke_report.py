#!/usr/bin/env python
"""Render an embedding artifact as an auditable reference-verifier report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import reference_smoke  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default="reference_smoke")
    ap.add_argument("--pairwise-dino", type=Path, default=None,
                    help="held-out DINO artifact; switches to selector report")
    args = ap.parse_args(argv)
    embeddings = json.loads(args.embeddings.read_text())
    if args.pairwise_dino:
        dino = json.loads(args.pairwise_dino.read_text())
        result = reference_smoke.evaluate_pairwise(embeddings, dino)
        md = reference_smoke.render_pairwise(result)
    else:
        result = reference_smoke.evaluate(embeddings)
        md = reference_smoke.render(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.stem}.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / f"{args.stem}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
