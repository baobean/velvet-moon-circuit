#!/usr/bin/env python
"""Report final VLM decisions against pass and non-pass labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, verifier_eval  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="screen run containing labels.csv and stream_b.json")
    ap.add_argument("--column", choices=c1.VERDICT_COLUMNS,
                    default="verdict_identity")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="decision JSON (default: <run>/stream_b.json)")
    ap.add_argument("--stem", default=None,
                    help="output stem (default: verifier_<column>)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="artifact directory (default: --run)")
    args = ap.parse_args(argv)
    labels = args.run / "labels.csv"
    decisions = args.decisions or args.run / "stream_b.json"
    for path in (labels, decisions):
        if not path.is_file():
            print(f"[error] missing {path}")
            return 2

    result = verifier_eval.evaluate(c1.load_labels(labels),
                                    json.loads(decisions.read_text()),
                                    args.column)
    stem = args.stem or f"verifier_{args.column}"
    output_dir = args.output_dir or args.run
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    md = verifier_eval.render(result)
    (output_dir / f"{stem}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
