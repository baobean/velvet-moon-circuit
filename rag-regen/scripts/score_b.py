#!/usr/bin/env python
"""Stream B over every labelled draft (spec §3).

Loads Qwen2.5-VL and nothing else. GroundingDINO and the crop scorer must not
be in this process: the card is 24 GB and other researchers routinely hold
~9 GB of it.

Usage: ./scripts/run.sh score-b --labels outputs/screen_<ts>/labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, env, trace, vlm  # noqa: E402
from ragregen.verify.semantic import SemanticVerifier  # noqa: E402


def score_case(verifier, image, prompt: str) -> dict:
    v = verifier.judge(image, prompt)
    return {
        "ok": v.ok,
        "degenerate": v.degenerate,
        "raw": v.raw,
        "issues": [{"concept": i.concept, "problem": i.problem}
                   for i in v.issues],
    }


def _write(path: Path, data: dict) -> None:
    """Write-then-rename, so a crash cannot truncate completed work."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantize", default="nf4",
                    help="nf4 (default) or none; bf16 needs ~15 GB free")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = c1.load_labels(args.labels)
    out_path = args.out or args.labels.parent / "stream_b.json"
    existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
    if args.force:
        existing = {}

    todo = [r for r in rows if r.case_id not in existing]
    print(f"[score-b] {len(todo)} to score, {len(existing)} cached -> {out_path}")
    if not todo:
        return 0

    draft_paths = {r.case_id: c1.resolve_draft(r, args.labels) for r in todo}
    for r in todo:
        if not draft_paths[r.case_id].is_file():
            print(f"[error] case '{r.case_id}': draft not found: "
                  f"{draft_paths[r.case_id]}")
            return 2

    run = trace.open_run("score_b", argv=sys.argv, args=vars(args))

    from PIL import Image
    verifier = SemanticVerifier(
        vlm.QwenVLM(device=args.device, quantize=args.quantize))

    n_degenerate = 0
    for i, r in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {r.case_id}", flush=True)
        image = Image.open(draft_paths[r.case_id]).convert("RGB")
        rec = score_case(verifier, image, r.prompt)
        n_degenerate += bool(rec["degenerate"])
        existing[r.case_id] = rec
        _write(out_path, existing)

    run.finish("ok", {"n_scored": len(todo), "n_degenerate": n_degenerate})
    env.reclaim_gpu()
    print(f"[score-b] wrote {len(existing)} cases, "
          f"{n_degenerate} degenerate -> {out_path}")
    if n_degenerate:
        print("[score-b] WARNING: degenerate replies fail closed, which "
              "inflates semantic recall. The C1 report states the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
