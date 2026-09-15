#!/usr/bin/env python
"""Reference-aware VLM decisions over labeled drafts.

Oracle references are a diagnostic ceiling. Retrieved references can be read
from an existing pipeline run without loading the retriever beside Qwen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, config, env, trace, vlm  # noqa: E402
from ragregen.verify.semantic import ReferenceSemanticVerifier  # noqa: E402
from scripts.score_b import _write  # noqa: E402


def score_case(verifier, image, prompt, concept, references) -> dict:
    verdict = verifier.judge(image, prompt, concept, references)
    return {
        "ok": verdict.ok, "degenerate": verdict.degenerate,
        "raw": verdict.raw,
        "issues": [{"concept": i.concept, "problem": i.problem}
                   for i in verdict.issues],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--arm", choices=("oracle", "retrieved"), default="oracle")
    ap.add_argument("--references-run", type=Path, default=None,
                    help="required for retrieved: <case>/refs.json")
    ap.add_argument("--max-refs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantize", default="nf4")
    ap.add_argument("--max-image-pixels", type=int, default=640 * 640,
                    help="per-image vision budget; bounds multi-image VRAM")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.arm == "retrieved" and args.references_run is None:
        ap.error("--arm retrieved requires --references-run")
    if args.max_refs < 1:
        ap.error("--max-refs must be >= 1")

    rows = c1.load_labels(args.labels)
    ds = config.load_dataset(args.dataset)
    by_id = {case.id: case for case in ds.cases}
    out_path = args.out or args.labels.parent / f"stream_b_ref_{args.arm}.json"
    existing = {} if args.force or not out_path.is_file() \
        else json.loads(out_path.read_text())
    todo = [row for row in rows if row.case_id not in existing]
    print(f"[score-b-ref] {len(todo)} to score, {len(existing)} cached -> "
          f"{out_path}")
    if not todo:
        return 0

    def paths_for(cid):
        if args.arm == "oracle":
            return list(by_id[cid].gt_refs)[:args.max_refs]
        path = args.references_run / cid / "refs.json"
        if not path.is_file():
            raise FileNotFoundError(f"{cid}: missing {path}")
        return [Path(p) for p in json.loads(path.read_text())][:args.max_refs]

    drafts = {row.case_id: c1.resolve_draft(row, args.labels) for row in todo}
    references = {row.case_id: paths_for(row.case_id) for row in todo}
    for row in todo:
        paths = [drafts[row.case_id], *references[row.case_id]]
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{row.case_id}: missing {', '.join(missing)}")

    run = trace.open_run("score_b_reference", argv=sys.argv, args=vars(args))
    from PIL import Image
    verifier = ReferenceSemanticVerifier(
        vlm.QwenVLM(device=args.device, quantize=args.quantize,
                    max_image_pixels=args.max_image_pixels))
    for i, row in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {row.case_id}", flush=True)
        image = Image.open(drafts[row.case_id]).convert("RGB")
        refs = [Image.open(p).convert("RGB") for p in references[row.case_id]]
        existing[row.case_id] = score_case(
            verifier, image, row.prompt, row.concept, refs)
        _write(out_path, existing)
    run.finish("ok", {"n_scored": len(todo), "arm": args.arm,
                      "out": str(out_path)})
    env.reclaim_gpu()
    print(f"[score-b-ref] wrote {len(existing)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    # torch + transformers + PIL occasionally segfault while unloading after
    # every artifact has already been committed (the same native teardown
    # failure run_pipeline avoids). Flush and exit without unloading them.
    env.exit_now(main())
