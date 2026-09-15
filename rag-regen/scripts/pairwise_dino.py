#!/usr/bin/env python
"""Held-out-DINO answer key for cached draft/repair comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, metrics  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    ds = config.load_dataset(args.dataset)
    pipe = config.load_pipeline(args.pipeline)
    encoder = metrics.DinoEncoder(pipe.eval_dino, args.device)
    artifact = {"schema": 1, "metric": pipe.eval_dino,
                "heldout_refs": pipe.eval_heldout_refs, "cases": {}}
    try:
        for case in ds.cases:
            case_dir = args.candidate_run / case.id
            attempts = sorted(case_dir.glob("attempt_*.png"))
            mask_path = case_dir / "mask.png"
            if not attempts or not mask_path.is_file():
                continue
            mask = Image.open(mask_path).convert("L")
            _, heldout = metrics.split_refs(case.gt_refs,
                                            pipe.eval_heldout_refs)
            ref_vecs = []
            for path in heldout:
                with Image.open(path) as source:
                    ref_vecs.append(encoder.embed(source.convert("RGB")))

            images = {"draft": args.screen_run / case.id / "draft.png"}
            images.update({p.stem: p for p in attempts})
            scores = {}
            for label, path in images.items():
                with Image.open(path) as source:
                    crop = metrics.crop_to_mask(source.convert("RGB"), mask)
                vector = encoder.embed(crop)
                scores[label] = float(np.mean([vector @ ref for ref in ref_vecs]))
            mask.close()
            artifact["cases"][case.id] = scores
            print(f"[pairwise-dino] {case.id}: {len(attempts)} attempts", flush=True)
    finally:
        encoder.free()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    print(f"[pairwise-dino] wrote {args.out}")
    env.exit_now(0)


if __name__ == "__main__":
    main()
