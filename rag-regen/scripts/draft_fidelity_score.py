#!/usr/bin/env python
"""Automated half of the wk1 "DINO + eyeball" FLUX-failure check (spec section 8).

screen_premise.py only drafts and writes labels.csv for HAND labelling -- it
never scores anything. This script adds the automated DINO-cosine proxy: does
the draft actually look like the real species, per the SAME identity encoder
the rest of the pipeline uses for evaluation (never the retriever).

Scores the COARSE-term OBJECT CROP, never the whole frame, exactly like the
rest of this codebase's identity scoring (metrics.py's own docstring: "a rare
parrot in a wide scene scores against its own background unless the crop is
taken"). v1 of this script scored whole images and a real CUB case (Common
Tern, wk1 screen 2026-09-17) exposed why that's wrong: FLUX drew a plain
white gull, not a tern, but the whole-image score was 0.516 -- one of the
HIGHEST in the batch -- because both images share a generic "pale bird on a
sandy shore" composition that whole-image DINO rewards regardless of species.
Cropping to the grounded bird/plant before scoring removes that confound.

A case whose coarse term can't be grounded (in the draft or in enough refs)
is dropped from the mean, not silently scored on the ungrounded whole frame
-- that would reintroduce the exact bug this rewrite fixes.

Usage:
  ./scripts/draft_fidelity_score.py --dataset configs/partgraph_cub_wk1_screen.yaml \
      --run outputs/screen_20260917_131912
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--run", type=Path, required=True,
                    help="a screen_premise.py run dir, e.g. outputs/screen_<ts>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from PIL import Image

    from ragregen import mask, models

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)

    print(f"[fidelity] dino={pipe_cfg.eval_dino} device={args.device} "
          f"cases={len(ds.cases)}", flush=True)
    encoder = metrics.DinoEncoder(pipe_cfg.eval_dino, device=args.device)
    masker = mask.Masker(models.DinoDetector(device=args.device),
                         sam=models.SamSegmenter(device=args.device))

    rows = []
    dropped = []
    for case in ds.cases:
        draft_path = args.run / case.id / "draft.png"
        if not draft_path.is_file():
            print(f"  [skip] {case.id}: no draft at {draft_path}")
            dropped.append({"case_id": case.id, "reason": "no draft file"})
            continue
        draft = Image.open(draft_path).convert("RGB")
        draft_crop = mask.crop_to_object(draft, case.coarse, masker)
        if draft_crop is None:
            print(f"  [drop] {case.id}: '{case.coarse}' not grounded in the draft")
            dropped.append({"case_id": case.id, "reason": f"'{case.coarse}' not grounded in draft"})
            continue

        ref_crops = []
        for p in case.gt_refs:
            ref = Image.open(p).convert("RGB")
            crop = mask.crop_to_object(ref, case.coarse, masker)
            if crop is not None:
                ref_crops.append(crop)
        if not ref_crops:
            print(f"  [drop] {case.id}: '{case.coarse}' not grounded in any gt_ref")
            dropped.append({"case_id": case.id, "reason": f"'{case.coarse}' not grounded in any gt_ref"})
            continue

        score = metrics.dino_identity(draft_crop, ref_crops, encoder)
        rows.append({"case_id": case.id, "concept": case.concept, "dino": score,
                     "n_refs_grounded": len(ref_crops), "n_refs_total": len(case.gt_refs)})

    encoder.free()
    masker.free()
    rows.sort(key=lambda r: r["dino"])

    print("\nworst (most likely FLUX-fails) first:")
    for r in rows:
        print(f"  {r['dino']:.3f}  {r['case_id']}  ({r['concept']})")
    if dropped:
        print(f"\n[fidelity] {len(dropped)} case(s) dropped (ungrounded coarse term):")
        for d in dropped:
            print(f"  {d['case_id']}: {d['reason']}")

    scores = [r["dino"] for r in rows]
    mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"\n[fidelity] mean draft-vs-gt_refs DINO cosine (object-cropped): "
          f"{mean:.3f} over {len(rows)} cases ({len(dropped)} dropped)")

    result = {"dataset": ds.name, "run": str(args.run), "dino_model": pipe_cfg.eval_dino,
              "mean_dino": mean, "per_case": rows, "dropped": dropped,
              "generated": datetime.now().isoformat(timespec="seconds")}
    out = args.out or (env.PROJECT_ROOT / "outputs" /
                       f"fidelity_{ds.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[fidelity] wrote {out}")
    return 0


if __name__ == "__main__":
    env.exit_now(main())
