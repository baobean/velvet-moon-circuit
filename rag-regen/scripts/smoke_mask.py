#!/usr/bin/env python
"""Look at what the masker actually selects, on real photographs.

Unit tests pin the selection rules against fakes. This answers the different
question of whether the phrases in `configs/dataset.yaml` are usable detector
prompts at all -- design §10 risk M2: the `coarse` terms were authored as
contrastive vocabulary for the verifier, and nothing has checked that "food"
boxes the sushi rather than the plate.

**These are reference photographs, not drafts.** No drafts exist yet. A gt_ref
is the real concept rendered correctly, so `mask_draft`'s coarse grounding is
exercised here on an easier image than it will face. Read the output as "does
this phrase ground at all", not as a repair-quality measurement.

Usage:
  ./scripts/smoke_mask.py --cases sushi,stack_of_books --out outputs/smoke_mask
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, mask, models  # noqa: E402

DRAFT_RGB = (220, 40, 40)      # what mask_draft would destroy
REF_RGB = (40, 190, 90)        # what mask_reference would copy


def _overlay(image, result, rgb, alpha: float = 0.45):
    """Tint the masked pixels and outline the selected box."""
    from PIL import Image, ImageDraw
    import numpy as np

    out = image.convert("RGB").copy()
    arr = np.asarray(out).astype(np.float32)
    m = (np.asarray(result.mask) > 0.5)[..., None]
    tint = np.array(rgb, dtype=np.float32)
    arr = np.where(m, arr * (1 - alpha) + tint * alpha, arr)
    out = Image.fromarray(arr.astype("uint8"), "RGB")

    d = ImageDraw.Draw(out)
    d.rectangle([int(v) for v in result.box], outline=rgb, width=3)
    return out


def _row(case_id: str, kind: str, phrase: str, result) -> str:
    if result is None:
        return f"| {case_id} | {kind} | `{phrase}` | — | NOT GROUNDED | — | — |"
    import numpy as np

    frac = float((np.asarray(result.mask) > 0.5).mean())
    box = ", ".join(f"{v:.0f}" for v in result.box)
    return (f"| {case_id} | {kind} | `{phrase}` | {result.n_candidates} | "
            f"{result.selected_by} | ({box}) | {frac:.3f} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--cases", default="",
                    help="comma-separated case ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", type=Path, default=Path("outputs/smoke_mask"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dilate-px", type=int, default=12)
    ap.add_argument("--prototypes", action="store_true",
                    help="build a gt_refs prototype bank and use it to select "
                         "among candidates (the ceiling arm's selection rule)")
    ap.add_argument("--whole-photo-protos", action="store_true",
                    help="build prototypes from uncropped references. A "
                         "diagnostic: it reproduces the size bias measured in "
                         "findings/2026-07-28-masking-result.md, where the "
                         "durian prototype selected the backdrop.")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = [c for c in ds.cases if not wanted or c.id in wanted]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print(f"[smoke-mask] no cases matched {wanted}")
        return 2

    missing = [c.id for c in cases if not c.gt_refs or not Path(c.gt_refs[0]).is_file()]
    if missing:
        print(f"[error] no readable gt_ref for: {', '.join(missing)}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[smoke-mask] {len(cases)} cases -> {args.out}", flush=True)

    from PIL import Image

    masker = mask.Masker(models.DinoDetector(device=args.device),
                         sam=models.SamSegmenter(device=args.device))

    bank = None
    if args.prototypes:
        from ragregen import encoders
        from ragregen.verify import prototype

        pipe = config.load_pipeline(config.DEFAULT_PIPELINE_PATH)
        embedder = encoders.build_encoder(pipe.retriever, device=args.device)

        # Only the cases under test: each scores against its own concept, and
        # building all 22 costs 66 SAM passes for prototypes nothing reads.
        subset = replace(ds, cases=cases)
        prepare = None
        if not args.whole_photo_protos:
            def prepare(image, phrase):
                return mask.crop_to_object(image, phrase, masker)

        kind = "whole-photo" if args.whole_photo_protos else "object-cropped"
        bank = prototype.CropClassifier(
            prototype.bank_from_gt_refs(subset, embedder, prepare=prepare),
            embedder)
        print(f"[smoke-mask] {kind} prototypes for {len(cases)} concepts",
              flush=True)

    rows = ["| case | wrapper | phrase | cands | selected_by | box | mask frac |",
            "|---|---|---|---|---|---|---|"]
    for i, case in enumerate(cases, 1):
        path = Path(case.gt_refs[0])
        image = Image.open(path).convert("RGB")
        print(f"  [{i}/{len(cases)}] {case.id}  {path.name}", flush=True)

        drafted = mask.mask_draft(image, case.coarse, masker, prototypes=bank,
                                  score_phrase=case.concept,
                                  dilate_px=args.dilate_px)
        referenced = mask.mask_reference(image, case.concept, masker,
                                         prototypes=bank)

        rows.append(_row(case.id, "draft", case.coarse, drafted))
        rows.append(_row(case.id, "reference", case.concept, referenced))

        if drafted is not None:
            _overlay(image, drafted, DRAFT_RGB).save(
                args.out / f"{case.id}_draft.png")
        if referenced is not None:
            _overlay(image, referenced, REF_RGB).save(
                args.out / f"{case.id}_reference.png")

    masker.free()
    env.reclaim_gpu()

    table = "\n".join(rows)
    (args.out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\n[smoke-mask] overlays and summary.md -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
