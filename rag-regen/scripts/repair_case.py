#!/usr/bin/env python
"""Repair one case, oracle arm. The first image in this project to look at.

Deliberately no retrieval, no queue, no VLM: the oracle arm needs none of them,
and each is a separate failure surface between the input and the output.

**Drafts do not exist yet.** Until a screen run produces them, --draft accepts
any image -- including a gt_ref, which repairs a correct image and should
therefore change little. A weak end-to-end check, and honest about being one.

Usage:
  ./scripts/repair_case.py --case boston_bull --mechanism stitch --prototypes
  ./scripts/repair_case.py --case boston_bull --mechanism inpaint --device cuda

Pass --prototypes wherever the coarse term grounds more than one candidate.
Without it the mask can be the wrong PART of the right animal, and a repair
onto the wrong part tells you nothing about the mechanism under test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import composite, config, env, mask, models, regen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap.add_argument("--case", required=True)
    ap.add_argument("--mechanism", choices=regen.MECHANISMS, default="stitch")
    ap.add_argument("--draft", type=Path, default=None,
                    help="defaults to the case's first gt_ref")
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--out", type=Path, default=Path(f"outputs/repair_{ts}"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref-prep", choices=("none", "crop", "crop_matte"),
                    default=None,
                    help="override pipeline.yaml; none reproduces the old "
                         "full-reference behavior")
    ap.add_argument("--prototypes", action="store_true",
                    help="select among candidates by nearest object-cropped "
                         "prototype instead of detector confidence. Needed "
                         "wherever the coarse term grounds more than one "
                         "candidate: on boston_bull, confidence takes the HEAD "
                         "and the repair pastes a whole dog onto a neck "
                         "(findings/2026-07-28-masking-result.md).")
    ap.add_argument("--proto-device", default="cpu",
                    help="device for the prototype encoder. Defaults to cpu: "
                         "prototype selection makes SigLIP resident beside "
                         "DINO and SAM, which OOMed the shared 4090 even for a "
                         "single case (same finding, §6).")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    ref_prep = args.ref_prep or pipe_cfg.ref_prep
    cases = {c.id: c for c in ds.cases}
    if args.case not in cases:
        print(f"[error] no case '{args.case}'. Known: {', '.join(sorted(cases))}")
        return 2
    case = cases[args.case]

    draft_path = args.draft or Path(case.gt_refs[0])
    if not draft_path.is_file():
        print(f"[error] draft not found: {draft_path}")
        return 2

    from PIL import Image

    draft = Image.open(draft_path).convert("RGB")
    ref_path = Path(case.gt_refs[-1])
    reference = Image.open(ref_path).convert("RGB")
    print(f"[repair] {case.id}  draft={draft_path.name}  ref={ref_path.name}  "
          f"mechanism={args.mechanism}", flush=True)

    masker = mask.Masker(models.DinoDetector(device=args.device),
                         sam=models.SamSegmenter(device=args.device))

    bank = None
    if args.prototypes:
        from dataclasses import replace

        from ragregen import encoders
        from ragregen.verify import prototype

        embedder = encoders.build_encoder(pipe_cfg.retriever,
                                          device=args.proto_device)
        # This case only -- the other 21 concepts cost three SAM passes each
        # for prototypes nothing here reads.
        subset = replace(ds, cases=[case])
        bank = prototype.CropClassifier(
            prototype.bank_from_gt_refs(
                subset, embedder,
                # The draft is a gt_ref on this arm. Letting it into its own
                # prototype would score the crop against itself
                # (finegrained_seg.py:85).
                exclude=[draft_path],
                prepare=lambda im, phrase: mask.crop_to_object(im, phrase,
                                                               masker)),
            embedder)
        print(f"[repair] object-cropped prototypes on {args.proto_device}",
              flush=True)

    draft_mask = mask.mask_draft(draft, case.coarse, masker, prototypes=bank,
                                 score_phrase=case.concept,
                                 dilate_px=pipe_cfg.mask_dilate_px)
    if draft_mask is None:
        print(f"[repair] '{case.coarse}' not grounded in the draft -- "
              f"nothing to repair. The draft stands.")
        return 0

    ref_mask = mask.mask_reference(reference, case.concept, masker,
                                   prototypes=bank)
    cutout = composite.cutout_from(reference, ref_mask)
    prepared_reference, prep_info = regen.prepare_reference(
        reference, case.concept, mode=ref_prep,
        masker=lambda _image, _phrase: ref_mask)
    masker.free()
    # The encoder holds SigLIP; the inpaint path needs those 12 GB back.
    bank = embedder = None
    env.reclaim_gpu()

    from PIL import Image as _I
    mask_pil = _I.fromarray((draft_mask.mask > 0.5).astype("uint8") * 255, "L")

    if args.mechanism == "stitch":
        if cutout is None:
            print(f"[error] '{case.concept}' not grounded in the reference; "
                  f"stitch needs a segmented object.")
            return 2
        result = regen.stitch(draft, mask_pil, cutout,
                              feather_px=regen.KontextConfig().feather_px)
    else:
        prompt = regen.edit_prompt(case.prompt, case.coarse, case.concept)
        cfg = regen.KontextConfig(device=args.device, steps=pipe_cfg.steps,
                                  seed=pipe_cfg.seed)
        try:
            inpainter = regen.Inpainter(cfg).load()
        except RuntimeError as exc:
            # A contended card is an operating condition, not a crash. The
            # guard's message already names the remedy; a traceback on top of
            # it only buries that.
            print(f"[repair] {exc}")
            return 2
        try:
            result = inpainter.regen(draft, mask_pil, prepared_reference,
                                     prompt)
        finally:
            inpainter.free()

    out_dir = args.out / f"{case.id}_{args.mechanism}"
    out_dir.mkdir(parents=True, exist_ok=True)
    draft.save(out_dir / "before.png")
    result.image.save(out_dir / "after.png")
    mask_pil.save(out_dir / "mask.png")
    prepared_reference.save(out_dir / "reference_prepped.png")
    if result.raw is not None:
        result.raw.save(out_dir / "raw.png")
    (out_dir / "meta.json").write_text(json.dumps(
        {"case": case.id, "mechanism": result.mechanism,
         "draft": str(draft_path), "reference": str(ref_path),
         "selected_by": draft_mask.selected_by,
         "n_candidates": draft_mask.n_candidates,
         "prototypes": bool(args.prototypes),
         "ref_prep": prep_info,
         "meta": result.meta}, indent=2, default=str))

    print(f"[repair] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
