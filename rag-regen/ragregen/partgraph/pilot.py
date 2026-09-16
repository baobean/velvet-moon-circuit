"""Run one species through one arm and score it (held-out DINO + preservation)."""
from __future__ import annotations

from PIL import Image

from ragregen import metrics
from ragregen.partgraph.arms import reference_for


def run_case(*, arm, store, global_id, draft, mask, gt_refs, encoder, sink,
             heldout: int = 1, tile: int = 384) -> dict:
    reference = reference_for(arm, store, global_id, tile=tile)
    result = sink(draft, mask, reference)
    output = result.image

    _edit_pool, held = metrics.split_refs(gt_refs, heldout)
    out_crop = metrics.crop_to_mask(output, mask)
    held_crops = [metrics.crop_to_mask(Image.open(p).convert("RGB"),
                                       Image.new("L", Image.open(p).size, 255))
                  for p in held]
    dino = metrics.dino_identity(out_crop, held_crops, encoder)
    pres = metrics.preservation(draft, output, result.alpha)
    return {"arm": arm, "global_id": global_id, "dino": dino,
            "preservation": pres["score"], "ref_size": list(reference.size)}
