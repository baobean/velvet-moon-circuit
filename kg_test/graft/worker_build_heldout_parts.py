"""GPU worker: for each HELD-OUT ref of one concept, detect+segment+crop+embed every
part -> the held-out part-geometry store. Used by the restore worker for (a) target
selection, (b) scoring references (other held-out P crops), (c) the cached box+SAM mask.
Mirrors worker_build_partcrops but on the held-out split (disjoint from the build store)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.kg_build import PART_NAMES, _part_phrase
from graft.models import Models
from graft import parts


def heldout_record(concept, part, ref_path, box, mask_path, crop_path, emb):
    return {"concept": concept, "part": part, "ref_path": ref_path,
            "box": [int(round(v)) for v in box], "mask_path": mask_path,
            "crop_path": crop_path, "siglip2": list(emb)}


def main():
    ap = argparse.ArgumentParser()
    for f in ("root", "concept", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    sp = load_species(a.root, a.concept)
    _build, heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
    models = Models(cfg)
    cdir = os.path.join(a.out, "_heldcrops", a.concept); os.makedirs(cdir, exist_ok=True)
    mdir = os.path.join(a.out, "_heldmasks", a.concept); os.makedirs(mdir, exist_ok=True)
    recs = []
    for ri, ref_path in enumerate(heldout):
        img = Image.open(ref_path).convert("RGB")
        for part in PART_NAMES:
            boxes = models.detector.detect(img, _part_phrase(part))
            if not boxes:
                continue
            box = parts.clamp_box(boxes[0], *img.size)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            mask = models.segmenter.mask(img, box)
            crop = parts.crop_region(img, box)
            cp = os.path.join(cdir, f"{part}_{ri}.png"); crop.save(cp)
            mp = os.path.join(mdir, f"{part}_{ri}.npy"); np.save(mp, mask)
            emb = models.siglip.embed_image([crop])[0].tolist()
            recs.append(heldout_record(a.concept, part, ref_path, box, mp, cp, emb))
    models.unload("detector"); models.unload("segmenter"); models.unload("siglip")
    os.makedirs(os.path.join(a.out, "heldout_parts"), exist_ok=True)
    with open(os.path.join(a.out, "heldout_parts", f"{a.concept}.json"), "w") as f:
        json.dump(recs, f)
    print(f"[heldout_parts] {a.concept}: {len(recs)} records over {len(heldout)} held-out refs", flush=True)


if __name__ == "__main__":
    main()
