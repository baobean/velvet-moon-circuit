"""GPU worker: crop every part from every BUILD reference of one concept and
embed each crop with SigLIP2 -> PartInstanceRec list. Held-out refs are never
touched (spec §7.4). Own subprocess per the graft GPU discipline."""
from __future__ import annotations
import argparse, json, os
from dataclasses import asdict
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs    # split_refs -> (build_refs, heldout_refs)
from graft.kg_build import PART_NAMES, _part_phrase
from graft.graph_schema import PartInstanceRec
from graft.models import Models
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--concept", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    sp = load_species(a.root, a.concept)
    build_refs, _heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)  # seed=0 matches run_eval_fast
    models = Models(cfg)
    crops_dir = os.path.join(a.out, "_crops", a.concept); os.makedirs(crops_dir, exist_ok=True)
    recs = []
    for ri, ref_path in enumerate(build_refs):
        img = Image.open(ref_path).convert("RGB")
        for part in PART_NAMES:
            boxes = models.detector.detect(img, _part_phrase(part))
            if not boxes:
                continue
            x0, y0, x1, y1 = boxes[0]
            crop = img.crop((x0, y0, x1, y1))
            cp = os.path.join(crops_dir, f"{part}_{ri}.png"); crop.save(cp)
            emb = models.siglip.embed_image([crop])[0].tolist()   # L2-normed
            recs.append(PartInstanceRec(id=f"{a.concept}::{part}::{ri}", concept=a.concept,
                                        part=part, ref_path=ref_path, crop_path=cp, siglip2=emb))
    models.unload("detector"); models.unload("siglip")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.concept}.json"), "w") as f:
        json.dump([asdict(r) for r in recs], f)

if __name__ == "__main__":
    main()
