"""M1/M3 instrumentation (spec section 8.2): GroundingDINO per-part hit-rate
and part-similarity distribution on a validation subset, so the prune decision
(section 8.3) rests on evidence rather than assumption.

`summarize_audit` is pure (no torch) and unit-tested. `main` is a manual
GPU smoke-run: it detects each KG part in a species' medoid build-ref and in
the GRAFT image of ONE matrix cell (selected with --name-mode/--ip-scale, which
together name the file run_eval wrote), scores the crop pair where both boxes
exist, and writes the records plus their summary to ``outputs/audit/audit.json``.
"""
from __future__ import annotations

from typing import List, Optional


def generated_image_name(name_mode: str, ip_scale: float) -> str:
    """Filename of the GRAFT image `run_eval.evaluate` persists for one matrix
    cell: it writes ``eval_images/{method}_{name_mode}_ip{ip_scale}.png``, so
    the audit must address that exact cell. (Keep in step with the copy at the
    end of `run_eval.evaluate`'s cell loop.)"""
    return f"ours_{name_mode}_ip{ip_scale}.png"


def summarize_audit(records: List[dict]) -> dict:
    parts = sorted({r["part"] for r in records})
    out = {}
    for part in parts:
        rs = [r for r in records if r["part"] == part]
        sims = [r["sim"] for r in rs if r.get("sim") is not None]
        out[part] = {
            "n": len(rs),
            "ref_hit_rate": sum(r["ref_box"] for r in rs) / len(rs),
            "gen_hit_rate": sum(r["gen_box"] for r in rs) / len(rs),
            "mean_sim": (sum(sims) / len(sims)) if sims else None,
            "n_scored": len(sims),
        }
    return out


def _crop_sim(ref_crop, gen_crop, models) -> Optional[float]:
    """Mean of SigLIP2 + DINOv3 cosine between the two part crops."""
    import numpy as np

    from graft.verify import _cosine

    sims = [
        _cosine(models.siglip.embed_image([ref_crop])[0], models.siglip.embed_image([gen_crop])[0]),
        _cosine(models.dino.embed_image([ref_crop])[0], models.dino.embed_image([gen_crop])[0]),
    ]
    return float(np.mean(sims))


def main(argv: List[str]) -> int:
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description="MMKG part-tree audit (spec section 8.2)")
    ap.add_argument("--root", required=True, help="Treevill species-parent directory")
    ap.add_argument("--species", required=True, help="comma-separated subset of species names")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--images-dir", default="eval_images", help="subdir of outputs/<species>/ holding the generated image")
    ap.add_argument("--name-mode", default="neutral", choices=("neutral", "named"),
                    help="which arm's generated image to audit (matches run_eval's filenames)")
    ap.add_argument("--ip-scale", type=float, default=0.6,
                    help="which ip_scale cell's generated image to audit")
    ap.add_argument("--out", default=None, help="output dir (default: <outputs_dir>/audit)")
    args = ap.parse_args(argv)

    del args.root  # species are resolved under outputs_dir; --root kept for call-site parity

    from graft import env

    env.setup()

    from PIL import Image

    from graft.config import GraftConfig
    from graft.kg_build import PART_NAMES, _part_phrase
    from graft.models import Models
    from graft.schema import ConceptKG
    from graft.selection import medoid_index

    cfg = GraftConfig.from_yaml(args.config) if os.path.exists(args.config) else GraftConfig()
    models = Models(cfg)
    species = [s.strip() for s in args.species.split(",") if s.strip()]

    records: List[dict] = []
    for sp in species:
        outputs_dir = os.path.join(cfg.outputs_dir, sp)
        kg_path = os.path.join(outputs_dir, "kg.json")
        gen_path = os.path.join(
            outputs_dir, args.images_dir, generated_image_name(args.name_mode, args.ip_scale)
        )
        if not os.path.exists(kg_path):
            print(f"[audit] {sp}: no kg.json at {kg_path} -- skipping")
            continue
        if not os.path.exists(gen_path):
            print(f"[audit] {sp}: no generated image at {gen_path} -- skipping")
            continue

        kg = ConceptKG.from_json(kg_path)
        if not kg.ref_paths or not kg.ref_embeddings:
            print(f"[audit] {sp}: kg.json has no ref embeddings -- skipping")
            continue
        ref_path = kg.ref_paths[medoid_index(kg.ref_embeddings)]  # same rule kg_build uses for crop_source
        if not os.path.exists(ref_path):
            print(f"[audit] {sp}: medoid ref image missing at {ref_path} -- skipping")
            continue

        ref_img = Image.open(ref_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        for part in PART_NAMES:
            phrase = _part_phrase(part)
            ref_boxes = models.detector.detect(ref_img, phrase)
            gen_boxes = models.detector.detect(gen_img, phrase)
            sim = None
            if ref_boxes and gen_boxes:
                sim = _crop_sim(ref_img.crop(ref_boxes[0]), gen_img.crop(gen_boxes[0]), models)
            records.append(
                {"part": part, "ref_box": bool(ref_boxes), "gen_box": bool(gen_boxes), "sim": sim}
            )
        print(f"[audit] {sp}: {len(PART_NAMES)} parts recorded")

    models.unload("detector")
    models.unload("siglip")
    models.unload("dino")

    summary = summarize_audit(records)
    out_dir = args.out or os.path.join(cfg.outputs_dir, "audit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "audit.json")
    with open(out_path, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[audit] wrote {out_path} ({len(records)} records from {len(species)} species)")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
