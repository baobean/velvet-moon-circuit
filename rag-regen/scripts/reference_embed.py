#!/usr/bin/env python
"""Embed the controlled drafts and oracle references for verifier smoke tests.

Run FG-CLIP with the isolated interpreter:
  .venv-fgclip/bin/python scripts/reference_embed.py --model fgclip ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, metrics  # noqa: E402


FGCLIP_REVISION = "454d76372c2cf5eb48fa0d871fd0534481484d97"
MODEL_IDS = {
    "fgclip": "qihoo360/fg-clip-base",
    "siglip": "google/siglip-so400m-patch14-384",
    "siglip2": "google/siglip2-base-patch16-384",
}


def _unit(array):
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12, None)


def _crop(image: Image.Image, box, pad_fraction: float = 0.06):
    if not box:
        return image.convert("RGB")
    x1, y1, x2, y2 = map(float, box)
    pad = max(x2 - x1, y2 - y1) * pad_fraction
    bounds = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
              min(image.width, int(x2 + pad)), min(image.height, int(y2 + pad)))
    return image.convert("RGB").crop(bounds)


class Encoder:
    def __init__(self, name: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.name = name
        self.device = device
        self.torch = torch
        model_id = MODEL_IDS[name]
        common = {"cache_dir": str(env.HF_CACHE), "local_files_only": True}
        if name == "fgclip":
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, revision=FGCLIP_REVISION, **common)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=FGCLIP_REVISION, **common)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=FGCLIP_REVISION, trust_remote_code=True,
                **common).to(device).eval()
        else:
            self.processor = AutoProcessor.from_pretrained(model_id, **common)
            self.tokenizer = None
            self.model = AutoModel.from_pretrained(model_id, **common).to(device).eval()

    @staticmethod
    def _tensor(features):
        return getattr(features, "pooler_output", features)

    def images(self, images, batch_size=8):
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                feats = self._tensor(self.model.get_image_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
            print(f"[embed] images {min(start + batch_size, len(images))}/{len(images)}", flush=True)
        return _unit(np.concatenate(chunks))

    def texts(self, texts):
        if self.name == "fgclip":
            ids = self.tokenizer(texts, max_length=77, padding="max_length",
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
            with self.torch.no_grad():
                feats = self.model.get_text_features(ids, walk_short_pos=True)
        else:
            inputs = self.processor(text=texts, padding="max_length",
                                    return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                feats = self._tensor(self.model.get_text_features(**inputs))
        return _unit(feats.float().cpu().numpy())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--candidate-run", type=Path, default=None,
                    help="optional pipeline run whose attempts are embedded "
                         "against their draft")
    args = ap.parse_args(argv)

    ds = config.load_dataset(args.dataset)
    labels = {}
    with (args.screen_run / "labels.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["case_id"]] = row["verdict_identity"].strip().upper()
    stream = json.loads((args.screen_run / "stream_a.json").read_text())

    images, layout = [], {}
    for case in ds.cases:
        if case.id not in labels:
            continue
        draft_path = args.screen_run / case.id / "draft.png"
        candidate_dir = args.candidate_run / case.id if args.candidate_run else None
        mask_path = candidate_dir / "mask.png" if candidate_dir else None
        mask = Image.open(mask_path).convert("L") if mask_path and mask_path.is_file() else None
        with Image.open(draft_path) as source:
            if mask is not None:
                draft = metrics.crop_to_mask(source.convert("RGB"), mask)
            else:
                box = stream[case.id][case.concept].get("box")
                draft = _crop(source, box)
        draft_index = len(images)
        images.append(draft)
        candidate_indices = {}
        if candidate_dir and candidate_dir.is_dir() and mask is not None:
            for attempt in sorted(candidate_dir.glob("attempt_*.png")):
                with Image.open(attempt) as source:
                    crop = metrics.crop_to_mask(source.convert("RGB"), mask)
                candidate_indices[attempt.stem] = len(images)
                images.append(crop)
        if mask is not None:
            mask.close()
        ref_indices = []
        edit_refs, heldout_refs = metrics.split_refs(case.gt_refs, heldout=1)
        for path in edit_refs:
            with Image.open(path) as source:
                ref = source.convert("RGB")
            ref_indices.append(len(images))
            images.append(ref)
        layout[case.id] = (case, draft_index, candidate_indices, ref_indices,
                           [str(p) for p in heldout_refs])

    encoder = Encoder(args.model, args.device)
    image_embeddings = encoder.images(images)
    texts = []
    for case, _, _, _, _ in layout.values():
        texts.extend([f"a photo of {case.concept}", f"a photo of {case.coarse}"])
    text_embeddings = encoder.texts(texts)

    artifact = {"schema": 1, "model": args.model, "device": args.device,
                "reference_source": "oracle_gt_refs", "cases": {}}
    for pos, (cid, (case, draft_i, candidate_i, ref_i,
                    heldout_paths)) in enumerate(layout.items()):
        artifact["cases"][cid] = {
            "truth": labels[cid], "concept": case.concept,
            "coarse": case.coarse,
            "draft_embedding": image_embeddings[draft_i].tolist(),
            "candidate_embeddings": {
                label: image_embeddings[index].tolist()
                for label, index in candidate_i.items()
            },
            "reference_embeddings": image_embeddings[ref_i].tolist(),
            "heldout_reference_paths": heldout_paths,
            "fine_text_embedding": text_embeddings[pos * 2].tolist(),
            "coarse_text_embedding": text_embeddings[pos * 2 + 1].tolist(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact))
    print(f"[embed] wrote {args.out} ({len(layout)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
