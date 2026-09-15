#!/usr/bin/env python
"""Score controlled drafts/cached repairs with Qwen3-VL-Reranker-2B."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, metrics  # noqa: E402
from scripts.reference_embed import _crop  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
MAX_IMAGE_SIDE = 448
PROMPT = (
    "Score whether the document image depicts the exact fine-grained visual "
    "concept in the query. Focus on identity-defining shape, texture, "
    "markings, and structure; ignore composition and background."
)


def _score(model, query, documents):
    import torch

    pairs = [(query, {"image": image}) for image in documents]
    values = model.predict(pairs, prompt=PROMPT, batch_size=1,
                           activation_fn=torch.nn.Sigmoid(),
                           show_progress_bar=False)
    return [float(value) for value in values]


def _thumbnail(image: Image.Image) -> Image.Image:
    result = image.copy()
    result.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most this many labeled cases")
    args = ap.parse_args(argv)

    import torch
    from sentence_transformers import CrossEncoder

    labels = {}
    with (args.screen_run / "labels.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["case_id"]] = row["verdict_identity"].strip().upper()
    stream = json.loads((args.screen_run / "stream_a.json").read_text())
    dataset = config.load_dataset(args.dataset)

    model = CrossEncoder(
        MODEL_ID,
        device=args.device,
        cache_folder=str(env.HF_CACHE),
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.bfloat16,
                      "attn_implementation": "sdpa"},
    )
    artifact = {"schema": 1, "model": "qwen3-vl-reranker-2b",
                "prompt": PROMPT, "reference_source": "first_edit_gt_ref",
                "max_image_side": MAX_IMAGE_SIDE,
                "cases": {}}
    for position, case in enumerate(dataset.cases, start=1):
        if case.id not in labels:
            continue
        if args.limit is not None and len(artifact["cases"]) >= args.limit:
            break
        case_dir = args.candidate_run / case.id
        mask_path = case_dir / "mask.png"
        mask = Image.open(mask_path).convert("L") if mask_path.is_file() else None
        with Image.open(args.screen_run / case.id / "draft.png") as source:
            if mask is not None:
                draft = metrics.crop_to_mask(source.convert("RGB"), mask)
            else:
                box = stream[case.id][case.concept].get("box")
                draft = _crop(source, box)
        documents = {"draft": _thumbnail(draft)}
        draft.close()
        if mask is not None:
            for attempt in sorted(case_dir.glob("attempt_*.png")):
                with Image.open(attempt) as source:
                    crop = metrics.crop_to_mask(source.convert("RGB"), mask)
                    documents[attempt.stem] = _thumbnail(crop)
                    crop.close()
        if mask is not None:
            mask.close()

        edit_refs, heldout_refs = metrics.split_refs(case.gt_refs, heldout=1)
        with Image.open(edit_refs[0]) as source:
            converted = source.convert("RGB")
            reference = _thumbnail(converted)
            converted.close()
        exact_query = (
            f"A clear photo of the exact fine-grained concept '{case.concept}', "
            f"not merely a generic {case.coarse}."
        )
        ref_query = {"text": exact_query, "image": reference}
        images = list(documents.values())
        text_scores = _score(model, exact_query, images)
        reference_scores = _score(model, ref_query, images)
        artifact["cases"][case.id] = {
            "truth": labels[case.id], "concept": case.concept,
            "coarse": case.coarse, "heldout_reference_paths": [str(p) for p in heldout_refs],
            "scores": {
                label: {"text_relevance": text_score,
                        "reference_relevance": reference_score}
                for label, text_score, reference_score in zip(
                    documents, text_scores, reference_scores)
            },
        }
        for image in documents.values():
            image.close()
        reference.close()
        print(f"[reranker] {position}/{len(dataset.cases)} {case.id}: "
              f"{len(documents)} images", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact))
    print(f"[reranker] wrote {args.out} ({len(artifact['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
