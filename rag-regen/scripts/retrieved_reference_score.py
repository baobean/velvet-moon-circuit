#!/usr/bin/env python
"""Retrieved-reference re-scoring for the deployable detector Phase A.

Two model residencies, never co-resident:

1. Retriever (SigLIP + LAION FAISS index): for each case, build the frozen
   reference query, search deterministically, take the top hit, and check it for
   exact-hash contamination against the case's ground-truth references. Written to
   an immutable ``retrieval.json``.
2. Reranker (Qwen3-VL-Reranker-2B): re-score the DRAFT's reference_relevance
   against the retrieved reference AND re-score it against the oracle reference so
   parity against ``qwen_reranker.json`` can be required before any oracle-vs-
   retrieved comparison. text_relevance is copied bit-for-bit from the oracle
   artifact (it is scored against a text-only query, reference-independent).

Recovery contract: ``retrieval.json`` is reused when present (never refused), so a
reranker or parity failure can be resumed without re-running GPU retrieval; only
the final ``retrieved_reranker.json`` is overwrite-guarded. ``--retrieval-only``
stops after retrieval.

No image generation, no attempts, no DINO, no label edits.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, metrics, retrieve  # noqa: E402
from ragregen.eval_manifest import sha256_file  # noqa: E402
from ragregen.retrieved_reference import build_case_record, contamination  # noqa: E402
from scripts.qwen_reranker_score import (  # noqa: E402
    MAX_IMAGE_SIDE, MODEL_ID, PROMPT, _score, _thumbnail)
from scripts.reference_embed import _crop  # noqa: E402

PARITY_ATOL = 2e-3


def _draft_crop(screen_run: Path, candidate_run: Path, stream_a, case):
    mask_path = candidate_run / case.id / "mask.png"
    with Image.open(screen_run / case.id / "draft.png") as source:
        if mask_path.is_file():
            mask = Image.open(mask_path).convert("L")
            crop = metrics.crop_to_mask(source.convert("RGB"), mask)
            mask.close()
        else:
            box = stream_a[case.id][case.concept].get("box")
            crop = _crop(source, box)
    return _thumbnail(crop)


def _exact_query(case) -> str:
    return (f"A clear photo of the exact fine-grained concept '{case.concept}', "
            f"not merely a generic {case.coarse}.")


def _retrieve_all(args) -> dict:
    """GPU residency 1. Returns the immutable retrieval record."""
    dataset = config.load_dataset(args.dataset)
    ocases = json.loads(args.oracle_reranker.read_text()).get("cases", {})
    db = config.load_retrieval_db()

    from ragregen import encoders
    retriever = retrieve.Retriever.from_index(
        db.index_path, encoders.build_encoder(db.encoder, device=args.device))

    cases = {}
    for case in dataset.cases:
        if case.id not in ocases:
            continue
        query = retrieve.reference_query(case.concept, case.coarse)
        hits = retriever.search(query, args.k)
        if not hits:
            raise ValueError(f"{case.id}: retrieval returned no hits")
        top = hits[0]
        cases[case.id] = {
            "query": query,
            "hits": [{"path": str(h.path), "score": h.score, "rank": h.rank}
                     for h in hits],
            "top": {"path": str(top.path), "score": top.score, "rank": top.rank},
            "image_sha256": sha256_file(top.path),
            "contaminated": contamination(top.path, case.gt_refs),
        }
        print(f"[retrieve] {case.id}: top={Path(top.path).name} "
              f"score={top.score:.3f} contaminated={cases[case.id]['contaminated']}",
              flush=True)

    del retriever
    env.reclaim_gpu()
    return {
        "provenance": {
            "query_builder": "ragregen.retrieve.reference_query",
            "encoder": db.encoder, "index_path": str(db.index_path),
            "index_sha256": sha256_file(db.index_path), "corpus": db.name,
            "dataset_sha256": sha256_file(args.dataset)},
        "cases": cases,
    }


def _rescore_all(args, retrieval: dict):
    """GPU residency 2. Returns (results, package_versions).

    ``results[case]`` carries both the retrieved reference_relevance and an oracle
    re-score of the same draft against the oracle reference, for parity.
    """
    dataset = config.load_dataset(args.dataset)
    ocases = json.loads(args.oracle_reranker.read_text()).get("cases", {})
    stream_a = json.loads((args.screen_run / "stream_a.json").read_text())

    import torch
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_ID, device=args.device,
                         cache_folder=str(env.HF_CACHE), local_files_only=True,
                         model_kwargs={"torch_dtype": torch.bfloat16,
                                       "attn_implementation": "sdpa"})

    results = {}
    for case in dataset.cases:
        if case.id not in ocases:
            continue
        draft = _draft_crop(args.screen_run, args.candidate_run, stream_a, case)
        query = _exact_query(case)

        with Image.open(retrieval["cases"][case.id]["top"]["path"]) as src:
            retrieved_ref = _thumbnail(src.convert("RGB"))
        retrieved_rel = _score(model, {"text": query, "image": retrieved_ref},
                               [draft])[0]
        retrieved_ref.close()

        edit_refs, _ = metrics.split_refs(case.gt_refs, heldout=1)
        with Image.open(edit_refs[0]) as src:
            oracle_ref = _thumbnail(src.convert("RGB"))
        oracle_rel = _score(model, {"text": query, "image": oracle_ref},
                            [draft])[0]
        oracle_ref.close()
        draft.close()

        results[case.id] = {"reference_relevance": retrieved_rel,
                            "oracle_reference_relevance": oracle_rel}
        print(f"[reranker] {case.id}: retrieved={retrieved_rel:.4f} "
              f"oracle_rescored={oracle_rel:.4f}", flush=True)

    env.reclaim_gpu()
    versions = {}
    for pkg in ("sentence_transformers", "torch", "transformers", "faiss-cpu"):
        try:
            versions[pkg] = _pkg_version(pkg)
        except Exception:
            versions[pkg] = None
    return results, versions


def main(argv=None, *, retrieve_fn=_retrieve_all, rescore_fn=_rescore_all) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--oracle-reranker", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--retrieval-out", type=Path, required=True)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--parity-atol", type=float, default=PARITY_ATOL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    # Only the final artifact is overwrite-guarded; retrieval.json is immutable
    # and reused, which is what makes recovery after a reranker failure possible.
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    if args.retrieval_out.is_file():
        retrieval = json.loads(args.retrieval_out.read_text())
        print(f"[retrieved-ref] reusing {args.retrieval_out}")
    else:
        retrieval = retrieve_fn(args)
        args.retrieval_out.parent.mkdir(parents=True, exist_ok=True)
        args.retrieval_out.write_text(json.dumps(retrieval, indent=2))
        print(f"[retrieved-ref] wrote {args.retrieval_out}")

    if args.retrieval_only:
        n_contaminated = sum(1 for v in retrieval["cases"].values()
                             if v["contaminated"])
        print(f"[retrieved-ref] retrieval-only: {len(retrieval['cases'])} cases, "
              f"{n_contaminated} contaminated")
        return 0

    results, versions = rescore_fn(args, retrieval)

    # Oracle parity: the re-scored oracle references must reproduce the stored
    # oracle numbers, or the two score sets are not comparable.
    oracle = json.loads(args.oracle_reranker.read_text())["cases"]
    diffs = {cid: abs(results[cid]["oracle_reference_relevance"]
                      - oracle[cid]["scores"]["draft"]["reference_relevance"])
             for cid in results}
    max_diff = max(diffs.values()) if diffs else 0.0
    if max_diff > args.parity_atol:
        worst = max(diffs, key=diffs.get)
        raise SystemExit(
            f"oracle parity FAILED: max abs diff {max_diff:.6f} > "
            f"{args.parity_atol} (worst: {worst}). Refusing to combine "
            f"incomparable scores -- environment/model mismatch.")

    artifact = {
        "schema": 1, "model": "qwen3-vl-reranker-2b", "prompt": PROMPT,
        "reference_source": "retrieved_laion", "max_image_side": MAX_IMAGE_SIDE,
        "retriever": retrieval.get("provenance", {}),
        "oracle_artifact": {"path": str(args.oracle_reranker),
                            "sha256": sha256_file(args.oracle_reranker)},
        "package_versions": versions,
        "oracle_parity": {"max_abs_diff": max_diff, "atol": args.parity_atol,
                          "per_case": diffs},
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cases": {},
    }
    for cid, res in results.items():
        rc = retrieval["cases"][cid]
        artifact["cases"][cid] = build_case_record(
            oracle[cid], retrieved_reference_relevance=res["reference_relevance"],
            hit={"query": rc["query"], **rc["top"]},
            image_sha256=rc["image_sha256"], contaminated=rc["contaminated"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    n_contaminated = sum(1 for v in retrieval["cases"].values()
                         if v["contaminated"])
    print(f"[retrieved-ref] parity OK (max diff {max_diff:.6f}); wrote {args.out} "
          f"({len(artifact['cases'])} cases, {n_contaminated} contaminated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
