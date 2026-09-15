#!/usr/bin/env python
"""Freeze the complete Phase B detector policy manifest (no GPU, seconds).

Phase A's ``frozen_detector_policy.json`` recorded only the two thresholds;
that is insufficient to reproduce a scoring run byte-for-byte. This CLI hashes
every weight file backing the frozen rule (semantic verifier, reranker,
retrieval encoder, generator), the retrieval index, the full environment
manifest (``pip freeze`` of the executing interpreter, since only the pinned
four packages is not enough), and a clean code-snapshot hash over the exact
source files executed (not ``git HEAD`` -- the worktree is dirty).

Refuses to overwrite ``--out`` *before* touching any model artifact, so a
stale run can never be clobbered and provenance is never half-written.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config  # noqa: E402
from ragregen.detector_policy import (  # noqa: E402
    DEGENERATE_RULE, build_policy_manifest, policy_sha256,
    snapshot_code_sha256)
from ragregen.eval_manifest import sha256_file  # noqa: E402
from ragregen.verify.semantic import JUDGE_TEMPLATE  # noqa: E402
from ragregen.vlm import VLM_ID  # noqa: E402
from scripts.qwen_reranker_score import (  # noqa: E402
    MAX_IMAGE_SIDE, MODEL_ID as RERANKER_MODEL_ID, PROMPT as RERANKER_PROMPT)

GENERATOR_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
FROZEN_THRESHOLDS = {"text": 0.5, "retrieved_reference": 0.25048828125}


def _hash_weights(weight_dir: Path) -> str:
    """Sorted, combined SHA-256 over every ``*.safetensors`` under a dir."""
    files = sorted(Path(weight_dir).rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {weight_dir}")
    return snapshot_code_sha256(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds-json", type=json.loads,
                    default=json.dumps(FROZEN_THRESHOLDS))
    ap.add_argument("--semantic-weights", type=Path, required=True)
    ap.add_argument("--reranker-weights", type=Path, required=True)
    ap.add_argument("--encoder-weights", type=Path, required=True)
    ap.add_argument("--generator-weights", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--env-freeze", type=Path, required=True,
                    help="a `pip freeze` capture of the executing interpreter")
    ap.add_argument("--code-paths", nargs="+", required=True,
                    help="glob(s) over the exact source files executed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    # Refuse overwrite first: never hash a single artifact if we cannot write.
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    db = config.load_retrieval_db()

    env_lines = [ln.strip() for ln in args.env_freeze.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    env_manifest = dict(
        line.split("==", 1) for line in env_lines if "==" in line)
    env_manifest["_env_freeze_sha256"] = sha256_file(args.env_freeze)

    code_paths = [Path(p) for pattern in args.code_paths
                  for p in sorted(glob.glob(pattern, recursive=True))
                  if Path(p).is_file()]
    if not code_paths:
        raise ValueError(f"no files matched --code-paths {args.code_paths!r}")

    manifest = build_policy_manifest(
        thresholds=args.thresholds_json,
        semantic={
            "model_id": VLM_ID,
            "weight_sha256": _hash_weights(args.semantic_weights),
            "template": JUDGE_TEMPLATE,
            "decoding": {"max_new_tokens": 512},
        },
        reranker={
            "model_id": RERANKER_MODEL_ID,
            "weight_sha256": _hash_weights(args.reranker_weights),
            "prompt": RERANKER_PROMPT,
            "max_image_side": MAX_IMAGE_SIDE,
        },
        retrieval={
            "encoder": db.encoder,
            "encoder_weight_sha256": _hash_weights(args.encoder_weights),
            "index_sha256": sha256_file(args.index),
            "corpus": db.name,
            "query_builder": "ragregen.retrieve.reference_query",
            "k": 5,
            "ref_prep": "crop_to_mask+thumbnail448",
        },
        generator={
            "model_id": GENERATOR_MODEL_ID,
            "weight_sha256": _hash_weights(args.generator_weights),
        },
        env_manifest=env_manifest,
        code_snapshot_sha256=snapshot_code_sha256(code_paths),
        degenerate_behaviour=DEGENERATE_RULE,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"manifest": manifest, "policy_sha256": policy_sha256(manifest)},
        indent=2, sort_keys=True))
    print(f"wrote {args.out}: policy_sha256={policy_sha256(manifest)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
