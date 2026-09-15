#!/usr/bin/env python
"""Build the retrieval index from the corpus in configs/retrieval_db.yaml.

Usage: ./scripts/run.sh build-index
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, retrieve, trace, validate  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    db = config.load_retrieval_db(args.db)

    problems = validate.validate_disk()
    for p in problems:
        print(f"[{p.severity}] {p.code}: {p.message}")
    if any(p.severity == "error" for p in problems):
        return 2

    paths = sorted(p for p in db.images_root.rglob("*")
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"[corpus] {len(paths)} images under {db.images_root}")

    run = trace.open_run("build_index", argv=sys.argv, args=vars(args))

    from ragregen.encoders import build_encoder
    encoder = build_encoder(db.encoder, device=args.device)

    n = retrieve.build_index(paths, encoder, db.index_path,
                             batch_size=args.batch_size)
    print(f"[index] {n} vectors, dim={encoder.dim} -> {db.index_path}")
    run.finish("ok", {"n_indexed": n, "dim": encoder.dim})
    env.reclaim_gpu()
    return 0


if __name__ == "__main__":
    # Not SystemExit: unloading torch+faiss+PIL together segfaults in this
    # env, which scores a finished stage as a failure. See env.exit_now.
    env.exit_now(main())
