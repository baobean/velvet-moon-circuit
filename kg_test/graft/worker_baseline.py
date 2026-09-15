#!/usr/bin/env python
"""Subprocess worker: produce one image for one method, then exit.

See refine.py's module docstring for why every GPU-touching phase runs in
its own subprocess. "ours" and "ours_notree" delegate to refine() (which
itself dispatches each attempt to a further-nested worker_generate_verify.py
subprocess -- fine, a subprocess spawning a subprocess is normal); b0/b1/b2
call the baseline functions directly against a fresh Models() that only this
process ever sees.

Usage:
    python -m graft.worker_baseline <method> <species> <cfg_yaml> \
        <kg_json_or_dash> <out_image_path> [<build_ref>...]

<kg_json_or_dash> is the concept's kg.json path for the GRAFT methods "ours"
and "ours_notree" (ignored, pass "-", for b0/b1/b2). <build_ref>... is
required for "b1" (its retrieval pool), ignored for the others.
"""
from __future__ import annotations

import sys
from dataclasses import replace

from graft import env


def cfg_for_method(cfg, method):
    return replace(cfg, use_part_tree=False) if method == "ours_notree" else cfg


def main(argv: list[str]) -> int:
    env.setup()
    if len(argv) < 5:
        print(
            "usage: worker_baseline.py <method> <species> <cfg_yaml> "
            "<kg_json_or_dash> <out_image_path> [<build_ref>...]",
            file=sys.stderr,
        )
        return 2
    method, species, cfg_yaml, kg_json_or_dash, out_image_path, *build_refs = argv

    from graft.config import GraftConfig

    cfg = cfg_for_method(GraftConfig.from_yaml(cfg_yaml), method)

    try:
        if method in ("ours", "ours_notree"):
            from graft.refine import refine
            from graft.schema import ConceptKG

            kg = ConceptKG.from_json(kg_json_or_dash)
            image, _report, _prompt = refine(kg, None, cfg)
        else:
            from graft.baselines import b0_vanilla, b1_imagerag, b2_ravel
            from graft.models import Models

            models = Models(cfg)
            if method == "b0":
                image = b0_vanilla(species, models, cfg, seed=0)
            elif method == "b1":
                image = b1_imagerag(species, build_refs, models, cfg, seed=0)
            elif method == "b2":
                image = b2_ravel(species, models, cfg, seed=0)
            else:
                print(f"unknown method '{method}'", file=sys.stderr)
                return 2
    except Exception as exc:  # noqa: BLE001 -- classify OOM for the caller
        if "out of memory" in str(exc).lower() or "OutOfMemoryError" in type(exc).__name__:
            print(f"OOM in worker_baseline ({method}): {exc}", file=sys.stderr)
            return 3
        raise

    image.save(out_image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
