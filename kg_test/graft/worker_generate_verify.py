#!/usr/bin/env python
"""Subprocess worker: one generate+verify attempt, then exit.

See worker_build_kg.py's module docstring for why this runs isolated in its
own process rather than in-process within refine()'s loop.

Usage:
    python -m graft.worker_generate_verify <kg_json> <cfg_yaml> <seed> \
        <out_image_path> <out_report_json>

On success, writes the generated image to <out_image_path> and a JSON
report ({"prompt", "attr_pass", "ok", "failing_parts", "part_scores"}) to
<out_report_json>, and exits 0. On failure, prints the traceback to stderr
and exits 1 (or 3 specifically for a CUDA OOM, so a caller can tell "no
signal produced, retry" apart from "unexpected error").
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from graft import env

env.setup()


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "usage: worker_generate_verify.py <kg_json> <cfg_yaml> <seed> "
            "<out_image_path> <out_report_json>",
            file=sys.stderr,
        )
        return 2
    kg_json, cfg_yaml, seed_s, out_image_path, out_report_path = argv
    seed = int(seed_s)

    from graft.config import GraftConfig
    from graft.generate import generate_lever_a
    from graft.models import Models
    from graft.schema import ConceptKG
    from graft.verify import verify

    cfg = GraftConfig.from_yaml(cfg_yaml)
    kg = ConceptKG.from_json(kg_json)
    models = Models(cfg)

    try:
        image, prompt = generate_lever_a(kg, models, cfg, seed=seed)
    except Exception as exc:  # noqa: BLE001 -- classify OOM for the caller
        if "out of memory" in str(exc).lower() or "OutOfMemoryError" in type(exc).__name__:
            print(f"OOM during generate: {exc}", file=sys.stderr)
            return 3
        raise

    report = verify(image, kg, models, cfg)

    image.save(out_image_path)
    report_dict = {
        "prompt": prompt,
        "attr_pass": report.attr_pass,
        "ok": report.ok,
        "failing_parts": report.failing_parts,
        "part_scores": [asdict(p) for p in report.part_scores],
    }
    with open(out_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
