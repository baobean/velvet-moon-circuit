#!/usr/bin/env python
"""Phase 1b of the fast driver: recall b2 (RAVEL-style) attributes for every
species with the VLM resident, so the SDXL generation phase never needs the
VLM. b2 attrs depend only on the concept name. One VLM load for all species.

Usage: python -m graft.worker_recall_b2 <cfg_yaml> <out_json> <species>...
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

from graft import env

env.setup()


def recall_all(species_names, describe_fn) -> Dict[str, List[str]]:
    return {name: describe_fn(name) for name in species_names}


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: worker_recall_b2.py <cfg_yaml> <out_json> <species>...", file=sys.stderr)
        return 2
    cfg_yaml, out_json, *species = argv

    from PIL import Image

    from graft.baselines import RAVEL_ATTR_INSTRUCTION, _parse_attr_list
    from graft.config import GraftConfig
    from graft.models import Models

    cfg = GraftConfig.from_yaml(cfg_yaml)
    models = Models(cfg)
    blank = Image.new("RGB", (16, 16), (128, 128, 128))

    def describe(concept: str):
        raw = models.vlm.describe([blank], RAVEL_ATTR_INSTRUCTION.format(concept=concept))
        return _parse_attr_list(raw)

    out = recall_all(species, describe)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
