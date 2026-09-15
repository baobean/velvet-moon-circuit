#!/usr/bin/env python
"""One-off: build a real MMKG for a sample species and persist it under
outputs/ and tests/fixtures/, so later GPU tests (generate, verify,
pipeline) have a real ConceptKG with real ref_paths to load instead of
rebuilding it (which reloads the VLM/detector/embedders every time).

Ashok, not Akashmoni: hand-checking Treevill's per-species folders found
heavy duplication (see graft/dataset.py's module docstring) and Akashmoni's
2 unique images are both root/bark close-ups with no leaf or canopy visible
at all, so the VLM correctly reports "not visible" for nearly every
attribute. Ashok has 15 unique images including at least one canopy/leaf
shot, giving the schema something real to describe.
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft import env

env.setup()

from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.kg_build import build_kg
from graft.models import Models

ROOT = "data/treevill/rawdata2"
SPECIES_NAME = "Ashok"


def main():
    sp = load_species(ROOT, SPECIES_NAME)
    build_refs, heldout_refs = split_refs(sp, k_build=5, seed=0)

    cfg = GraftConfig()
    models = Models(cfg)
    kg = build_kg(SPECIES_NAME, build_refs, models, cfg)

    os.makedirs("tests/fixtures", exist_ok=True)
    shutil.copy(
        os.path.join(cfg.outputs_dir, SPECIES_NAME, "kg.json"),
        "tests/fixtures/sample_kg.json",
    )
    print("wrote", os.path.join(cfg.outputs_dir, SPECIES_NAME, "kg.json"))
    print("wrote tests/fixtures/sample_kg.json")
    print("build_refs:", build_refs)
    print("heldout_refs:", heldout_refs)


if __name__ == "__main__":
    main()
