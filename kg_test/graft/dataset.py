"""Treevill dataset loader: each immediate subdirectory of the root is one
species, holding its raw reference images.

Treevill's per-species folders are heavily duplicated: hand-checked species
each have only 3-15 distinct images out of 2000 files (e.g. Akashmoni: 2
unique images x ~1000 copies each; Aloe Wood: 3 unique / 2000; Ashok: 15 /
2000). Every loader here therefore dedupes a species' images by content
hash and keeps one path per unique image -- without this, split_refs'
build/heldout split could put byte-identical images on both sides, making
concept-fidelity scoring trivially perfect rather than a real held-out test.

Hashing is done per-species, not eagerly over the whole root: this NAS is
slow enough (measured: a full 132k-file scan across all 66 species stalled
past 9 minutes with 0% GPU use, pure I/O wait) that hashing every species up
front is impractical when a caller only needs one or a handful. list_species
lists species names via a cheap directory scan (no hashing); load_species
hashes only the one requested directory; load_treevill composes both for
callers that really do want the full, deduped corpus.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Species:
    name: str
    images: List[str]


def _content_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def list_species(root: str) -> List[str]:
    """Cheap: species (subdirectory) names under `root`, no file hashing."""
    return sorted(
        entry.name for entry in os.scandir(root) if entry.is_dir()
    )


def load_species(root: str, name: str) -> Species:
    """Load and dedupe one species' images by content hash."""
    species_dir = os.path.join(root, name)
    candidates = sorted(
        os.path.join(species_dir, f)
        for f in os.listdir(species_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    seen_hashes = set()
    images = []
    for path in candidates:
        h = _content_hash(path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        images.append(path)
    return Species(name=name, images=images)


def load_treevill(root: str) -> List[Species]:
    """Full, deduped corpus: every species under `root`, each hashed. Slow
    on a large or slow-I/O root (see module docstring) -- prefer
    list_species + load_species when only a handful of species are needed."""
    species = []
    for name in list_species(root):
        sp = load_species(root, name)
        if sp.images:
            species.append(sp)
    return species


def split_refs(sp: Species, k_build: int, seed: int, n_heldout_min: int = 1) -> Tuple[List[str], List[str]]:
    """Adaptive split into disjoint (build_refs, heldout_refs). `k_build` is a
    CAP: effective build = min(k_build, n_unique - n_heldout_min), always
    reserving n_heldout_min held-out images. A 1-image species yields an empty
    build (the eval driver then skips it)."""
    rng = random.Random(seed)
    shuffled = list(sp.images)
    rng.shuffle(shuffled)
    n = len(shuffled)
    build_n = min(k_build, max(0, n - n_heldout_min))
    return shuffled[:build_n], shuffled[build_n:]
