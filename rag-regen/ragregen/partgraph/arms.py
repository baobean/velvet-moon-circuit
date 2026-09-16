"""The reference source per arm — single interface, no silent fallback."""
from __future__ import annotations

from ragregen.mmkg_store.reference_source import medoid_reference
from ragregen.partgraph.compose import compose_reference
from ragregen.partgraph.select import select_part_images

ARMS = ("single_medoid", "partgraph")


def reference_for(arm, store, global_id, *, tile: int = 384):
    if arm == "single_medoid":
        ref = medoid_reference(store, global_id)
        if ref is None:
            raise LookupError(f"no medoid for {global_id!r}")
        return ref
    if arm == "partgraph":
        parts = select_part_images(store, global_id)
        if not parts:
            raise LookupError(f"no part crops for {global_id!r}")
        return compose_reference(parts, tile=tile)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
