from __future__ import annotations
from ragregen.mmkg.schema import target_attributes, n_targets, decidable
from ragregen.retrieved_reference import contamination
from ragregen.eval_manifest import sha256_file

def dedup_and_guard(paths, gt_ref_paths):
    """Drop sha256 duplicates and byte-identical ground-truth contamination.

    Also drops any path whose bytes can't be hashed (missing/unreadable
    file): the VLM read step downstream opens the image directly, so an
    unreadable file cannot survive the guard either way. Order is preserved.
    """
    seen, out = set(), []
    for p in paths:
        try:
            h = sha256_file(p)
        except Exception:
            continue
        if h in seen or contamination(p, gt_ref_paths):
            continue
        seen.add(h); out.append(p)
    return out

def feasibility(concept_reads):
    by = {c: n_targets(target_attributes(rs)) for c, rs in concept_reads.items()}
    undec = sorted([c for c, n in by.items() if n < 2])
    return {"decidable_concepts": sum(1 for n in by.values() if n >= 2),
            "undecidable": undec, "by_concept": by}

def build_concept(concept, query, index_slice_paths, gt_ref_paths, read_fn):
    kept = dedup_and_guard(index_slice_paths, gt_ref_paths)
    excluded = [p for p in index_slice_paths if p not in set(kept)]
    reads = [read_fn(p) for p in kept]
    tgt = target_attributes(reads)
    return {"concept": concept, "query": query, "built_from_slice": kept,
            "slice_size": len(kept), "contamination_excluded": excluded,
            "target_attributes": tgt, "n_target_attributes": n_targets(tgt),
            "decidable": decidable(tgt)}
