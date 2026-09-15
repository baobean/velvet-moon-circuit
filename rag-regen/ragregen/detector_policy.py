"""Complete, hashable frozen policy manifest for the Phase B holdout.

Phase A's ``frozen_detector_policy.json`` recorded only the two thresholds;
that is insufficient to reproduce a scoring run byte-for-byte. This module
builds the *complete* frozen policy manifest -- rule text, every model's
weight-file hash, prompts, the full environment manifest, retrieval/index
hashes, and a clean code-snapshot hash (not ``git HEAD``, since the worktree
is dirty) -- and hashes the whole thing with one order-independent SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ragregen.eval_manifest import sha256_file

RULE = ("route = semantic_failure OR (draft_text_relevance < 0.5 AND "
        "draft_retrieved_reference_relevance < 0.25048828125); "
        "routed -> attempt_1 on success else draft(generation_failed=true); "
        "not routed -> draft")

#: Frozen before any score is visible (spec sec 2): a semantic parse
#: failure fails closed (counts as semantic_failure, routes); a missing or
#: non-finite text/retrieved-reference score is a coverage failure that
#: makes the study INCONCLUSIVE rather than being silently defaulted.
DEGENERATE_RULE = {
    "semantic_parse_failure": "fail_closed_route",
    "missing_or_nonfinite_score": "coverage_failure_inconclusive",
}


def snapshot_code_sha256(paths: list[Path]) -> str:
    """Deterministic tree hash over the exact executed source files."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode())
        h.update(b"\0")
        h.update(Path(p).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_policy_manifest(*, thresholds, semantic, reranker, retrieval, generator,
                          env_manifest, code_snapshot_sha256, degenerate_behaviour) -> dict:
    return {
        "schema": 1,
        "protocol_status": "frozen_before_score",
        "rule": RULE,
        "thresholds": thresholds,
        "semantic": semantic,
        "reranker": reranker,
        "retrieval": retrieval,
        "generator": generator,
        "env_manifest": env_manifest,
        "code_snapshot_sha256": code_snapshot_sha256,
        "degenerate_behaviour": degenerate_behaviour,
    }


def policy_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
