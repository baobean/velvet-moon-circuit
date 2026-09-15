"""Immutable identity/eligibility manifest for verifier-v2 development.

The protocol correction this module enforces (design section 3): identity truth
and pipeline eligibility are *separate* immutable facts. An invalid mask never
rewrites ``identity_truth``; it only sets ``selector_eligible=False`` with a
mechanical ``eligibility_reason``. All identity-labelled cases stay in the
detector population; only selector/end-to-end metrics filter on eligibility.

The builder is pure and deterministic: it hashes source files, sorts every
container before serialisation, and embeds no wall-clock time. Provenance that
is genuinely time- or process-dependent (git status, timestamps) belongs to the
CLI wrapper, not here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

#: The only identity verdicts a draft may carry. ``EXCLUDE`` is deliberately
#: absent: it conflates truth with eligibility, which is exactly the mistake the
#: v1 label mutation made.
IDENTITY_TRUTHS = ("PASS", "FAIL", "UNJUDGEABLE")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large images stay off-heap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_entry(path: Path) -> dict | None:
    """A ``{path, sha256}`` record, or ``None`` when the file is absent.

    Absence is recorded rather than raised: a mask-failed case legitimately has
    no ``mask.png`` or attempts, and the eligibility reason already explains it.
    """
    path = Path(path)
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def _eligibility(state: Mapping, case_dir: Path) -> tuple[bool, str]:
    """Mechanical eligibility from queue status plus on-disk artifacts.

    Order matters: a failed mask stage is the *reason* even if a stray file
    exists, and a corrupt artifact (stage done but mask gone) is distinct from a
    case that simply produced no attempts.
    """
    stages = state.get("stages", {}) if isinstance(state, Mapping) else {}
    if stages.get("mask") == "failed":
        return False, "mask_not_grounded"
    if not (case_dir / "mask.png").is_file():
        return False, "corrupt_artifact"
    if not sorted(case_dir.glob("attempt_*.png")):
        return False, "no_prepared_reference"
    return True, "eligible"


def build_manifest(dataset, identity_rows: Mapping[str, str],
                   screen_run: Path, candidate_run: Path, *,
                   protocol_status: str,
                   source_paths: Mapping[str, Path]) -> dict:
    """One immutable row per dataset case.

    ``dataset`` is a loaded :class:`ragregen.config.DatasetConfig`.
    ``identity_rows`` maps every case id to one of :data:`IDENTITY_TRUTHS`.
    ``screen_run`` holds ``<case>/draft.png``; ``candidate_run`` holds
    ``queue.json`` and ``<case>/`` artifacts. ``source_paths`` are shared files
    (dataset, identity source, semantic/reranker/DINO JSON, queue) to hash for
    provenance.
    """
    if protocol_status not in ("frozen_before_score",
                               "reconstructed_after_score"):
        raise ValueError(f"invalid protocol_status {protocol_status!r}")

    screen_run = Path(screen_run)
    candidate_run = Path(candidate_run)

    dataset_ids = {case.id for case in dataset.cases}
    identity_ids = set(identity_rows)
    if dataset_ids != identity_ids:
        missing = dataset_ids - identity_ids
        extra = identity_ids - dataset_ids
        raise ValueError(
            f"identity ids must match dataset exactly; "
            f"missing={sorted(missing)} extra={sorted(extra)}")

    import json
    queue_path = candidate_run / "queue.json"
    queue_cases: dict = {}
    if queue_path.is_file():
        queue_cases = json.loads(queue_path.read_text()).get("cases", {})

    cases: dict[str, dict] = {}
    for case in sorted(dataset.cases, key=lambda c: c.id):
        cid = case.id
        truth = str(identity_rows[cid]).strip().upper()
        if truth not in IDENTITY_TRUTHS:
            raise ValueError(
                f"{cid}: identity_truth {identity_rows[cid]!r} not one of "
                f"{IDENTITY_TRUTHS}")

        case_dir = candidate_run / cid
        eligible, reason = _eligibility(queue_cases.get(cid, {}), case_dir)

        attempts = {
            path.stem: _hash_entry(path)
            for path in sorted(case_dir.glob("attempt_*.png"))
        }
        sources = {
            "draft": _hash_entry(screen_run / cid / "draft.png"),
            "mask": _hash_entry(case_dir / "mask.png"),
            "attempts": attempts,
        }
        cases[cid] = {
            "identity_truth": truth,
            "cohort": case.cohort,
            "kind": case.kind,
            "concept": case.concept,
            "selector_eligible": eligible,
            "eligibility_reason": reason,
            "sources": sources,
        }

    detector_rows = sum(1 for row in cases.values()
                        if row["identity_truth"] in ("PASS", "FAIL"))
    eligible_rows = sum(1 for row in cases.values()
                        if row["selector_eligible"])
    counts = {
        "detector_rows": detector_rows,
        "selector_eligible": eligible_rows,
        "excluded": len(cases) - eligible_rows,
    }

    sources = {
        name: _hash_entry(Path(path))
        for name, path in sorted(source_paths.items())
    }

    return {
        "schema": 1,
        "protocol_status": protocol_status,
        "dataset": dataset.name,
        "counts": counts,
        "sources": sources,
        "cases": cases,
    }
