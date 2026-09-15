"""Phase B holdout freeze: fixed 24+24 denominators, backfill at freeze time.

Design section 3: the final frozen cohort is exactly 24 judgeable rare FAIL
cases and 24 judgeable control PASS cases. A case is never simply "dropped"
in a way that reduces either denominator -- an ineligible or contaminated
case is backfilled from the replacement reserve *at freeze time*, and the
substitution is recorded. If the reserve is exhausted before reaching 24
judgeable cases, the cohort (and thus the whole holdout) is INCONCLUSIVE
rather than proceeding with a reduced denominator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ragregen.eval_manifest import sha256_file

_WANT_TRUTH = {"rare": "FAIL", "control": "PASS"}


def select_cohort(candidates: Sequence[Mapping], *, cohort: str, needed: int = 24) -> dict:
    """Pick the first ``needed`` judgeable+eligible+uncontaminated cases.

    ``candidates`` is an ordered list of
    ``{case_id, identity_truth, eligible, contaminated}``. Any judgeable case
    that is skipped (ineligible or contaminated) is recorded as a backfill
    event. Returns ``status="inconclusive"`` -- never a shrunk cohort -- if
    fewer than ``needed`` judgeable+eligible+uncontaminated cases exist.
    """
    want = _WANT_TRUTH[cohort]
    chosen: list[str] = []
    backfilled: list[str] = []
    for c in candidates:
        if len(chosen) == needed:
            break
        judgeable = c["identity_truth"] == want
        ok = judgeable and c["eligible"] and not c["contaminated"]
        if ok:
            chosen.append(c["case_id"])
        elif judgeable:            # a would-be case skipped -> backfill event
            backfilled.append(c["case_id"])
    status = "complete" if len(chosen) == needed else "inconclusive"
    return {"cases": chosen, "backfilled": backfilled, "status": status}


def build_holdout_manifest(rare_selection: Mapping, control_selection: Mapping, *,
                           dino_reserves: Mapping, references: Mapping,
                           sources: Mapping[str, Path]) -> dict:
    """Immutable 24+24 holdout manifest. Raises if either cohort is short.

    ``dino_reserves`` and ``references`` are kept as two distinct maps (design
    section 3): the DINO reference reserve is never used for editing or
    routing, and the reference map is the deterministically retrieved top-1
    per case. ``sources`` are hashed for provenance.
    """
    for name, selection in (("rare", rare_selection), ("control", control_selection)):
        if selection["status"] != "complete" or len(selection["cases"]) != 24:
            raise ValueError(
                f"{name} cohort is not a complete 24-case selection "
                f"(status={selection['status']!r}, n={len(selection['cases'])})")

    hashed_sources = {
        name: sha256_file(Path(path)) for name, path in sorted(sources.items())
    }

    return {
        "schema": 1,
        "protocol_status": "frozen_before_score",
        "cohorts": {
            "rare": {"cases": list(rare_selection["cases"]),
                     "backfilled": list(rare_selection["backfilled"])},
            "control": {"cases": list(control_selection["cases"]),
                        "backfilled": list(control_selection["backfilled"])},
        },
        "dino_reserves": dict(dino_reserves),
        "references": dict(references),
        "sources": hashed_sources,
    }
