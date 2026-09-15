"""Three offline arms, assembled with one frozen flagged-fallback rule.

Design section 6: every arm always yields exactly one output per case --
``attempt_1`` on success, or the draft as an explicitly flagged operational
fallback (``generation_failed=true``) on failure. No arm's denominator ever
shrinks. For the Phase B *study* specifically (not the deployable behaviour),
any post-freeze failure on a *routed* case makes the whole study
INCONCLUSIVE rather than computing end-to-end metrics over a
fallback-contaminated set.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping


class PhaseBInconclusive(Exception):
    """A routed case's attempt_1 failed after the holdout was frozen."""


def _out(path: str, failed: bool) -> dict:
    return {"path": path, "generation_failed": failed}


def assemble_arms(cases: Sequence[Mapping]) -> dict[str, dict]:
    """Arm rule (design section 6):

    - arm1 (draft): the original draft, always.
    - arm2 (route-all): every draft replaced by attempt_1, or the draft
      flagged ``generation_failed=true`` if attempt_1 failed.
    - arm3 (frozen-detector): attempt_1 where the frozen rule routed and it
      succeeded; the draft otherwise. An un-routed case's draft is an
      ordinary preserved draft, never flagged.
    """
    result: dict[str, dict] = {}
    for c in cases:
        if c["routed"] and not c["attempt_ok"]:
            raise PhaseBInconclusive(f"{c['case_id']}: routed attempt failed post-freeze")
        draft, attempt, ok = c["draft_path"], c["attempt_path"], c["attempt_ok"]
        arm2 = _out(attempt, False) if ok else _out(draft, True)
        arm3 = _out(attempt, False) if (c["routed"] and ok) else _out(draft, False)
        result[c["case_id"]] = {
            "cohort": c["cohort"],
            "arm1": _out(draft, False),
            "arm2": arm2,
            "arm3": arm3,
        }
    return result
