"""Combine the grounded and semantic streams into one verdict.

Pure function, no models. The `agreement` field is not bookkeeping: the rate at
which the two streams disagree is the evidence that decomposing ImageRAG's
single GPT-4o judge buys something (spec §2, claim C1).

Failure is decided on `ConceptScore.state` alone, never on whether a box is
present -- see FAILING_STATES. A scorer that violates its contract can return
MISSING or FINE_MISMATCH with `box=None`, which is indistinguishable from
ABSTAIN if this module pattern-matches on the box (Task 6 carry-forward).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragregen.verify.grounded import ConceptScore
from ragregen.verify.semantic import SemanticVerdict

AGREEMENTS = ("BOTH_PASS", "BOTH_FAIL", "GROUNDED_ONLY", "SEMANTIC_ONLY")

#: Grounded states that fail a case. FINE_MISMATCH joins MISSING here rather
#: than being folded into it so the C1 report can attribute each catch to a
#: mechanism (spec §2).
FAILING_STATES = ("MISSING", "FINE_MISMATCH")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    target: str | None
    score: float | None
    agreement: str
    evidence: dict = field(default_factory=dict)


def fuse(grounded_scores: dict[str, ConceptScore],
         semantic: SemanticVerdict) -> Verdict:
    failing = [s for s in grounded_scores.values() if s.state in FAILING_STATES]
    grounded_failed = bool(failing)
    semantic_failed = not semantic.ok

    if grounded_failed and semantic_failed:
        agreement = "BOTH_FAIL"
    elif grounded_failed:
        agreement = "GROUNDED_ONLY"
    elif semantic_failed:
        agreement = "SEMANTIC_ONLY"
    else:
        agreement = "BOTH_PASS"

    ok = not (grounded_failed or semantic_failed)

    target = None
    if grounded_failed:
        target = min(failing, key=lambda s: s.sim if s.sim is not None else 0.0).phrase
    elif semantic_failed and semantic.issues:
        target = semantic.issues[0].concept

    #: min over the concepts Stream A could actually see. None -- not 1.0 --
    #: when it saw none: a fabricated perfect score outranks every real one in
    #: select_best, which is how three failing attempts were promoted over
    #: their drafts in outputs/doc5_20260810_183939.
    sims = [s.sim for s in grounded_scores.values()
            if s.state != "ABSTAIN" and s.sim is not None]
    score = float(min(sims)) if sims else None

    return Verdict(
        ok=ok, target=target, score=score, agreement=agreement,
        evidence={
            "grounded": {p: {"state": s.state, "sim": s.sim,
                             "sim_coarse": s.sim_coarse, "kind": s.kind}
                         for p, s in grounded_scores.items()},
            "semantic": {"ok": semantic.ok, "degenerate": semantic.degenerate,
                         "issues": [{"concept": i.concept, "problem": i.problem}
                                    for i in semantic.issues]},
        },
    )
