"""The one way a ConceptScore becomes JSON, and comes back.

Two callers persisted Stream A independently and discarded opposite halves:
score_a.py kept every similarity and no verdict, run_pipeline.py kept the
verdict and no similarity. The pipeline's half is why 202 verdicts of GPU time
cannot be re-graded at any other tau. One serialiser, two callers, no drift.
"""
from __future__ import annotations

from ragregen.verify.grounded import STATES, CandidateScore, ConceptScore


def _cand_to_dict(c: CandidateScore) -> dict:
    return {"box": list(c.box), "dino_conf": c.dino_conf, "sim": c.sim,
            "sim_coarse": c.sim_coarse, "sim_proto": c.sim_proto,
            "sim_proto_coarse": c.sim_proto_coarse}


def _cand_from_dict(d: dict) -> CandidateScore:
    return CandidateScore(box=tuple(d["box"]), dino_conf=d["dino_conf"],
                          sim=d["sim"], sim_coarse=d.get("sim_coarse"),
                          sim_proto=d.get("sim_proto"),
                          sim_proto_coarse=d.get("sim_proto_coarse"))


def to_dict(score: ConceptScore, *, with_state: bool) -> dict:
    """Serialise one concept.

    `with_state=False` is score_a.py's contract: its tau is arbitrary, so a
    verdict in that file would be a number nobody should trust. The pipeline
    passes True because the scheduler resumes on verdicts.

    Key order matches the pre-refactor score_a.py comprehension exactly
    (phrase first, then sim..kind), because `_write` dumps with `indent=2`
    and no `sort_keys` -- order is part of stream_a.json's byte contract,
    not an implementation detail.
    """
    out = {
        "phrase": score.phrase,
        "sim": score.sim,
        "sim_coarse": score.sim_coarse,
        "sim_proto": score.sim_proto,
        "sim_proto_coarse": score.sim_proto_coarse,
        "candidates": [_cand_to_dict(c) for c in score.candidates],
        "box": list(score.box) if score.box is not None else None,
        "dino_conf": score.dino_conf,
        "kind": score.kind,
    }
    if with_state:
        out["state"] = score.state
    return out


def from_dict(d: dict | str) -> ConceptScore:
    """Rehydrate. A bare string is an old streams.json entry, not an error.

    Runs produced before 2026-08-12 persisted `{phrase: "MISSING"}` and nothing
    else. Those directories must stay resumable, so a string rehydrates to the
    lossy score it honestly is.

    A dict with no `state` key defaults to ABSTAIN, not PRESENT: the one file
    shape that legitimately omits `state` is a current stream_a.json --
    score_a.py always calls to_dict(with_state=False), see its own docstring
    -- and fabricating a pass when the file does not say is the same class of
    error this branch exists to fix (spec: verify/grounded.py's ABSTAIN
    docstring -- "the state that means no verdict was reached"). The pipeline
    itself never reaches this default: _stash_grounded always passes
    with_state=True.
    """
    if isinstance(d, str):
        if d not in STATES:
            raise ValueError(f"{d!r} is not a verifier state")
        return ConceptScore("", "subject", d)

    state = d.get("state", "ABSTAIN")
    if state not in STATES:
        raise ValueError(f"{state!r} is not a verifier state")
    box = d.get("box")
    return ConceptScore(
        d.get("phrase", ""), d.get("kind", "subject"), state,
        box=tuple(box) if box is not None else None,
        dino_conf=d.get("dino_conf"), sim=d.get("sim"),
        sim_coarse=d.get("sim_coarse"), sim_proto=d.get("sim_proto"),
        sim_proto_coarse=d.get("sim_proto_coarse"),
        candidates=tuple(_cand_from_dict(c) for c in d.get("candidates", ())),
    )
