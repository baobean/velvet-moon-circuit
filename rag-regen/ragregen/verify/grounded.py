"""Stream A: GroundingDINO proposes boxes, a region-text model scores the crops.

Produces a continuous per-concept score, which is what makes the verifier
calibratable -- the property a single VLM verdict cannot offer (spec §2).

ABSTAIN is not MISSING. When the detector returns no box, that means either the
concept is absent OR the detector failed, and those are not the same claim.
Reporting MISSING here would make every unboxable prompt a false failure
(spec §9 R3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragregen.concepts import Concept

STATES = ("PRESENT", "MISSING", "FINE_MISMATCH", "ABSTAIN")

Box = tuple[float, float, float, float]

#: Kinds Stream A structurally cannot check -- no box expresses a quantity or
#: an arrangement. Fusion routes these to Stream B.
UNBOXABLE_KINDS = ("count", "relation")


class Detector(Protocol):
    def all_boxes(self, image, phrase: str, max_boxes: int = 8
                  ) -> list[tuple[Box, float]]: ...


class CropScorer(Protocol):
    def score(self, crop, phrase: str) -> float: ...


@dataclass(frozen=True)
class CandidateScore:
    """One detector proposal, scored every way we know how.

    The previous finding could not size the box-selection asymmetry because
    `stream_a.json` kept only the winning box's score. Persisting all of them
    makes the selection rule an offline choice that can be varied and reported.
    """
    box: Box
    dino_conf: float
    sim: float
    sim_coarse: float | None = None
    sim_proto: float | None = None
    sim_proto_coarse: float | None = None


@dataclass(frozen=True)
class ConceptScore:
    phrase: str
    kind: str
    state: str
    box: Box | None = None
    dino_conf: float | None = None
    sim: float | None = None
    #: Similarity of the same crop to the concept's superordinate term. None
    #: when the concept carries no coarse term or when no box was found.
    sim_coarse: float | None = None
    sim_proto: float | None = None
    sim_proto_coarse: float | None = None
    candidates: tuple[CandidateScore, ...] = ()


class GroundedVerifier:
    """Scores each concept by detecting it and scoring the best crop.

    Args:
        detector: proposes candidate boxes for a phrase.
        scorer:   scores a crop against a phrase, in [0, 1].
        tau:      crops scoring below this are MISSING.
        delta:    a crop whose similarity to the concept exceeds its
                  similarity to the concept's coarse term by less than this
                  is FINE_MISMATCH. Only applies to concepts carrying a
                  coarse term. Defaults to 0.0.
    """

    def __init__(self, detector: Detector, scorer: CropScorer, tau: float,
                 delta: float = 0.0, prototypes=None, embedder=None,
                 prototype_delta: float | None = None):
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if not -1.0 <= delta <= 1.0:
            raise ValueError(f"delta must be in [-1, 1], got {delta}")
        if prototype_delta is not None and not -2.0 <= prototype_delta <= 2.0:
            raise ValueError(
                f"prototype_delta must be in [-2, 2], got {prototype_delta}")
        self.detector = detector
        self.scorer = scorer
        self.tau = tau
        self.delta = delta
        # Recorded, never acted on in-process: the live state logic must stay
        # byte-for-byte what the pinned tau = 0.25 baseline was graded against.
        self.prototypes = prototypes
        self.embedder = embedder
        self.prototype_delta = prototype_delta

    def score(self, image, concepts: list[Concept]) -> dict[str, ConceptScore]:
        out: dict[str, ConceptScore] = {}
        for concept in concepts:
            out[concept.phrase] = self._score_one(image, concept)
        return out

    def _score_one(self, image, concept: Concept) -> ConceptScore:
        if concept.kind in UNBOXABLE_KINDS:
            return ConceptScore(concept.phrase, concept.kind, "ABSTAIN")

        candidates = self.detector.all_boxes(image, concept.phrase, max_boxes=8)
        if not candidates:
            return ConceptScore(concept.phrase, concept.kind, "ABSTAIN")

        crops = [image.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                 for b, _ in candidates]

        proto_f: list[float | None] = [None] * len(candidates)
        proto_c: list[float | None] = [None] * len(candidates)
        if self.prototypes is not None and self.embedder is not None:
            vecs = self.embedder.encode_pil(crops)
            for i, vec in enumerate(vecs):
                proto_f[i] = self.prototypes.score(vec, concept.phrase)
                if concept.coarse:
                    proto_c[i] = self.prototypes.score(vec, concept.coarse)

        scored = []
        for i, (box, conf) in enumerate(candidates):
            sim = float(self.scorer.score(crops[i], concept.phrase))
            coarse = (float(self.scorer.score(crops[i], concept.coarse))
                      if concept.coarse else None)
            scored.append(CandidateScore(box=box, dino_conf=conf, sim=sim,
                                         sim_coarse=coarse,
                                         sim_proto=proto_f[i],
                                         sim_proto_coarse=proto_c[i]))

        # Unchanged selection: argmax of the fine-phrase similarity, first one
        # winning ties. Changing this would move `sim`, hence MISSING at
        # tau = 0.25, hence the pre-registered baseline.
        best = max(scored, key=lambda c: c.sim)
        best_box, best_conf, best_sim = best.box, best.dino_conf, best.sim
        sim_coarse = best.sim_coarse

        # Prototype contrast is evaluated at the detector-confidence box, not
        # at the box selected by fine-text similarity. Selecting with the same
        # fine signal we later grade would bias the comparison in its favour.
        proto_candidate = max(scored, key=lambda c: c.dino_conf)
        proto_margin = (
            None if (proto_candidate.sim_proto is None or
                     proto_candidate.sim_proto_coarse is None)
            else proto_candidate.sim_proto - proto_candidate.sim_proto_coarse)

        if best_sim < self.tau:
            state = "MISSING"
        elif (self.prototype_delta is not None and proto_margin is not None
              and round(proto_margin, 9) < self.prototype_delta):
            state = "FINE_MISMATCH"
        # round(): binary floats make e.g. 0.60 - 0.50 land 2e-17 below 0.10
        # rather than exactly on it, which would misfire the margin < delta
        # boundary. Similarity scores don't carry meaningful precision past
        # 9 decimal places, so this cannot mask any real margin.
        elif sim_coarse is not None and round(best_sim - sim_coarse, 9) < self.delta:
            state = "FINE_MISMATCH"
        else:
            state = "PRESENT"

        return ConceptScore(concept.phrase, concept.kind, state,
                            box=best_box, dino_conf=best_conf, sim=best_sim,
                            sim_coarse=sim_coarse,
                            sim_proto=best.sim_proto,
                            sim_proto_coarse=best.sim_proto_coarse,
                            candidates=tuple(scored))
