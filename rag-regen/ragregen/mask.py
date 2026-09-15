"""Locate the region to repaint, and the object to copy from.

Two callers, one output: a float mask over the input image. The draft mask is
dilated and handed to the inpainter; the reference mask is consumed by
`kontext_engine.prepare_reference`, which derives its own bbox and crops. The
asymmetry the parent spec described lives downstream, not here
(specs/2026-07-28-masking-design.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    """One detector proposal, segmented and cropped.

    SAM runs on every candidate before selection, matching the harness that
    measured 0.917 (finegrained_seg.py:198-205): the mask and the choice then
    come from one candidate set rather than two passes that can disagree.
    """
    box: Box
    dino_conf: float
    mask: np.ndarray
    crop: object


@dataclass(frozen=True)
class MaskResult:
    """A mask over the input image, plus how it was chosen.

    `mask` is float in [0, 1] rather than bool so it satisfies both of
    `kontext_engine.prepare_reference`'s uses -- the `mask > 0.5` bbox
    derivation and the `clip(mask, 0, 1)` matte arithmetic -- with no second
    conversion at the call site.
    """
    mask: np.ndarray
    box: Box
    score: float
    n_candidates: int
    selected_by: str


def _prototype_scores(candidates, prototypes, phrase):
    """Scored candidates, or [] when the prototype rule cannot apply.

    A single candidate is not a choice, so the classifier is never called for
    one -- embedding it would be wasted work on a loaded encoder.
    """
    if prototypes is None or phrase is None or len(candidates) < 2:
        return []
    scored = [(prototypes.score(c.crop, phrase), c) for c in candidates]
    return [(s, c) for s, c in scored if s is not None]


def selection_rule(candidates: list[Candidate], prototypes,
                   phrase: str | None) -> str:
    """Which rule select_box will apply. Recorded per case, never inferred.

    Kept as the single source of truth so the trace cannot drift from the
    behaviour -- the mistake plan 4's finding named, where the persisted
    artifact did not say which box selection produced the number.
    """
    return "prototype" if _prototype_scores(candidates, prototypes, phrase) \
        else "confidence"


def select_box(candidates: list[Candidate], *, prototypes=None,
               phrase: str | None = None) -> Candidate:
    """Pick the candidate to segment.

    Highest detector confidence by default -- phrase-neutral, and the rule
    Stream A already grades against. With a classifier and more than one
    candidate, nearest-prototype instead.

    `phrase` here is the phrase to SCORE, which is not necessarily the phrase
    that was grounded. Grounding "dog" on boston_bull returns the head and the
    whole dog, and scoring dog-ness cannot separate them -- both crops are dog.
    "Which is the Boston bull" can.
    """
    usable = _prototype_scores(candidates, prototypes, phrase)
    if usable:
        return max(usable, key=lambda p: p[0])[1]
    return max(candidates, key=lambda c: c.dino_conf)


class Masker:
    """DINO proposes, SAM segments, `select_box` chooses.

    Both models are injected. Nothing loads at import, so the suite runs on
    CPU against fakes and a contended GPU costs one retried stage rather than
    the whole run (design §8).
    """

    def __init__(self, detector, sam, max_boxes: int = 8):
        self.detector = detector
        self.sam = sam
        self.max_boxes = max_boxes

    def segment(self, image, phrase: str, *, prototypes=None,
                score_phrase: str | None = None) -> MaskResult | None:
        """Locate `phrase` in `image`. None when it is not grounded.

        None, never an empty mask: an all-zeros mask would silently repaint
        nothing and read downstream as a successful attempt. Not grounded means
        "do not repair this case" -- the verifier's ABSTAIN, not its MISSING.
        """
        proposals = self.detector.all_boxes(image, phrase,
                                            max_boxes=self.max_boxes)
        if not proposals:
            return None

        candidates = [
            Candidate(box=box, dino_conf=conf,
                      mask=self.sam.mask_from_box(image, box),
                      crop=image.crop((int(box[0]), int(box[1]),
                                       int(box[2]), int(box[3]))))
            for box, conf in proposals
        ]

        scored_as = score_phrase or phrase
        winner = select_box(candidates, prototypes=prototypes,
                            phrase=scored_as)
        arr = np.asarray(winner.mask, dtype=np.float64)
        if not arr.any():
            return None

        return MaskResult(mask=arr, box=winner.box, score=winner.dino_conf,
                          n_candidates=len(candidates),
                          selected_by=selection_rule(candidates, prototypes,
                                                     scored_as))

    def to_cpu(self) -> None:
        for m in (self.detector, self.sam):
            if hasattr(m, "to_cpu"):
                m.to_cpu()

    def free(self) -> None:
        """Drop both models and return their VRAM.

        SAM-ViT-H is ~2.4 GB of weights and allocates roughly another 1 GB of
        windowed-attention activations at 1024 px. Held alongside FLUX on a
        card another researcher is already using, that last GB is what OOMs --
        ../rag-edit died exactly this way at ragedit/mask.py:49. Masking for
        every case finishes before any inpainting starts, so callers
        precompute masks, then free().
        """
        for m in (self.detector, self.sam):
            if hasattr(m, "free"):
                m.free()
        self.detector = None
        self.sam = None


def _dilate(arr: np.ndarray, px: int) -> np.ndarray:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    grown = cv2.dilate((arr * 255).astype(np.uint8), k, iterations=1)
    return (grown > 127).astype(np.float64)


def mask_draft(image, coarse: str, masker: Masker, *, prototypes=None,
               score_phrase: str | None = None,
               dilate_px: int = 12) -> MaskResult | None:
    """The region of the draft to destroy.

    Grounds the COARSE term. The draft is by definition the image that got the
    fine concept wrong, so the fine phrase may ground poorly or not at all, and
    how well it grounds would vary with how wrong the draft is -- making mask
    geometry a function of draft quality. The coarse term is present whether
    the draft succeeded or failed (design §4).

    `score_phrase` is the concept, not the coarse term. Grounding "dog" on
    boston_bull returns the head and the whole dog; only "which is the Boston
    bull" separates them. It is also the prototype that exists -- gt_refs cover
    every concept, while coarse prototypes need coarse_refs, which no dataset
    carries yet.
    """
    found = masker.segment(image, coarse, prototypes=prototypes,
                           score_phrase=score_phrase)
    if found is None or dilate_px <= 0:
        return found
    return replace(found, mask=_dilate(found.mask, dilate_px))


def mask_reference(image, concept: str, masker: Masker, *,
                   prototypes=None) -> MaskResult | None:
    """The object in a reference photograph to copy from.

    Grounds the FINE concept: a reference is by construction a photograph of
    the real thing, so the fine phrase is the accurate one, and clutter -- a
    second animal, a handler, an enclosure -- is what the classifier is for.

    Never dilates. This mask feeds `kontext_engine.prepare_reference`, which
    crops to its bbox; growing it would matte background into the cutout.
    """
    return masker.segment(image, concept, prototypes=prototypes)


def crop_to_object(image, phrase: str, masker: Masker, *, prototypes=None,
                   pad_frac: float = 0.08):
    """Crop `image` to the object `phrase` names. None when not grounded.

    Prototypes built from whole reference photographs encode framing,
    background and composition -- "a photograph of a durian" rather than "a
    durian". Crops that look like photographs then outscore tight object
    crops whatever they contain, which is a size preference wearing identity's
    clothes. Cropping each reference to its object first makes both sides of
    the comparison frame alike.

    Padding mirrors kontext_engine.prepare_reference (pad_frac of the box plus
    a 4px floor) so a prototype reference and a prepared reference are framed
    the same way.
    """
    found = masker.segment(image, phrase, prototypes=prototypes)
    if found is None:
        return None

    ys, xs = np.where(found.mask > 0.5)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pw = int((x1 - x0) * pad_frac) + 4
    ph = int((y1 - y0) * pad_frac) + 4
    return image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(image.width, x1 + pw + 1),
                       min(image.height, y1 + ph + 1)))
