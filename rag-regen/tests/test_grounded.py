# tests/test_grounded.py
import pytest
from PIL import Image

from ragregen.concepts import Concept
from ragregen.verify import grounded


class FakeDetector:
    """Returns pre-programmed boxes per phrase."""

    def __init__(self, boxes):
        self.boxes = boxes

    def all_boxes(self, image, phrase, max_boxes=8):
        return self.boxes.get(phrase, [])


class FakeScorer:
    def __init__(self, sims):
        self.sims = sims

    def score(self, crop, phrase):
        return self.sims.get(phrase, 0.0)


IMG = Image.new("RGB", (64, 64), (10, 20, 30))
BOX = (0.0, 0.0, 32.0, 32.0)


def test_high_similarity_is_present():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.8}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "PRESENT"
    assert out["fox"].sim == 0.8


def test_low_similarity_is_missing():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.05}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "MISSING"


def test_no_box_abstains_and_never_reports_missing():
    gv = grounded.GroundedVerifier(
        FakeDetector({}), FakeScorer({}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "ABSTAIN"
    assert out["fox"].sim is None


def test_count_concepts_always_abstain():
    gv = grounded.GroundedVerifier(
        FakeDetector({"three": [(BOX, 0.9)]}), FakeScorer({"three": 0.9}), tau=0.25)
    out = gv.score(IMG, [Concept("three", "count")])
    assert out["three"].state == "ABSTAIN"


def test_relation_concepts_always_abstain():
    gv = grounded.GroundedVerifier(
        FakeDetector({"behind": [(BOX, 0.9)]}), FakeScorer({"behind": 0.9}), tau=0.25)
    out = gv.score(IMG, [Concept("behind", "relation")])
    assert out["behind"].state == "ABSTAIN"


def test_best_of_several_boxes_wins():
    boxes = [((0.0, 0.0, 16.0, 16.0), 0.5), ((16.0, 16.0, 48.0, 48.0), 0.4)]

    class PositionScorer:
        def score(self, crop, phrase):
            return 0.9 if crop.size == (32, 32) else 0.1

    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": boxes}), PositionScorer(), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].sim == 0.9
    assert out["fox"].box == (16.0, 16.0, 48.0, 48.0)


@pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.1, 1.5])
def test_constructor_rejects_tau_outside_open_unit_interval(bad_tau):
    with pytest.raises(ValueError, match="tau"):
        grounded.GroundedVerifier(FakeDetector({}), FakeScorer({}), tau=bad_tau)


def test_constructor_accepts_tau_inside_the_interval():
    gv = grounded.GroundedVerifier(FakeDetector({}), FakeScorer({}), tau=0.25)
    assert gv.tau == 0.25


def test_similarity_exactly_at_tau_is_present():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.25}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "PRESENT", "sim == tau must be PRESENT (>= not >)"


def test_similarity_just_below_tau_is_missing():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.2499}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "MISSING"


def _gv(sims, tau=0.25, delta=0.0):
    return grounded.GroundedVerifier(
        FakeDetector({"African grey parrot": [(BOX, 0.9)]}),
        FakeScorer(sims), tau=tau, delta=delta)


PARROT = Concept("African grey parrot", "subject", "parrot")


def test_coarse_term_winning_by_more_than_delta_is_a_fine_mismatch():
    # margin = 0.40 - 0.55 = -0.15, which is below delta=0.05
    gv = _gv({"African grey parrot": 0.40, "parrot": 0.55}, delta=0.05)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "FINE_MISMATCH"
    assert out["African grey parrot"].sim_coarse == 0.55


def test_margin_exactly_at_delta_is_present():
    # margin = 0.60 - 0.50 = 0.10 == delta, and the test is strict `<`
    gv = _gv({"African grey parrot": 0.60, "parrot": 0.50}, delta=0.10)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "PRESENT"


def test_margin_just_below_delta_is_a_fine_mismatch():
    # margin = 0.599 - 0.50 = 0.099 < delta = 0.10
    gv = _gv({"African grey parrot": 0.599, "parrot": 0.50}, delta=0.10)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "FINE_MISMATCH"


def test_missing_takes_precedence_over_fine_mismatch():
    # sim 0.10 < tau 0.25, and the margin 0.10-0.90 is also below delta
    gv = _gv({"African grey parrot": 0.10, "parrot": 0.90}, delta=0.05)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "MISSING"


def test_no_box_still_abstains_regardless_of_delta():
    gv = grounded.GroundedVerifier(
        FakeDetector({}), FakeScorer({"parrot": 0.9}), tau=0.25, delta=0.5)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "ABSTAIN"
    assert out["African grey parrot"].sim_coarse is None


def test_concept_without_a_coarse_term_is_never_a_fine_mismatch():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.30}),
        tau=0.25, delta=0.9)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "PRESENT"
    assert out["fox"].sim_coarse is None


@pytest.mark.parametrize("bad", [-1.5, 1.5])
def test_delta_outside_the_closed_unit_range_is_rejected(bad):
    with pytest.raises(ValueError, match="delta"):
        grounded.GroundedVerifier(FakeDetector({}), FakeScorer({}),
                                  tau=0.25, delta=bad)


class OrderedScorer:
    """Returns queued scores per phrase, in call order; the last value repeats.

    The module-level FakeScorer is constant per phrase, so it cannot express a
    candidate-dependent similarity -- which is exactly what the per-candidate
    persistence tests need to observe.
    """

    def __init__(self, by_phrase):
        self.by_phrase = {k: list(v) for k, v in by_phrase.items()}

    def score(self, crop, phrase):
        queue = self.by_phrase.get(phrase)
        if not queue:
            return 0.0
        return queue.pop(0) if len(queue) > 1 else queue[0]


CAT = Concept("cat", "subject", "animal")
TWO_BOXES = [((0.0, 0.0, 4.0, 4.0), 0.9), ((0.0, 0.0, 8.0, 8.0), 0.3)]


def test_every_candidate_box_is_persisted_not_just_the_winner():
    v = grounded.GroundedVerifier(
        FakeDetector({"cat": TWO_BOXES}), OrderedScorer({"cat": [0.4, 0.8]}),
        tau=0.25)
    out = v.score(IMG, [CAT])
    assert len(out["cat"].candidates) == 2
    assert [c.dino_conf for c in out["cat"].candidates] == [0.9, 0.3]
    assert [c.sim for c in out["cat"].candidates] == pytest.approx([0.4, 0.8])


def test_prototype_scores_are_recorded_for_every_candidate():
    import numpy as np

    from ragregen.verify import prototype

    bank = prototype.PrototypeBank({"cat": np.array([1.0, 0.0]),
                                    "animal": np.array([0.0, 1.0])})

    class Emb:
        def encode_pil(self, images, batch_size=32):
            return np.stack([np.array([1.0, 0.0])] * len(list(images)))

    v = grounded.GroundedVerifier(
        FakeDetector({"cat": TWO_BOXES}), OrderedScorer({"cat": [0.4, 0.8]}),
        tau=0.25, prototypes=bank, embedder=Emb())
    got = v.score(IMG, [CAT])["cat"]
    assert [c.sim_proto for c in got.candidates] == pytest.approx([1.0, 1.0])
    assert [c.sim_proto_coarse for c in got.candidates] == pytest.approx([0.0, 0.0])


def test_wiring_prototypes_does_not_move_the_state_or_the_selected_box():
    """The pinned tau = 0.25 baseline must reproduce bit for bit."""
    import numpy as np

    from ragregen.verify import prototype

    plain = grounded.GroundedVerifier(
        FakeDetector({"cat": TWO_BOXES}),
        OrderedScorer({"cat": [0.4, 0.8], "animal": [0.1]}), tau=0.25)
    before = plain.score(IMG, [CAT])["cat"]

    class Emb:
        def encode_pil(self, images, batch_size=32):
            return np.stack([np.array([0.0, 1.0])] * len(list(images)))

    withp = grounded.GroundedVerifier(
        FakeDetector({"cat": TWO_BOXES}),
        OrderedScorer({"cat": [0.4, 0.8], "animal": [0.1]}), tau=0.25,
        prototypes=prototype.PrototypeBank({"cat": np.array([1.0, 0.0])}),
        embedder=Emb())
    after = withp.score(IMG, [CAT])["cat"]

    assert (after.state, after.box, after.sim, after.sim_coarse) == \
           (before.state, before.box, before.sim, before.sim_coarse)


def test_enabled_prototype_margin_can_flag_a_fine_mismatch():
    import numpy as np
    from ragregen.verify import prototype

    bank = prototype.PrototypeBank({"cat": np.array([1.0, 0.0]),
                                    "animal": np.array([0.0, 1.0])})

    class Emb:
        def encode_pil(self, images, batch_size=32):
            return np.stack([np.array([0.2, 0.8])] * len(list(images)))

    verifier = grounded.GroundedVerifier(
        FakeDetector({"cat": [(BOX, 0.9)]}),
        FakeScorer({"cat": 0.8, "animal": 0.1}), tau=0.25,
        prototypes=bank, embedder=Emb(), prototype_delta=0.0)
    got = verifier.score(IMG, [Concept("cat", "subject", "animal")])["cat"]
    assert got.state == "FINE_MISMATCH"


def test_prototype_rule_uses_detector_confidence_box():
    import numpy as np
    from ragregen.verify import prototype

    boxes = [((0.0, 0.0, 8.0, 8.0), 0.9),
             ((0.0, 0.0, 16.0, 16.0), 0.2)]
    bank = prototype.PrototypeBank({"cat": np.array([1.0, 0.0]),
                                    "animal": np.array([0.0, 1.0])})

    class Emb:
        def encode_pil(self, images, batch_size=32):
            # High-confidence box is coarse-like; low-confidence box fine-like.
            return np.array([[0.0, 1.0], [1.0, 0.0]])

    verifier = grounded.GroundedVerifier(
        FakeDetector({"cat": boxes}),
        OrderedScorer({"cat": [0.4, 0.9], "animal": [0.1, 0.1]}),
        tau=0.25, prototypes=bank, embedder=Emb(), prototype_delta=0.0)
    got = verifier.score(IMG, [Concept("cat", "subject", "animal")])["cat"]
    assert got.box == boxes[1][0], "text winner remains persisted"
    assert got.state == "FINE_MISMATCH", "neutral box drives proto verdict"
