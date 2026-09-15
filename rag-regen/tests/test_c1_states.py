import pytest

from ragregen import c1


def test_state_at_margin_matches_state_at_for_the_text_mechanism():
    for sim in (0.0, 0.1, 0.25, 0.6, 0.9):
        for coarse in (None, 0.0, 0.3, 0.95):
            for delta in (-1.0, -0.2, 0.0, 0.2):
                margin = None if coarse is None else sim - coarse
                assert c1.state_at(sim, coarse, 0.25, delta) == \
                       c1.state_at_margin(sim, margin, 0.25, delta)


def test_missing_still_wins_over_a_failing_prototype_margin():
    # sim below tau, prototype margin far below delta: MISSING regardless.
    assert c1.state_at_margin(0.10, -0.9, 0.25, 0.0) == "MISSING"


def _cand(conf, proto, proto_coarse, sim=0.9, sim_coarse=0.0):
    return {"box": [0, 0, 1, 1], "dino_conf": conf, "sim": sim,
            "sim_coarse": sim_coarse, "sim_proto": proto,
            "sim_proto_coarse": proto_coarse}


def test_proto_margin_reads_the_prototype_fields_not_the_text_ones():
    concept = {"sim": 0.8, "sim_coarse": 0.1,
               "candidates": [_cand(0.9, 0.3, 0.5)]}
    assert c1.text_margin(concept) == pytest.approx(0.7)
    assert c1.proto_margin(concept) == pytest.approx(-0.2)


def test_proto_margin_uses_the_highest_confidence_box_not_the_selected_one():
    """The graded selection rule is detector confidence, and it is phrase-neutral.

    The top-level sim_proto belongs to the argmax-of-fine-phrase box. Grading
    on that would reinherit asymmetry (b) -- selection favouring the fine side
    before the margin is taken -- which is precisely what this mechanism exists
    to remove. This test is the guard against a refactor quietly switching back.
    """
    concept = {
        "sim": 0.9, "sim_coarse": 0.0,
        "sim_proto": 0.90, "sim_proto_coarse": 0.10,   # selected box: +0.80
        "candidates": [
            _cand(0.20, 0.90, 0.10),   # high fine-phrase sim, LOW confidence
            _cand(0.95, 0.10, 0.90),   # highest confidence: margin -0.80
        ],
    }
    assert c1.proto_margin(concept) == pytest.approx(-0.80)


def test_margins_are_none_when_the_coarse_side_is_absent():
    assert c1.text_margin({"sim": 0.8, "sim_coarse": None}) is None
    assert c1.proto_margin({"sim": 0.8, "candidates": []}) is None
    assert c1.proto_margin({"sim": 0.8}) is None
    assert c1.proto_margin(
        {"sim": 0.8, "candidates": [_cand(0.9, 0.5, None)]}) is None


def test_grounded_fails_can_be_driven_by_the_prototype_margin():
    scores = {"cat": {"sim": 0.8, "sim_coarse": 0.1,
                      "candidates": [_cand(0.9, 0.3, 0.5)]}}
    assert not c1.grounded_fails(scores, tau=0.25, delta=0.0)
    assert c1.grounded_fails(scores, tau=0.25, delta=0.0,
                             margin_of=c1.proto_margin)
