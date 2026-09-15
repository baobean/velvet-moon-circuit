import pytest

from ragregen import c1
from ragregen.concepts import Concept


def A(**phrases):
    """Stream A record: phrase -> sim (None means ABSTAIN)."""
    return {p: {"sim": s, "kind": "subject"} for p, s in phrases.items()}


def B(ok, degenerate=False):
    return {"ok": ok, "degenerate": degenerate, "raw": "", "issues": []}


# --- state_at, tau only -----------------------------------------------------

def test_abstain_when_sim_is_none_regardless_of_tau():
    assert c1.state_at(None, None, 0.01, c1.DELTA_OFF) == "ABSTAIN"
    assert c1.state_at(None, None, 0.99, c1.DELTA_OFF) == "ABSTAIN"


def test_present_at_or_above_tau_missing_below():
    assert c1.state_at(0.30, None, 0.25, c1.DELTA_OFF) == "PRESENT"
    assert c1.state_at(0.20, None, 0.25, c1.DELTA_OFF) == "MISSING"


def test_sim_exactly_equal_to_tau_is_present():
    """Must match `best_sim >= self.tau` at grounded.py:83, which Plan 1 pins
    with its own boundary test. Flipping >= to > here would silently make this
    module disagree with the verifier it is grading."""
    assert c1.state_at(0.25, None, 0.25, c1.DELTA_OFF) == "PRESENT"


def test_delta_off_can_never_fire_the_contrastive_rule():
    """DELTA_OFF is the mechanism switched off, unlike delta=0.0.

    Similarities live in [0, 1], so the widest possible margin is -1.0 and the
    strict `<` in state_at cannot fire at DELTA_OFF. delta=0.0 does fire on the
    same input, which is exactly why it is not a safe default.
    """
    assert c1.DELTA_OFF == -1.0
    assert c1.state_at(0.0, 1.0, 0.0, c1.DELTA_OFF) == "PRESENT"
    assert c1.state_at(0.0, 1.0, 0.0, 0.0) == "FINE_MISMATCH"


# --- arms -----------------------------------------------------------------

def test_grounded_fails_when_any_concept_is_missing():
    assert c1.grounded_fails(A(fox=0.10, tree=0.90), tau=0.25) is True


def test_grounded_passes_when_all_present():
    assert c1.grounded_fails(A(fox=0.30, tree=0.90), tau=0.25) is False


def test_abstain_alone_never_fails_the_grounded_arm():
    """ABSTAIN != MISSING (parent spec §2). A detector that found nothing is
    not evidence the concept is absent."""
    assert c1.grounded_fails(A(fox=None, tree=None), tau=0.25) is False


def test_grounded_arm_on_an_empty_record_passes():
    assert c1.grounded_fails({}, tau=0.25) is False


def test_semantic_arm_reads_ok():
    assert c1.semantic_fails(B(ok=False)) is True
    assert c1.semantic_fails(B(ok=True)) is False


@pytest.mark.parametrize("g_sim,b_ok,expected", [
    (0.10, False, True),   # both fail
    (0.10, True, True),    # grounded only
    (0.90, False, True),   # semantic only
    (0.90, True, False),   # neither
])
def test_fused_is_the_or_of_both_arms(g_sim, b_ok, expected):
    """fuse() fails if EITHER stream fails (fusion.py). This mirrors it, and
    the OR is precisely why fused recall cannot be evidence for C1."""
    assert c1.fused_fails(A(fox=g_sim), B(ok=b_ok), tau=0.25) is expected


# --- rates ----------------------------------------------------------------

def test_abstain_rate_counts_concepts_not_cases():
    stream_a = {"c1": A(fox=None, tree=0.5), "c2": A(fox=0.4)}
    assert c1.abstain_rate(stream_a, tau=0.25) == pytest.approx(1 / 3)


def test_abstain_rate_of_nothing_is_zero():
    assert c1.abstain_rate({}, tau=0.25) == 0.0


# --- sweep ----------------------------------------------------------------

def test_sweep_returns_one_row_per_tau_with_all_three_arms():
    stream_a = {"a": A(fox=0.10), "b": A(fox=0.90)}
    stream_b = {"a": B(ok=False), "b": B(ok=True)}
    rows = c1.sweep_tau(stream_a, stream_b, ["a", "b"], [True, False],
                        grid=(0.05, 0.5))
    assert [r["tau"] for r in rows] == [0.05, 0.5]
    for r in rows:
        assert set(r["arms"]) == {"grounded", "semantic", "fused"}


def test_sweep_tracks_tau_changing_the_grounded_arm():
    """At tau=0.05 nothing is MISSING; at tau=0.5 case 'a' is. The grounded arm
    must therefore go from useless to perfect across the sweep."""
    stream_a = {"a": A(fox=0.10), "b": A(fox=0.90)}
    stream_b = {"a": B(ok=True), "b": B(ok=True)}
    lo, hi = c1.sweep_tau(stream_a, stream_b, ["a", "b"], [True, False],
                          grid=(0.05, 0.5))
    assert lo["arms"]["grounded"].recall == 0.0
    assert hi["arms"]["grounded"].recall == 1.0
    assert hi["arms"]["grounded"].balanced_accuracy == 1.0


def test_sweep_rejects_a_case_missing_from_a_stream():
    with pytest.raises(KeyError, match="case 'b' missing from stream_b"):
        c1.sweep_tau({"a": A(fox=0.1), "b": A(fox=0.1)}, {"a": B(ok=True)},
                     ["a", "b"], [True, True], grid=(0.25,))


def test_sweep_tau_does_not_apply_the_contrastive_rule():
    """A negative margin above tau must not fail the grounded arm in the tau report.

    The tau tables are the pre-contrastive verifier, and are compared directly
    against the published C1 numbers. A delta default of 0.0 fires
    FINE_MISMATCH here and silently moves a published row with no model change.
    """
    sa = {"c": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    rows = c1.sweep_tau(sa, {"c": B(ok=True)}, ["c"], [True], grid=(0.25,))
    assert rows[0]["arms"]["grounded"].tp == 0
    assert rows[0]["arms"]["fused"].tp == 0


def test_default_grid_spans_the_open_interval():
    assert min(c1.TAU_GRID) > 0.0
    assert max(c1.TAU_GRID) < 1.0
    assert len(c1.TAU_GRID) >= 20


# --- state_at / delta -------------------------------------------------------

def test_state_at_matches_the_verifier_on_the_same_inputs():
    """The report must not grade the verifier against a rule it does not use."""
    from ragregen.verify import grounded as g

    class D:
        def all_boxes(self, image, phrase, max_boxes=8):
            return [((0.0, 0.0, 8.0, 8.0), 0.9)]

    class S:
        def score(self, crop, phrase):
            return {"fine": 0.40, "coarse": 0.55}[phrase]

    from PIL import Image
    gv = g.GroundedVerifier(D(), S(), tau=0.25, delta=0.05)
    out = gv.score(Image.new("RGB", (16, 16)),
                   [Concept("fine", "subject", "coarse")])
    live = out["fine"]
    assert c1.state_at(live.sim, live.sim_coarse, 0.25, 0.05) == live.state

    # Boundary case: margin == delta exactly. round(0.60 - 0.50, 9) == 0.10,
    # so this pins the strict `<` (PRESENT here, not FINE_MISMATCH). A mutant
    # that weakens the comparison to `<=` passes the case above (margin is
    # -0.15, nowhere near its delta of 0.05) but is caught here.
    class S2:
        def score(self, crop, phrase):
            return {"fine": 0.60, "coarse": 0.50}[phrase]

    gv2 = g.GroundedVerifier(D(), S2(), tau=0.25, delta=0.10)
    out2 = gv2.score(Image.new("RGB", (16, 16)),
                     [Concept("fine", "subject", "coarse")])
    live2 = out2["fine"]
    assert live2.state == "PRESENT"
    assert c1.state_at(live2.sim, live2.sim_coarse, 0.25, 0.10) == live2.state

    # Precedence case: sim < tau AND round(sim - sim_coarse, 9) < delta are
    # BOTH true here (0.10 < 0.25, and round(0.10 - 0.90, 9) == -0.8 < 0.05),
    # so MISSING and FINE_MISMATCH genuinely compete. Without this case a
    # c1-only edit that checks FINE_MISMATCH before MISSING passes every
    # other fixture in this file (they all have sim >= tau whenever the
    # margin condition matters) and silently inverts the precedence.
    class S3:
        def score(self, crop, phrase):
            return {"fine": 0.10, "coarse": 0.90}[phrase]

    gv3 = g.GroundedVerifier(D(), S3(), tau=0.25, delta=0.05)
    out3 = gv3.score(Image.new("RGB", (16, 16)),
                     [Concept("fine", "subject", "coarse")])
    live3 = out3["fine"]
    assert live3.state == "MISSING"
    assert c1.state_at(live3.sim, live3.sim_coarse, 0.25, 0.05) == live3.state


def test_state_at_is_missing_below_tau_even_when_the_margin_is_wide():
    assert c1.state_at(0.10, 0.05, 0.25, 0.0) == "MISSING"


def test_state_at_abstains_on_a_none_similarity():
    assert c1.state_at(None, None, 0.25, 0.5) == "ABSTAIN"


def test_state_at_ignores_delta_when_no_coarse_similarity_exists():
    assert c1.state_at(0.30, None, 0.25, 0.9) == "PRESENT"


def test_fine_mismatch_makes_the_grounded_arm_fail():
    case = {"p": {"sim": 0.40, "sim_coarse": 0.55}}
    assert c1.grounded_fails(case, tau=0.25, delta=0.05) is True
    assert c1.grounded_fails(case, tau=0.25, delta=-0.5) is False


def test_delta_grid_spans_both_signs_and_includes_zero():
    assert c1.DELTA_GRID[0] == -0.2
    assert c1.DELTA_GRID[-1] == 0.2
    assert 0.0 in c1.DELTA_GRID
    assert len(c1.DELTA_GRID) == 81


def test_c1_and_fusion_agree_on_what_fails():
    """c1.FAILING_STATES is a deliberate copy of fusion.FAILING_STATES.

    c1 is offline analysis and must not import the verify package, so nothing
    but this test stops the two from drifting -- and if they drift, the report
    grades the verifier against a rule it does not use.
    """
    from ragregen.verify import fusion
    assert c1.FAILING_STATES == fusion.FAILING_STATES


def test_sweep_delta_returns_one_row_per_grid_value():
    # margin = 0.55 - 0.50 = 0.05, which straddles the (-0.1, 0.0, 0.1) grid:
    # FINE_MISMATCH only once delta exceeds the margin. sim=0.55 is well above
    # BASELINE_TAU=0.25 so the transition is delta-driven, not tau-driven.
    stream_a = {"c0": {"p": {"sim": 0.55, "sim_coarse": 0.50}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    rows = c1.sweep_delta(stream_a, stream_b, ["c0"], [True],
                          grid=(-0.1, 0.0, 0.1))
    assert [r["delta"] for r in rows] == [-0.1, 0.0, 0.1]
    assert rows[0]["arms"]["grounded"].recall == 0.0
    assert rows[2]["arms"]["grounded"].recall == 1.0
