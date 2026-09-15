from ragregen.verify import fusion
from ragregen.verify.fusion import fuse
from ragregen.verify.grounded import ConceptScore
from ragregen.verify.semantic import Issue, SemanticVerdict


def cs(phrase, state, sim=None, kind="subject"):
    return ConceptScore(phrase, kind, state, sim=sim)


PASS = SemanticVerdict(ok=True, raw="{}")


def fail(concept="x", problem="p"):
    return SemanticVerdict(ok=False, raw="{}", issues=[Issue(concept, problem)])


def test_both_pass_is_ok():
    v = fusion.fuse({"fox": cs("fox", "PRESENT", 0.8)}, PASS)
    assert v.ok is True
    assert v.agreement == "BOTH_PASS"
    assert v.target is None


def test_grounded_failure_fails_the_verdict():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, PASS)
    assert v.ok is False
    assert v.target == "fox"
    assert v.agreement == "GROUNDED_ONLY"


def test_semantic_failure_alone_fails_the_verdict():
    v = fusion.fuse({"fox": cs("fox", "PRESENT", 0.8)}, fail("fox", "wrong breed"))
    assert v.ok is False
    assert v.target == "fox"
    assert v.agreement == "SEMANTIC_ONLY"


def test_both_failing_is_recorded_as_both_fail():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, fail())
    assert v.agreement == "BOTH_FAIL"


def test_grounded_target_wins_over_semantic_target():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, fail("badger", "p"))
    assert v.target == "fox"


def test_lowest_scoring_missing_concept_is_the_target():
    scores = {"fox": cs("fox", "MISSING", 0.20),
              "tree": cs("tree", "MISSING", 0.02)}
    v = fusion.fuse(scores, PASS)
    assert v.target == "tree"


def test_abstain_alone_does_not_fail_the_verdict():
    v = fusion.fuse({"behind": cs("behind", "ABSTAIN", kind="relation")}, PASS)
    assert v.ok is True
    assert v.target is None


def test_all_abstain_score_is_none_regardless_of_the_semantic_verdict():
    """The old behaviour faked a score from the semantic stream's verdict --
    1.0 on pass, 0.0 on fail. Stream A saw nothing either way, so the score
    is absent either way (see test_score_is_none_when_stream_a_contributed_nothing)."""
    scores = {"behind": cs("behind", "ABSTAIN", kind="relation")}
    assert fusion.fuse(scores, PASS).score is None
    assert fusion.fuse(scores, fail()).score is None


def test_score_is_the_minimum_non_abstain_similarity():
    scores = {"fox": cs("fox", "PRESENT", 0.8),
              "tree": cs("tree", "PRESENT", 0.3),
              "behind": cs("behind", "ABSTAIN", kind="relation")}
    assert fusion.fuse(scores, PASS).score == 0.3


def test_missing_with_no_box_still_fails_the_verdict():
    """A contract-violating scorer yields MISSING with box=None, which looks
    exactly like ABSTAIN if fusion pattern-matches on the box (Task 6)."""
    score = ConceptScore("fox", "subject", "MISSING", box=None, sim=-1.0)
    v = fusion.fuse({"fox": score}, PASS)
    assert v.ok is False
    assert v.target == "fox"
    assert v.agreement == "GROUNDED_ONLY"


def test_missing_without_a_similarity_can_still_be_the_target():
    """sim=None must not crash the target selection, and must sort as the
    weakest candidate rather than being skipped."""
    scores = {"fox": cs("fox", "MISSING", None),
              "tree": cs("tree", "MISSING", 0.02)}
    v = fusion.fuse(scores, PASS)
    assert v.target == "fox"


def test_every_agreement_produced_is_a_declared_one():
    cases = [({"fox": cs("fox", "PRESENT", 0.8)}, PASS),
             ({"fox": cs("fox", "MISSING", 0.05)}, PASS),
             ({"fox": cs("fox", "PRESENT", 0.8)}, fail()),
             ({"fox": cs("fox", "MISSING", 0.05)}, fail())]
    produced = {fusion.fuse(g, s).agreement for g, s in cases}
    assert produced == set(fusion.AGREEMENTS)


def test_fine_mismatch_fails_the_case():
    scores = {"African grey parrot": ConceptScore(
        "African grey parrot", "subject", "FINE_MISMATCH",
        box=(0, 0, 8, 8), sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, PASS)
    assert v.ok is False
    assert v.agreement == "GROUNDED_ONLY"
    assert v.target == "African grey parrot"


def test_fine_mismatch_competes_with_missing_for_the_target():
    scores = {
        "parrot_a": ConceptScore("parrot_a", "subject", "FINE_MISMATCH",
                                 box=(0, 0, 8, 8), sim=0.10, sim_coarse=0.55),
        "parrot_b": ConceptScore("parrot_b", "object", "MISSING",
                                 box=(0, 0, 8, 8), sim=0.40),
    }
    v = fusion.fuse(scores, PASS)
    assert v.target == "parrot_a", "lowest sim among all failing states wins"


def test_evidence_carries_sim_coarse():
    scores = {"African grey parrot": ConceptScore(
        "African grey parrot", "subject", "FINE_MISMATCH",
        box=(0, 0, 8, 8), sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, PASS)
    assert v.evidence["grounded"]["African grey parrot"]["sim_coarse"] == 0.55


def test_fine_mismatch_with_a_none_box_still_fails():
    """A contract-violating scorer must not be rescued by box inspection."""
    scores = {"p": ConceptScore("p", "subject", "FINE_MISMATCH",
                                box=None, sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, PASS)
    assert v.ok is False


def test_score_is_none_when_stream_a_contributed_nothing():
    """Every concept abstained. 1.0 would be a number we do not have."""
    scores = {"three": ConceptScore("three", "count", "ABSTAIN")}
    v = fuse(scores, SemanticVerdict(ok=True, raw=""))
    assert v.score is None
    assert v.ok is True


def test_score_is_the_weakest_concept_when_stream_a_spoke():
    scores = {
        "fox": ConceptScore("fox", "subject", "PRESENT", sim=0.8),
        "tree": ConceptScore("tree", "subject", "PRESENT", sim=0.3),
    }
    assert fuse(scores, SemanticVerdict(ok=True, raw="")).score == 0.3
