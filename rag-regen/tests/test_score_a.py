import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.verify.grounded import ConceptScore  # noqa: E402
from scripts import score_a  # noqa: E402

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVerifier:
    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def score(self, image, concepts):
        self.calls.append([c.phrase for c in concepts])
        return self._scores


def test_score_case_serialises_every_field_needed_for_the_tau_sweep():
    scores = {"fox": ConceptScore("fox", "subject", "PRESENT",
                                  box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.8, sim=0.42)}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox", None)
    assert out["fox"]["sim"] == 0.42
    assert out["fox"]["kind"] == "subject"
    assert out["fox"]["box"] == [1.0, 2.0, 3.0, 4.0]
    assert out["fox"]["dino_conf"] == 0.8


def test_no_thresholded_verdict_is_persisted():
    """`state` is a verdict at this process's arbitrary tau and delta.

    Persisting it would contradict the module docstring's promise of raw
    similarities only, and invite a reader to trust a verdict at tau=0.5 with
    the contrastive rule switched off. The state is recovered from `sim` and
    `sim_coarse` by c1.state_at instead, at whatever tau and delta are asked
    for -- as the next two assertions demonstrate.
    """
    scores = {"fox": ConceptScore("fox", "subject", "PRESENT",
                                  box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.8,
                                  sim=0.42, sim_coarse=0.10)}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox", "canine")
    assert "state" not in out["fox"]

    from ragregen import c1
    assert c1.state_at(out["fox"]["sim"], out["fox"]["sim_coarse"],
                       0.25, c1.DELTA_OFF) == "PRESENT"
    assert c1.state_at(out["fox"]["sim"], out["fox"]["sim_coarse"],
                       0.50, c1.DELTA_OFF) == "MISSING"


def test_abstain_is_serialised_with_a_null_sim():
    """sim=None is what makes ABSTAIN recoverable offline at any tau."""
    scores = {"fox": ConceptScore("fox", "subject", "ABSTAIN")}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox", None)
    assert out["fox"]["sim"] is None
    assert out["fox"]["box"] is None

    from ragregen import c1
    assert c1.state_at(out["fox"]["sim"], out["fox"]["sim_coarse"],
                       0.25, c1.DELTA_OFF) == "ABSTAIN"


def test_the_case_concept_is_parsed_as_the_target():
    v = FakeVerifier({})
    score_a.score_case(v, IMG, "an Amur leopard on snow", "Amur leopard", None)
    assert v.calls[0][0] == "Amur leopard"


def test_output_is_json_serialisable():
    import json
    scores = {"fox": ConceptScore("fox", "subject", "MISSING",
                                  box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.5, sim=0.1)}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox", None)
    assert json.loads(json.dumps(out)) == out


class RecordingVerifier:
    """Captures the parsed concepts so the test can assert what was scored."""

    def __init__(self):
        self.seen = None

    def score(self, image, parsed):
        self.seen = parsed
        return {
            c.phrase: ConceptScore(c.phrase, c.kind, "PRESENT",
                                   box=(0.0, 0.0, 8.0, 8.0), dino_conf=0.9,
                                   sim=0.7, sim_coarse=0.3 if c.coarse else None)
            for c in parsed
        }


def test_score_case_forwards_the_coarse_term_to_the_parser():
    v = RecordingVerifier()
    score_a.score_case(v, IMG, "an African grey parrot on a branch",
                       "African grey parrot", "parrot")
    target = v.seen[0]
    assert target.phrase == "African grey parrot"
    assert target.coarse == "parrot"


def test_score_case_persists_sim_coarse():
    v = RecordingVerifier()
    out = score_a.score_case(v, IMG, "an African grey parrot on a branch",
                             "African grey parrot", "parrot")
    assert out["African grey parrot"]["sim_coarse"] == 0.3
    assert "sim" in out["African grey parrot"]


def test_score_case_serialises_every_candidate():
    from ragregen.verify.grounded import CandidateScore, ConceptScore

    class V:
        def score(self, image, concepts):
            return {"cat": ConceptScore(
                "cat", "object", "PRESENT", box=(0, 0, 1, 1), dino_conf=0.9,
                sim=0.8, sim_coarse=0.1, sim_proto=0.7, sim_proto_coarse=0.2,
                candidates=(
                    CandidateScore((0, 0, 1, 1), 0.9, 0.8, 0.1, 0.7, 0.2),
                    CandidateScore((0, 0, 2, 2), 0.4, 0.3, 0.2, 0.5, 0.4),
                ))}

    got = score_a.score_case(V(), object(), "p", "cat", "animal")
    assert got["cat"]["sim_proto"] == 0.7
    assert got["cat"]["sim_proto_coarse"] == 0.2
    assert len(got["cat"]["candidates"]) == 2
    assert got["cat"]["candidates"][1]["dino_conf"] == 0.4
    assert "state" not in got["cat"]


def test_siglip_scorer_exposes_the_embedding_interface_without_second_model():
    from ragregen.models import SigLIPScorer

    assert callable(SigLIPScorer.encode_pil)
    assert callable(SigLIPScorer.encode_text)
