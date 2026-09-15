import json

from PIL import Image

from ragregen.verify.semantic import Issue, SemanticVerdict
from scripts import score_b_reference


class FakeVerifier:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def judge(self, image, prompt, concept, references):
        self.calls.append((image, prompt, concept, references))
        return self.verdict


def test_score_case_supplies_ordered_references_and_serialises_reply():
    candidate = Image.new("RGB", (8, 8))
    references = [Image.new("RGB", (4, 4)), Image.new("RGB", (6, 6))]
    verdict = SemanticVerdict(
        ok=False, raw='{"verdict":"FAIL"}',
        issues=[Issue("axolotl", "external gills do not match")])
    verifier = FakeVerifier(verdict)

    got = score_b_reference.score_case(
        verifier, candidate, "an axolotl underwater", "axolotl", references)

    assert verifier.calls == [
        (candidate, "an axolotl underwater", "axolotl", references)]
    assert got["ok"] is False
    assert got["issues"] == [
        {"concept": "axolotl", "problem": "external gills do not match"}]
    assert json.loads(json.dumps(got)) == got
