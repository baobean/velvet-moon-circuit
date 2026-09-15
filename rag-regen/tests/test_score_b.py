import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.verify.semantic import Issue, SemanticVerdict  # noqa: E402
from scripts import score_b  # noqa: E402

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVerifier:
    def __init__(self, verdict):
        self._verdict = verdict
        self.prompts = []

    def judge(self, image, prompt):
        self.prompts.append(prompt)
        return self._verdict


def test_score_case_serialises_a_clean_pass():
    v = SemanticVerdict(ok=True, raw='{"verdict": "PASS"}')
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert out["ok"] is True
    assert out["degenerate"] is False
    assert out["issues"] == []
    assert out["raw"] == '{"verdict": "PASS"}'


def test_score_case_serialises_issues():
    v = SemanticVerdict(ok=False, raw="{}",
                        issues=[Issue("Amur leopard", "rosettes wrong")])
    out = score_b.score_case(FakeVerifier(v), IMG, "an Amur leopard")
    assert out["issues"] == [{"concept": "Amur leopard",
                              "problem": "rosettes wrong"}]


def test_degenerate_is_preserved_not_flattened_into_ok():
    """A degenerate reply fails closed, so it looks identical to a real FAIL
    unless the flag survives. The C1 report needs the count to explain an
    inflated semantic recall."""
    v = SemanticVerdict(ok=False, raw="!!!!", degenerate=True)
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert out["ok"] is False
    assert out["degenerate"] is True
    assert out["raw"] == "!!!!"


def test_the_case_prompt_is_what_gets_judged():
    v = SemanticVerdict(ok=True, raw="{}")
    fake = FakeVerifier(v)
    score_b.score_case(fake, IMG, "a durian on a market table")
    assert fake.prompts == ["a durian on a market table"]


def test_output_is_json_serialisable():
    import json
    v = SemanticVerdict(ok=False, raw="x", issues=[Issue("a", "b")])
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert json.loads(json.dumps(out)) == out
