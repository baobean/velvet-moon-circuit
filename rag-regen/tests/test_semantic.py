import pytest
from PIL import Image

from ragregen.verify import semantic

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVLM:
    def __init__(self, reply):
        self.reply = reply
        self.asked = []

    def ask(self, image, prompt):
        self.asked.append(prompt)
        return self.reply


def test_clean_pass_is_parsed():
    v = semantic.SemanticVerifier(FakeVLM('{"verdict": "PASS", "issues": []}'))
    out = v.judge(IMG, "a fox")
    assert out.ok is True
    assert out.issues == []
    assert out.degenerate is False


def test_fail_with_issues_is_parsed():
    reply = ('{"verdict": "FAIL", "issues": '
             '[{"concept": "Amur leopard", "problem": "rosettes are wrong"}]}')
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "an Amur leopard")
    assert out.ok is False
    assert out.issues[0].concept == "Amur leopard"
    assert out.issues[0].problem == "rosettes are wrong"


def test_json_embedded_in_prose_is_recovered():
    reply = 'Sure! Here is my answer:\n{"verdict": "FAIL", "issues": []}\nHope that helps.'
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is False


def test_unparseable_reply_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("I cannot tell.")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_empty_reply_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_nan_signature_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("!!!! !!!! !!!!")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_raw_reply_is_preserved_for_the_trace():
    out = semantic.SemanticVerifier(FakeVLM("garbage")).judge(IMG, "a fox")
    assert out.raw == "garbage"


def test_prompt_is_interpolated_into_the_template():
    vlm = FakeVLM('{"verdict": "PASS", "issues": []}')
    semantic.SemanticVerifier(vlm).judge(IMG, "a monkey puzzle tree")
    assert "a monkey puzzle tree" in vlm.asked[0]


# --- Additional coverage beyond the brief ---
#
# The brief's `_extract_json` uses `re.search(r"\{.*\}", text, re.DOTALL)`,
# which is greedy: it spans from the FIRST '{' to the LAST '}' in the whole
# reply. That mis-extracts whenever there is more than one JSON-braced
# fragment, or when prose *after* the real JSON object itself contains a
# stray '}' (e.g. a reply ending "... and that's my answer :}"). The fixed
# implementation uses `json.JSONDecoder().raw_decode` anchored at each '{'
# so it recovers exactly the first well-formed object and ignores trailing
# noise.


def test_stray_brace_in_trailing_prose_does_not_break_extraction():
    # A correct, well-formed reply followed by a smiley containing '}'.
    # The greedy regex captures through the smiley's '}', producing a
    # string with trailing junk after the JSON object, which fails to
    # parse -- a good PASS reply would wrongly be marked degenerate.
    reply = '{"verdict": "PASS", "issues": []}\nGreat, glad I could help :}'
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is True
    assert out.degenerate is False


def test_first_of_two_json_objects_in_reply_is_used():
    # A model that "thinks out loud" and emits two JSON-shaped fragments.
    # The greedy regex spans both, producing unparseable "extra data".
    # The fix takes the first well-formed object and ignores the rest.
    reply = ('First draft: {"verdict": "FAIL", "issues": []} '
             'actually wait, {"verdict": "PASS", "issues": []}')
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is False


def test_pass_verdict_with_nonempty_issues_is_contradictory_and_fails_closed():
    # The template instructs "issues must be empty when the verdict is
    # PASS". A reply that violates its own contract is not trustworthy
    # enough to grant a silent PASS.
    reply = ('{"verdict": "PASS", "issues": '
             '[{"concept": "fox", "problem": "actually two foxes"}]}')
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_top_level_json_array_fails_closed():
    # A bare JSON array (no enclosing object) parses fine as JSON but is
    # not the expected {"verdict": ..., "issues": ...} shape.
    reply = '[{"verdict": "PASS", "issues": []}]'
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_verdict_with_non_string_value_fails_closed():
    reply = '{"verdict": 1, "issues": []}'
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_reference_verifier_sends_candidate_before_references():
    class MultiVLM:
        def __init__(self):
            self.images = None
            self.prompt = None

        def ask_images(self, images, prompt):
            self.images, self.prompt = list(images), prompt
            return '{"verdict": "FAIL", "issues": []}'

    candidate = Image.new("RGB", (8, 8), "red")
    reference = Image.new("RGB", (8, 8), "blue")
    model = MultiVLM()
    got = semantic.ReferenceSemanticVerifier(model).judge(
        candidate, "an Amur leopard", "Amur leopard", [reference])
    assert model.images == [candidate, reference]
    assert "FIRST image" in model.prompt
    assert "Amur leopard" in model.prompt
    assert got.ok is False


def test_reference_verifier_refuses_an_empty_reference_set():
    with pytest.raises(ValueError, match="reference"):
        semantic.ReferenceSemanticVerifier(object()).judge(
            IMG, "a fox", "fox", [])


def test_reference_verifier_rejects_a_bare_unevidenced_pass():
    class MultiVLM:
        def ask_images(self, images, prompt):
            return '{"verdict":"PASS","issues":[]}'

    got = semantic.ReferenceSemanticVerifier(MultiVLM()).judge(
        IMG, "a fox", "fox", [IMG])
    assert got.ok is False
    assert got.degenerate is True


def test_reference_verifier_accepts_a_supported_pass():
    class MultiVLM:
        def ask_images(self, images, prompt):
            return (
                '{"reference_traits":["red coat","pointed ears","white '
                'muzzle"],"candidate_evidence":["red coat visible","ears '
                'are pointed","muzzle is white"],"mismatches":[],"verdict"'
                ':"PASS","issues":[]}')

    got = semantic.ReferenceSemanticVerifier(MultiVLM()).judge(
        IMG, "a fox", "fox", [IMG])
    assert got.ok is True
    assert got.degenerate is False
