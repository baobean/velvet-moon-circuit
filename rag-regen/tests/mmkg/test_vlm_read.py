from ragregen.mmkg import vlm_read as vr

class FakeAsker:
    def __init__(self, replies): self.replies = list(replies); self.calls = []
    def ask(self, image, prompt): self.calls.append((image, prompt)); return self.replies.pop(0)

def test_read_slots_parses_json_and_fills_missing(tmp_path):
    from PIL import Image
    from ragregen.mmkg.schema import SLOTS
    p = tmp_path / "x.png"; Image.new("RGB", (8, 8), "white").save(p)
    reply = '{"primary_color":"Red","surface_texture":"rough"}'   # 4 slots missing
    out = vr.read_slots(str(p), FakeAsker([reply]))
    assert out["primary_color"] == "Red" and out["surface_texture"] == "rough"
    assert out["distinctive_feature"] == "not visible"            # missing key filled
    assert set(out) == set(SLOTS)

def test_judge_bidirectional_and_notvisible_shortcircuit():
    a = FakeAsker(["Yes", "yes"]); assert vr.judge_match("red","crimson","primary_color", a) is True
    a = FakeAsker(["yes", "no"]);  assert vr.judge_match("red","blue","primary_color", a) is False
    a = FakeAsker([]);             assert vr.judge_match("not visible","red","primary_color", a) is False
    assert a.calls == []                                          # no VLM call when a value is not-visible
