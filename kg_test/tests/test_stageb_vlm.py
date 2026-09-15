from PIL import Image
from graft import stageb_vlm as sv


class FakeVLM:
    def __init__(self, replies): self.replies = list(replies); self.calls = []
    def describe(self, images, instruction, **kw):
        self.calls.append((len(images), instruction, kw)); return self.replies.pop(0)


class FakeModels:
    def __init__(self, vlm): self.vlm = vlm


def test_read_attributes_parse_only(monkeypatch):
    reply = ('{"global":{}, "parts":{"bark":{"texture":"rough","color":"brown"},'
             '"leaf":{"shape":"ovate"},"branching":{"pattern":"irregular"},'
             '"cone_or_flower":{"appearance":"not visible"}}}')
    m = FakeModels(FakeVLM([reply]))
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (8, 8)))
    out = sv.read_attributes(["x.jpg"], m)
    assert out[("bark", "texture")] == "rough"
    assert out[("leaf", "shape")] == "ovate"
    assert ("cone_or_flower", "appearance") not in out    # not one of the 5 PART_ATTRS


def test_judge_match_bidirectional_and():
    m = FakeModels(FakeVLM(["Yes", "yes"]))
    assert sv.judge_match("rough", "rough, flaky", "bark texture", m) is True
    m = FakeModels(FakeVLM(["yes", "no"]))
    assert sv.judge_match("rough", "smooth", "bark texture", m) is False
    m = FakeModels(FakeVLM([]))
    assert sv.judge_match("not visible", "rough", "bark texture", m) is False
    assert m.vlm.calls == []            # no VLM call on a not-visible non-answer
