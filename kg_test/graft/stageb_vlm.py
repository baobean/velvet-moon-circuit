"""The only VLM-touching code in Stage-B Gate 0: held-out attribute read (name-blind schema)
and the frozen bidirectional equivalence judge. Kept thin so graft.stageb_gate0 stays pure and
mock-testable. Prompts/decoding are frozen per the pre-registered spec."""
from __future__ import annotations
from graft import kg_build
from graft.stageb_gate0 import PART_ATTRS, NOT_VISIBLE, norm

_PA = set(PART_ATTRS)
_JUDGE = ('You are comparing two descriptions of a plant\'s {A}. A: "{x}". B: "{y}". '
          "Do A and B describe essentially the same {A}? Answer with only 'yes' or 'no'.")


def read_attributes(image_paths, models):
    """VLM-read the 5 part attributes from one or more images (name-blind schema)."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    raw = models.vlm.describe(imgs, kg_build.SCHEMA_INSTRUCTION_NEUTRAL,
                              do_sample=False, max_new_tokens=512)
    _global, parts = kg_build.parse_vlm_schema(raw)
    out = {}
    for part, nodes in parts.items():
        for n in nodes:
            if (part, n.name) in _PA:
                out[(part, n.name)] = n.value
    return out


def _yes(reply):
    return str(reply).strip().lower().startswith("yes")


def judge_match(x, y, attr_name, models):
    """Frozen bidirectional equivalence judge: match iff BOTH orders answer 'yes'.
    A non-answer (either side 'not visible') is an automatic miss with no VLM call.
    QwenVLM requires >=1 image, so a fixed blank 8x8 image is passed; the frozen text
    prompt is unchanged and decoding is deterministic (do_sample=False)."""
    if norm(x) in NOT_VISIBLE or norm(y) in NOT_VISIBLE:
        return False
    from PIL import Image
    blank = Image.new("RGB", (8, 8), "white")
    a = models.vlm.describe([blank], _JUDGE.format(A=attr_name, x=x, y=y),
                            do_sample=False, max_new_tokens=8)
    b = models.vlm.describe([blank], _JUDGE.format(A=attr_name, x=y, y=x),
                            do_sample=False, max_new_tokens=8)
    return _yes(a) and _yes(b)
