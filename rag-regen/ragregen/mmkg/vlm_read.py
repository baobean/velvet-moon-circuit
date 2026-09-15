from __future__ import annotations
import json, re
from ragregen.mmkg.schema import SLOTS, NOT_VISIBLE, norm

READ_PROMPT = ('Describe ONLY the single main subject in this image. Do not name it. '
               'Return a JSON object with exactly these keys: primary_color, secondary_color, '
               'pattern_or_markings, surface_texture, overall_shape_or_form, distinctive_feature. '
               'Each value is a short phrase, or "not visible" if you cannot tell. Output only the JSON.')
JUDGE_PROMPT = ('You are comparing two descriptions of an object\'s {slot}. A: "{x}". B: "{y}". '
                "Do A and B describe essentially the same {slot}? Answer with only 'yes' or 'no'.")

def _parse_json(text):
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def read_slots(image_path, asker):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    obj = _parse_json(asker.ask(img, READ_PROMPT))
    return {slot: (str(obj[slot]) if slot in obj and str(obj[slot]).strip() else "not visible")
            for slot in SLOTS}

def _yes(r): return str(r).strip().lower().startswith("yes")

def judge_match(x, y, slot, asker):
    if norm(x) in NOT_VISIBLE or norm(y) in NOT_VISIBLE:
        return False
    from PIL import Image
    blank = Image.new("RGB", (8, 8), "white")
    a = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=x, y=y))
    b = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=y, y=x))
    return _yes(a) and _yes(b)
