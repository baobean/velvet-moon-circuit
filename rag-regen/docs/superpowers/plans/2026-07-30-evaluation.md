# Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a finished pipeline run into a defensible number: does reference-guided repair beat the un-retrieved draft?

**Architecture:** `ragregen/metrics.py` holds pure metric functions plus the three eval encoders; everything except the encoder loads runs on CPU with no model, so the arithmetic is testable on a machine whose GPU is held by someone else. `scripts/report.py` walks a run directory, buckets cases by their *draft verdict* (never by `status`), and writes `report.json` + `report.md`. The hold-out split that makes DINO honest is plumbed in `run_pipeline.py`'s oracle arm.

**Tech Stack:** Python 3.11, numpy, Pillow, transformers, pytest. No new dependencies.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Bare `python` is conda 3.13 and fails on faiss.
- **Repair rate comes from `scores.json`'s draft verdict, never from `status`.** `status: "unrepaired"` is written by two paths meaning opposite things (spec §4).
- **The held-out references are never edited with.** Oracle edits `gt_refs[:-h]`, DINO scores `gt_refs[-h:]`.
- **Round count clamps to `len(edit refs)`.** `run_pipeline.py:249` reads `refs[attempt - 1]`; an unclamped budget raises on a missing cutout and marks good cases `failed`.
- **Healthy cases have no `mask.png`** — they never ran the `mask` stage. Whole-image DINO in a separate column; never averaged into the cropped mean.
- **Eval encoders must differ from `retriever` and `crop_scorer`.** Config load fails otherwise.
- All metrics compute at the draft's size. Size mismatch raises; never resize.
- Verifier pass-rate appears nowhere near a headline.
- `git add <exact paths>` — never `git add -A`. `outputs/` is gitignored.
- The 358 existing tests must stay green.
- **GPU etiquette:** `nvidia-smi` before anything that loads an encoder. Kill background runs with `kill -9 $(pgrep -f 'bin/python scripts/run_pipeline.py')`, never `$!`.

---

### Task 1: `kind` on the case schema

**Files:**
- Modify: `ragregen/config.py`, `configs/dataset.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Case.kind: str` (`"target"` | `"control"`, default `"target"`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_kind_defaults_to_target(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    gt_refs: [x.jpg]
""")
    ds = config.load_dataset(p)
    assert ds.cases[0].kind == "target"


def test_kind_is_read_when_present(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    kind: control
    gt_refs: [x.jpg]
""")
    assert config.load_dataset(p).cases[0].kind == "control"


def test_an_unknown_kind_is_refused(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    kind: banana
    gt_refs: [x.jpg]
""")
    #: A typo here would silently move a case between report strata.
    with pytest.raises(ValueError, match="kind"):
        config.load_dataset(p)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_config.py -k kind -v`
Expected: FAIL — `AttributeError: 'Case' object has no attribute 'kind'`

- [ ] **Step 3: Implement**

In `ragregen/config.py`, add to the `Case` dataclass after `coarse`:

```python
    #: "target" = a fine-grained concept the method aims at; "control" = one
    #: FLUX should already render. If the controls fail too, the drafts are
    #: broken rather than the concepts rare -- which is what makes the fail
    #: rate interpretable (configs/dataset.yaml header).
    kind: str = "target"
```

`gt_refs` already has a default, so `kind` must be declared before it or given one — it has one, so order is free.

In `load_dataset`, where each `Case` is constructed, add validation and pass the field:

```python
KINDS = ("target", "control")

...
        kind = str(raw.get("kind", "target"))
        if kind not in KINDS:
            raise ValueError(
                f"case {raw['id']!r} has kind {kind!r}; expected one of {KINDS}")
```

and pass `kind=kind` into the `Case(...)` call.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_config.py -k kind -v`
Expected: 3 passed

- [ ] **Step 5: Label the 11 controls**

In `configs/dataset.yaml`, add `kind: control` to each of the 11 control cases —
`hedgehog`, `golden_retriever`, `panda`, `polar_bear`, `fox`, `flamingo`, `cactus`,
`stack_of_books`, `red_ferrari`, `sushi`, `violin` — placing the line after `coarse:`.
Verify the count:

```bash
grep -c 'kind: control' configs/dataset.yaml   # must print 11
$PY -c "
from ragregen import config
ds = config.load_dataset('configs/dataset.yaml')
from collections import Counter
print(Counter(c.kind for c in ds.cases))"
```

Expected: `Counter({'target': 11, 'control': 11})`

- [ ] **Step 6: Commit**

```bash
git add ragregen/config.py configs/dataset.yaml tests/test_config.py
git commit -m "feat: targets and controls are data, not a comment"
```

---

### Task 2: The `eval` config block and its two guards

**Files:**
- Modify: `ragregen/config.py`, `ragregen/validate.py`, `configs/pipeline.yaml`
- Test: `tests/test_config.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: `Case.kind` (Task 1)
- Produces: `PipelineConfig.eval_heldout_refs: int`, `.eval_dino: str`, `.eval_clip: str`, `.eval_siglip: str`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_eval_block_has_defaults(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n")
    cfg = config.load_pipeline(p)
    assert cfg.eval_heldout_refs == 1
    assert "dinov3" in cfg.eval_dino


def test_an_eval_encoder_may_not_equal_the_retriever(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n"
                 "eval:\n  siglip: siglip_so400m_384\n")
    #: Scoring an output with the model that chose its reference is
    #: self-marking. It must be impossible to configure, not merely discouraged.
    with pytest.raises(ValueError, match="self-marking|eval encoder"):
        config.load_pipeline(p)


def test_heldout_refs_must_leave_something_to_edit_with(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n"
                 "eval:\n  heldout_refs: 0\n")
    with pytest.raises(ValueError, match="heldout_refs"):
        config.load_pipeline(p)
```

Add to `tests/test_validate.py`:

```python
def test_a_case_with_too_few_refs_for_the_holdout_is_reported():
    from ragregen import validate
    problems = validate.check_heldout_refs(
        [("thin", 1), ("fat", 3)], heldout_refs=1)
    #: 1 ref and h=1 leaves zero to edit with. Catch it in validate, not an
    #: hour into an encoding run.
    assert any("thin" in p for p in problems)
    assert not any("fat" in p for p in problems)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_config.py -k eval tests/test_validate.py -k heldout -v`
Expected: FAIL — `AttributeError: 'PipelineConfig' object has no attribute 'eval_heldout_refs'`

- [ ] **Step 3: Implement**

In `ragregen/config.py`, add to `PipelineConfig`:

```python
    eval_heldout_refs: int = 1
    eval_dino: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    eval_clip: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    eval_siglip: str = "google/siglip-base-patch16-384"
```

In `load_pipeline`, after the existing fields are read:

```python
    ev = raw.get("eval") or {}
    heldout = int(ev.get("heldout_refs", 1))
    if heldout < 1:
        raise ValueError(
            f"eval.heldout_refs is {heldout}; it must be >= 1, or DINO scores "
            f"the oracle arm against the very images it edited with.")

    eval_dino = str(ev.get("dino", PipelineConfig.eval_dino))
    eval_clip = str(ev.get("clip", PipelineConfig.eval_clip))
    eval_siglip = str(ev.get("siglip", PipelineConfig.eval_siglip))

    #: The paper retrieves with one checkpoint and evaluates with another on
    #: purpose (design §5). Scoring an output with the model that chose its
    #: reference is self-marking.
    in_loop = {str(retriever), str(crop_scorer)}
    for label, value in (("dino", eval_dino), ("clip", eval_clip),
                         ("siglip", eval_siglip)):
        if value in in_loop:
            raise ValueError(
                f"eval.{label} is {value!r}, which is also the retriever or "
                f"crop_scorer. That is self-marking -- pick a different "
                f"checkpoint for evaluation.")
```

Pass all four into the `PipelineConfig(...)` construction.

In `ragregen/validate.py`, add:

```python
def check_heldout_refs(case_ref_counts, heldout_refs: int) -> list[str]:
    """Cases that cannot afford the hold-out.

    Takes (case_id, n_refs) pairs rather than a dataset so it stays testable
    without touching disk.
    """
    problems = []
    for case_id, n in case_ref_counts:
        if n - heldout_refs < 1:
            problems.append(
                f"case {case_id!r} has {n} gt_refs; holding out "
                f"{heldout_refs} leaves nothing to edit with. Add a "
                f"reference or lower eval.heldout_refs.")
    return problems
```

In `configs/pipeline.yaml`, append:

```yaml

eval:                    # doc 4. Must differ from retriever/crop_scorer above.
  heldout_refs: 1        # refs reserved for DINO; never edited with
  dino: facebook/dinov3-vitl16-pretrain-lvd1689m
  clip: laion/CLIP-ViT-L-14-laion2B-s32B-b82K
  siglip: google/siglip-base-patch16-384
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_config.py tests/test_validate.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add ragregen/config.py ragregen/validate.py configs/pipeline.yaml tests/test_config.py tests/test_validate.py
git commit -m "feat: the eval block, and self-marking made unconfigurable"
```

---

### Task 3: `crop_to_mask`

**Files:**
- Create: `ragregen/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `crop_to_mask(image, mask, pad_frac=0.15, min_side=64) -> Image`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Metric arithmetic, proved without a model.

Every test here runs on synthetic arrays. The GPU on this machine is routinely
held by someone else (findings/2026-07-28-regeneration-result.md §4), and a
metric you cannot check on CPU is a metric you stop checking.
"""
import numpy as np
import pytest
from PIL import Image

from ragregen.metrics import crop_to_mask


def _img(w, h, colour=(10, 20, 30)):
    return Image.new("RGB", (w, h), colour)


def _mask(w, h, box):
    a = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = 255
    return Image.fromarray(a, "L")


def test_crops_to_the_box_plus_padding():
    img = _img(400, 400)
    crop = crop_to_mask(img, _mask(400, 400, (100, 100, 300, 300)))

    #: Filled pixels span 100..299, so the bbox width is 199 (not 200).
    #: pad = int(199 * 0.15) + 8 = 37, giving (63, 63, 336, 336).
    assert crop.size == (273, 273)


def test_padding_clamps_at_the_frame_edge():
    img = _img(200, 200)
    crop = crop_to_mask(img, _mask(200, 200, (0, 0, 100, 100)))

    #: The box touches the origin; padding must not produce negative
    #: coordinates or PIL silently returns an empty image.
    assert crop.size[0] > 0 and crop.size[1] > 0
    assert crop.size[0] <= 200 and crop.size[1] <= 200


def test_an_empty_mask_falls_back_to_the_full_frame():
    img = _img(120, 90)
    assert crop_to_mask(img, _mask(120, 90, (0, 0, 0, 0))).size == (120, 90)


def test_a_tiny_box_falls_back_rather_than_scoring_a_speck():
    img = _img(400, 400)
    #: A 4px crop carries no identity signal; scoring it would be noise
    #: dressed as a measurement.
    assert crop_to_mask(img, _mask(400, 400, (10, 10, 14, 14))).size == (400, 400)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.metrics'`

- [ ] **Step 3: Implement**

Create `ragregen/metrics.py`:

```python
"""Evaluation metrics. Doc 4.

Pure functions plus the three eval encoders. Nothing here runs inside the
repair loop -- selection uses the verifier only, and selecting on these would
be selecting on the thing being measured (design §5).

Ported from ImageRAG/scripts/kontext_metrics.py so numbers stay comparable
between the two projects' runs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop_to_mask(image: Image.Image, mask, pad_frac: float = 0.15,
                 min_side: int = 64) -> Image.Image:
    """Crop to the mask bbox + padding. Full frame if degenerate.

    Identity is a property of the object, not the canvas: a rare parrot in a
    wide scene scores against its own background unless the crop is taken
    (findings/2026-07-27-finegrained-result.md §3a).
    """
    a = (np.array(mask.convert("L")) if isinstance(mask, Image.Image)
         else np.asarray(mask))
    thr = 127 if a.dtype == np.uint8 else 0.5
    ys, xs = np.where(a > thr)
    if xs.size == 0:
        return image
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pw = int((x1 - x0) * pad_frac) + 8
    ph = int((y1 - y0) * pad_frac) + 8
    crop = image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(image.width, x1 + pw), min(image.height, y1 + ph)))
    if min(crop.size) < min_side:
        return image
    return crop
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/metrics.py tests/test_metrics.py
git commit -m "feat: crop to the object, because identity is not a property of the canvas"
```

---

### Task 4: `preservation`, the anti-gaming guard

**Files:**
- Modify: `ragregen/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `preservation(draft, output, alpha) -> dict` with keys `score`, `outside_l1`, `outside_max`, `outside_frac`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from ragregen.metrics import preservation


def _alpha(w, h, box):
    a = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = 1.0
    return a


def test_an_untouched_output_preserves_perfectly():
    draft = _img(64, 64)
    assert preservation(draft, draft.copy(),
                        _alpha(64, 64, (16, 16, 48, 48)))["score"] == 1.0


def test_changes_inside_the_mask_do_not_count_against_preservation():
    draft = _img(64, 64)
    out = np.array(draft).copy()
    out[16:48, 16:48] = (255, 0, 0)          # repaint only inside
    got = preservation(draft, Image.fromarray(out, "RGB"),
                       _alpha(64, 64, (16, 16, 48, 48)))

    #: This is the whole guarantee of the compositor: outside the mask is
    #: bit-identical, so the metric must read exactly 1.0, not 0.99.
    assert got["score"] == 1.0
    assert got["outside_l1"] == 0.0


def test_repainting_the_canvas_destroys_the_score():
    draft = _img(64, 64, (0, 0, 0))
    out = _img(64, 64, (255, 255, 255))
    got = preservation(draft, out, _alpha(64, 64, (16, 16, 48, 48)))

    #: High identity bought by repainting everything is not a repair.
    assert got["score"] == pytest.approx(0.0, abs=1e-6)
    assert got["outside_max"] == 255.0


def test_a_fully_masked_frame_has_no_outside_to_judge():
    draft = _img(32, 32)
    got = preservation(draft, draft.copy(), _alpha(32, 32, (0, 0, 32, 32)))
    assert np.isnan(got["score"])


def test_a_size_mismatch_raises_rather_than_resizing():
    #: Resampling artifacts read as real deltas (design §5). Refuse.
    with pytest.raises(ValueError, match="size"):
        preservation(_img(64, 64), _img(32, 32), _alpha(64, 64, (0, 0, 8, 8)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_metrics.py -k preserv -v`
Expected: FAIL — `ImportError: cannot import name 'preservation'`

- [ ] **Step 3: Implement**

Add to `ragregen/metrics.py`:

```python
def preservation(draft: Image.Image, output: Image.Image,
                 alpha: np.ndarray) -> dict:
    """How much of the region outside the mask survived, 0-1, higher better.

    `score` is 1 - mean|diff|/255 over the alpha==0 zone. For a composite it
    must be EXACTLY 1.0 -- pixels outside the mask are bit-identical by
    construction, and anything less means the compositor changed behaviour.

    Always read next to DINO: a method that improves identity by repainting
    the whole canvas has not solved the problem (RUNBOOK §5.2).
    """
    if output.size != draft.size:
        raise ValueError(
            f"output size {output.size} != draft size {draft.size}. All "
            f"metrics compute at one pinned resolution; resizing here would "
            f"show resampling artifacts as real deltas.")

    o = np.array(draft.convert("RGB")).astype(np.float32)
    r = np.array(output.convert("RGB")).astype(np.float32)

    outside = alpha == 0.0
    if not outside.any():
        return {"score": float("nan"), "outside_l1": float("nan"),
                "outside_max": float("nan"), "outside_frac": 0.0}

    d = np.abs(o - r)[outside]
    return {
        "score": float(1.0 - d.mean() / 255.0),
        "outside_l1": float(d.mean()),
        "outside_max": float(d.max()),
        "outside_frac": float(outside.mean()),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/metrics.py tests/test_metrics.py
git commit -m "feat: preservation, so identity bought by repainting cannot pass"
```

---

### Task 5: The hold-out split and round clamping

**Files:**
- Modify: `ragregen/metrics.py`, `scripts/run_pipeline.py`
- Test: `tests/test_metrics.py`, `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `PipelineConfig.eval_heldout_refs` (Task 2)
- Produces: `split_refs(gt_refs, heldout) -> tuple[list, list]`

This is the task that makes DINO honest, and the one most easily got wrong: shrinking the edit pool without clamping the rounds turns a hygiene fix into fabricated failures.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from ragregen.metrics import split_refs


def test_the_last_ref_is_held_out():
    edit, held = split_refs(["a", "b", "c"], heldout=1)
    assert edit == ["a", "b"]
    assert held == ["c"]


def test_the_held_out_ref_is_never_in_the_edit_pool():
    for n in (3, 4, 5):
        refs = [f"r{i}" for i in range(n)]
        edit, held = split_refs(refs, heldout=1)
        #: The entire point. If these overlap, DINO scores an edit against the
        #: image it was given.
        assert not (set(edit) & set(held))
        assert edit + held == refs


def test_holding_out_everything_is_refused():
    with pytest.raises(ValueError, match="nothing to edit"):
        split_refs(["a"], heldout=1)
```

Add to `tests/test_run_pipeline.py`:

```python
def test_rounds_clamp_to_the_edit_pool(tmp_path):
    """A 2-ref edit pool with retry_budget 3 must run 2 rounds, not 3.

    run_pipeline.py reads refs[attempt - 1] and _regen_one raises when the
    cutout is missing, so an unclamped budget marks a perfectly good case
    `failed` -- a hygiene fix that fabricates failures.
    """
    from scripts.run_pipeline import rounds_for
    assert rounds_for(retry_budget=3, n_edit_refs=2) == 2
    assert rounds_for(retry_budget=3, n_edit_refs=3) == 3
    assert rounds_for(retry_budget=2, n_edit_refs=5) == 2


def test_a_case_out_of_references_is_retired_not_failed(tmp_path):
    """A short edit pool must end the case, not manufacture a failure.

    This is the whole risk of the hold-out: _regen_one raises when
    cutout_<attempt>.png is absent, and run_stage turns that into
    `status: failed` -- a dataset with fewer references would read as a method
    that could not repair it.
    """
    import json

    from ragregen.schedule import Queue
    from scripts.run_pipeline import _retire_exhausted

    class _RunDir:
        def __init__(self, root):
            self.root = root

        def case_dir(self, cid):
            d = self.root / cid
            d.mkdir(parents=True, exist_ok=True)
            return d

    rd = _RunDir(tmp_path)
    q = Queue.open(tmp_path / "q.json", ["short"], retry_budget=3,
                   screen_run="s")
    (rd.case_dir("short") / "refs.json").write_text(json.dumps(["a", "b"]))
    (rd.case_dir("short") / "scores.json").write_text(
        json.dumps({"draft": [False, 0.0], "attempt_1": [False, 0.0],
                    "attempt_2": [False, 0.0]}))

    _retire_exhausted(q, rd, attempt=3)

    assert q.cases["short"].status != "failed"
    assert q.cases["short"].status == "unrepaired"
    assert q.cases["short"].best == "draft"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_metrics.py -k split tests/test_run_pipeline.py -k clamp -v`
Expected: FAIL — `ImportError: cannot import name 'split_refs'`

- [ ] **Step 3: Implement**

Add to `ragregen/metrics.py`:

```python
def split_refs(gt_refs, heldout: int = 1):
    """(edit pool, held-out) -- the split that makes DINO reportable.

    The oracle arm regenerates using its references and DINO scores the output
    against them. Scoring an edit with the image it was handed is self-marking,
    the same objection RUNBOOK line 247 makes about --proto-arm ceiling. The
    last `heldout` references are reserved for scoring and never edited with.
    """
    refs = list(gt_refs)
    if len(refs) - heldout < 1:
        raise ValueError(
            f"{len(refs)} refs with heldout={heldout} leaves nothing to edit "
            f"with. Add a reference or lower eval.heldout_refs.")
    return refs[:-heldout], refs[-heldout:]
```

Add to `scripts/run_pipeline.py`, near the top-level helpers:

```python
def rounds_for(retry_budget: int, n_edit_refs: int) -> int:
    """How many repair rounds this case can actually afford.

    Attempt N edits with refs[N - 1] and _regen_one raises when the cutout is
    absent, so a budget larger than the edit pool does not produce more
    attempts -- it produces spurious `failed` cases.
    """
    return min(int(retry_budget), int(n_edit_refs))
```

In `main()`'s oracle branch, replace the `refs` line so the edit pool is what
reaches disk, and record the held-out set beside it:

```python
    if args.arm == "oracle":
        for cid in list(queue.pending("retrieve")):
            edit, held = metrics.split_refs(by_id[cid].gt_refs,
                                            pipe_cfg.eval_heldout_refs)
            d = run_dir.case_dir(cid)
            (d / "refs.json").write_text(json.dumps([str(p) for p in edit]))
            #: The scoring target, written once so the report never has to
            #: re-derive the split and risk disagreeing with the run.
            (d / "heldout_refs.json").write_text(
                json.dumps([str(p) for p in held]))
            queue.mark(cid, "retrieve", "done")
```

Add `metrics` to the `from ragregen import (...)` line.

The round loop is global — `for attempt in range(1, pipe_cfg.retry_budget + 1)` —
but the edit pool now differs per case: a 3-ref case has 2 references, the
5-ref case has 4. A single global bound would ask some cases for an attempt
whose reference does not exist. Retire those cases instead of letting
`_regen_one` raise.

First extract the per-case half of `_finalise` so it can be called for one
case. Replace the body of `_finalise` with:

```python
def _finalise_one(queue, run_dir, case_id: str) -> None:
    """Pick the candidate to report for one case."""
    path = run_dir.case_dir(case_id) / "scores.json"
    if not path.is_file():
        return
    scored = json.loads(path.read_text())
    draft_ok, draft_score = scored.get("draft", [False, 0.0])
    attempts = [(label, bool(v[0]), float(v[1]))
                for label, v in sorted(scored.items()) if label != "draft"]
    best = schedule.select_best(float(draft_score), attempts)
    queue.cases[case_id].best = best
    queue.cases[case_id].status = ("unrepaired" if best == "draft"
                                   else "repaired")


def _finalise(queue, run_dir) -> None:
    for case_id in queue.cases:
        _finalise_one(queue, run_dir, case_id)
    queue.save()
```

Then add the retirement helper:

```python
def _refs_of(run_dir, case_id) -> list:
    p = run_dir.case_dir(case_id) / "refs.json"
    return json.loads(p.read_text()) if p.is_file() else []


def _retire_exhausted(queue, run_dir, attempt: int) -> None:
    """End the run for cases with no reference left for this attempt.

    Attempt N edits with refs[N - 1]. A case whose edit pool is shorter than
    `attempt` has nothing to try, and _regen_one would raise on the missing
    cutout -- which run_stage marks `failed`, inventing a failure out of a
    dataset that simply had fewer references.
    """
    for cid in list(queue.pending("regen", attempt=attempt)):
        if attempt > len(_refs_of(run_dir, cid)):
            _finalise_one(queue, run_dir, cid)
    queue.save()
```

And bound the loop, retiring first so `needs_round` sees the truth:

```python
    for attempt in range(1, rounds_for(pipe_cfg.retry_budget,
                                       max((len(_refs_of(run_dir, c))
                                            for c in queue.cases), default=0)) + 1):
        _retire_exhausted(queue, run_dir, attempt)
        if not queue.needs_round(attempt):
            break
```

leaving the rest of the loop body unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_metrics.py tests/test_run_pipeline.py -v`
Expected: all pass

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest -q`
Expected: all green. Task 5 changes the pipeline, so a regression here is a real one.

- [ ] **Step 6: Commit**

```bash
git add ragregen/metrics.py scripts/run_pipeline.py tests/test_metrics.py tests/test_run_pipeline.py
git commit -m "feat: hold a reference back, and clamp the rounds that would miss it"
```

---

### Task 6: The eval encoders

**Files:**
- Modify: `ragregen/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `PipelineConfig.eval_*` (Task 2)
- Produces: `DinoEncoder`, `dino_identity(crop, ref_crops, encoder) -> float`, `prompt_alignment(image, prompt, encoder) -> float`, `EvalEncoders`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from ragregen.metrics import EvalEncoders, dino_identity, prompt_alignment


class _FakeVisionEncoder:
    """Stands in for DINOv3. Returns a fixed vector per image size."""

    def __init__(self):
        self.freed = False
        self.seen = []

    def embed(self, image):
        self.seen.append(image.size)
        v = np.array([image.width, image.height], dtype=np.float32)
        return v / np.linalg.norm(v)

    def free(self):
        self.freed = True


def test_dino_identity_averages_over_every_held_out_ref():
    enc = _FakeVisionEncoder()
    out = _img(10, 10)
    refs = [_img(10, 10), _img(10, 10)]
    assert dino_identity(out, refs, enc) == pytest.approx(1.0)
    #: One forward per image: the output plus each reference.
    assert len(enc.seen) == 3


def test_dino_identity_needs_at_least_one_reference():
    with pytest.raises(ValueError, match="no held-out"):
        dino_identity(_img(10, 10), [], _FakeVisionEncoder())


class _FakeDualEncoder:
    def encode_pil(self, images, batch_size=32):
        return np.array([[1.0, 0.0]] * len(list(images)), dtype=np.float32)

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(list(texts)), dtype=np.float32)


def test_prompt_alignment_is_a_cosine():
    assert prompt_alignment(_img(8, 8), "a parrot",
                            _FakeDualEncoder()) == pytest.approx(1.0)


def test_eval_encoders_free_everything_they_loaded():
    a, b = _FakeVisionEncoder(), _FakeVisionEncoder()
    holder = EvalEncoders(dino=a, clip=b, siglip=None)
    holder.free()
    #: One load per report run, freed together -- the residency discipline
    #: stage_with_model established in doc 3.
    assert a.freed and b.freed
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_metrics.py -k "dino or prompt or eval_enc" -v`
Expected: FAIL — `ImportError: cannot import name 'EvalEncoders'`

- [ ] **Step 3: Implement**

Add to `ragregen/metrics.py`:

```python
class DinoEncoder:
    """DINOv3. Instance-level identity -- the headline metric.

    Loaded lazily so importing this module costs nothing on a busy card.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.proc = None

    def _load(self):
        from transformers import AutoImageProcessor, AutoModel

        from ragregen import env
        self.proc = AutoImageProcessor.from_pretrained(
            self.model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            self.model_id, cache_dir=str(env.HF_CACHE)).to(self.device).eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        import torch

        if self.model is None:
            self._load()
        inputs = self.proc(images=image.convert("RGB"),
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # pooler_output is the CLS token; mean-pool the patches when a
        # checkpoint exposes no pooler.
        feats = (out.pooler_output if getattr(out, "pooler_output", None)
                 is not None else out.last_hidden_state.mean(dim=1))
        v = feats.float().cpu().numpy()[0]
        return v / (np.linalg.norm(v) + 1e-12)

    def free(self):
        import torch

        self.model = None
        self.proc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dino_identity(output_crop: Image.Image, ref_crops, encoder) -> float:
    """Mean cosine between the output crop and each HELD-OUT reference crop.

    The references here must be the ones the repair never saw. Passing the
    edit pool makes this self-marking (spec §2).
    """
    refs = list(ref_crops)
    if not refs:
        raise ValueError(
            "no held-out references to score against; check "
            "heldout_refs.json for this case")
    o = encoder.embed(output_crop)
    return float(np.mean([float(o @ encoder.embed(r)) for r in refs]))


def prompt_alignment(image: Image.Image, prompt: str, encoder) -> float:
    """Cosine between the image and its prompt, for CLIP and SigLIP alike.

    Context, never the claim: prompt alignment cannot tell an Amur leopard
    from a generic one, which is the entire premise of the project. A flat
    CLIP delta beside a clear DINO gain is a success (RUNBOOK §5.2).
    """
    iv = np.asarray(encoder.encode_pil([image]), dtype=np.float32)[0]
    tv = np.asarray(encoder.encode_text([prompt]), dtype=np.float32)[0]
    iv = iv / (np.linalg.norm(iv) + 1e-12)
    tv = tv / (np.linalg.norm(tv) + 1e-12)
    return float(iv @ tv)


class EvalEncoders:
    """The three eval encoders, loaded once per report run and freed together."""

    def __init__(self, dino=None, clip=None, siglip=None):
        self.dino = dino
        self.clip = clip
        self.siglip = siglip

    @classmethod
    def load(cls, pipe_cfg, device: str = "cuda") -> "EvalEncoders":
        from ragregen import encoders

        return cls(
            dino=DinoEncoder(pipe_cfg.eval_dino, device),
            clip=encoders.HFEncoder(pipe_cfg.eval_clip, device),
            siglip=encoders.HFEncoder(pipe_cfg.eval_siglip, device),
        )

    def free(self):
        for enc in (self.dino, self.clip, self.siglip):
            free = getattr(enc, "free", None)
            if callable(free):
                free()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/metrics.py tests/test_metrics.py
git commit -m "feat: the three eval encoders, deliberately not the retrieval one"
```

---

### Task 7: Bucketing — the denominator that is not `status`

**Files:**
- Create: `scripts/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing
- Produces: `bucket(scores) -> str` (`"healthy"` | `"repaired"` | `"unrepairable"`), `repair_rate(buckets) -> float | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
"""Report arithmetic. No models, no run directory -- just the bookkeeping.

The bug this file exists to prevent is documented in
findings/2026-07-29-orchestration-result.md §2: `status` says "unrepaired" both
when a case never needed repair and when repair failed.
"""
import pytest

from scripts.report import bucket, repair_rate


def test_a_passing_draft_is_healthy():
    assert bucket({"draft": [True, 1.0]}) == "healthy"


def test_a_failing_draft_beaten_by_an_attempt_is_repaired():
    assert bucket({"draft": [False, 0.0],
                   "attempt_1": [True, 1.0]}) == "repaired"


def test_a_failing_draft_nothing_beat_is_unrepairable():
    assert bucket({"draft": [False, 0.0],
                   "attempt_1": [False, 0.0],
                   "attempt_2": [False, 0.0]}) == "unrepairable"


def test_repair_rate_excludes_healthy_cases_from_the_denominator():
    buckets = ["healthy", "repaired", "unrepairable"]
    #: 1 of the 2 that needed repair. `status` would give 1/3 or 2/3 -- both
    #: wrong, and both plausible enough to publish.
    assert repair_rate(buckets) == pytest.approx(0.5)


def test_repair_rate_is_undefined_when_nothing_needed_repair():
    #: Not 0.0. A rate over an empty denominator is not a number, and
    #: printing 0% would read as total failure.
    assert repair_rate(["healthy", "healthy"]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.report'`

- [ ] **Step 3: Implement**

Create `scripts/report.py`:

```python
#!/usr/bin/env python
"""Turn a finished pipeline run into the results table. Doc 4.

Buckets come from scores.json's DRAFT VERDICT, never from queue.json's
`status`, which is written by two paths meaning opposite things
(findings/2026-07-29-orchestration-result.md §2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402


def bucket(scores: dict) -> str:
    """Which of the three groups this case belongs to.

    healthy       the draft passed; there was nothing to repair
    repaired      the draft failed and some attempt beat it
    unrepairable  the draft failed and nothing beat it
    """
    draft_ok = bool(scores.get("draft", [False, 0.0])[0])
    if draft_ok:
        return "healthy"
    attempts = {k: v for k, v in scores.items() if k != "draft"}
    best_attempt = max((float(v[1]) for v in attempts.values()), default=0.0)
    draft_score = float(scores.get("draft", [False, 0.0])[1])
    return "repaired" if best_attempt > draft_score else "unrepairable"


def repair_rate(buckets) -> float | None:
    """repaired / (repaired + unrepairable), or None if nothing needed repair.

    Healthy cases are excluded from the denominator. Including them is the
    mistake `status` invites, and it inflates the rate.
    """
    needed = [b for b in buckets if b in ("repaired", "unrepairable")]
    if not needed:
        return None
    return sum(1 for b in needed if b == "repaired") / len(needed)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_report.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat: bucket on the draft verdict, not on the overloaded status"
```

---

### Task 8: Scoring one case

**Files:**
- Modify: `scripts/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `bucket` (Task 7), `crop_to_mask`/`preservation`/`dino_identity`/`prompt_alignment` (Tasks 3–6)
- Produces: `score_case(case, case_dir, draft, encoders, *, heldout_paths) -> dict`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py`:

```python
import json

import numpy as np
from PIL import Image

from scripts.report import score_case


class _Enc:
    def embed(self, image):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_pil(self, images, batch_size=32):
        return np.array([[1.0, 0.0]] * len(list(images)), dtype=np.float32)

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(list(texts)), dtype=np.float32)

    def free(self):
        pass


class _Holder:
    def __init__(self):
        self.dino = _Enc()
        self.clip = _Enc()
        self.siglip = _Enc()


class _Case:
    id = "c"
    prompt = "a parrot"
    kind = "target"


def _write_case(d, scores, *, with_mask, ref):
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(scores))
    Image.new("RGB", (64, 64), (10, 20, 30)).save(d / "attempt_1.png")
    if with_mask:
        a = np.zeros((64, 64), dtype=np.uint8)
        a[8:56, 8:56] = 255
        Image.fromarray(a, "L").save(d / "mask.png")
    Image.new("RGB", (64, 64), (10, 20, 30)).save(ref)


def test_a_repaired_case_is_scored_on_the_crop(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 0.0], "attempt_1": [True, 1.0]},
                with_mask=True, ref=ref)

    got = score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref])

    assert got["bucket"] == "repaired"
    assert got["dino_cropped"] == pytest.approx(1.0)
    assert got["dino_whole"] is None
    assert got["preservation"] == 1.0


def test_a_healthy_case_has_no_mask_and_is_scored_whole(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [True, 1.0]}, with_mask=False, ref=ref)

    got = score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref])

    #: It never ran the `mask` stage, so there is no box to crop to. Its DINO
    #: goes in a separate column rather than into the cropped mean.
    assert got["bucket"] == "healthy"
    assert got["dino_cropped"] is None
    assert got["dino_whole"] == pytest.approx(1.0)


def test_a_failed_draft_without_a_mask_raises(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 0.0], "attempt_1": [True, 1.0]},
                with_mask=False, ref=ref)

    #: This case should have run `mask`. A missing one means the run is
    #: damaged, and quietly scoring it whole-image would hide that.
    with pytest.raises(FileNotFoundError, match="mask"):
        score_case(_Case(), d, Image.new("RGB", (64, 64)), _Holder(),
                   heldout_paths=[ref])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_report.py -k score_case -v`
Expected: FAIL — `ImportError: cannot import name 'score_case'`

- [ ] **Step 3: Implement**

Add to `scripts/report.py`:

```python
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ragregen import metrics  # noqa: E402


def _selected_image(case_dir: Path, scores: dict, draft: Image.Image):
    """The candidate the pipeline chose. The draft when nothing beat it."""
    attempts = {k: v for k, v in scores.items() if k != "draft"}
    if not attempts:
        return draft, "draft"
    label = max(attempts, key=lambda k: float(attempts[k][1]))
    path = case_dir / f"{label}.png"
    if not path.is_file():
        return draft, "draft"
    return Image.open(path).convert("RGB"), label


def score_case(case, case_dir: Path, draft: Image.Image, encoders, *,
               heldout_paths) -> dict:
    """Every metric for one case, or an explanation of why not.

    Healthy cases never ran `mask`, so they have no bounding box. They are
    scored whole-image in their own column rather than being averaged into a
    cropped mean -- comparing a tight object crop against a whole scene and
    reading the difference as identity (spec §4).
    """
    scores = json.loads((case_dir / "scores.json").read_text())
    group = bucket(scores)
    output, label = _selected_image(case_dir, scores, draft)

    if output.size != draft.size:
        raise ValueError(
            f"{case.id}: output size {output.size} != draft {draft.size}")

    mask_path = case_dir / "mask.png"
    refs = [Image.open(p).convert("RGB") for p in heldout_paths]

    row = {
        "case_id": case.id,
        "kind": case.kind,
        "bucket": group,
        "selected": label,
        "dino_cropped": None,
        "dino_whole": None,
        "preservation": None,
        "clip": metrics.prompt_alignment(output, case.prompt, encoders.clip),
        "siglip": metrics.prompt_alignment(output, case.prompt,
                                           encoders.siglip),
    }

    if not mask_path.is_file():
        if group != "healthy":
            raise FileNotFoundError(
                f"{case.id}: no mask.png, but its draft failed -- this case "
                f"should have run the `mask` stage. The run is damaged.")
        row["dino_whole"] = metrics.dino_identity(output, refs, encoders.dino)
        return row

    mask = Image.open(mask_path).convert("L")
    alpha = (np.array(mask).astype(np.float32) / 255.0 > 0.5).astype(
        np.float32)
    row["dino_cropped"] = metrics.dino_identity(
        metrics.crop_to_mask(output, mask),
        [metrics.crop_to_mask(r, mask) if r.size == mask.size else r
         for r in refs],
        encoders.dino)
    row["preservation"] = metrics.preservation(draft, output, alpha)["score"]
    return row
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_report.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat: score a case, and refuse to score a damaged one quietly"
```

---

### Task 9: The report, stratified

**Files:**
- Modify: `scripts/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `score_case` (Task 8), `repair_rate` (Task 7)
- Produces: `summarise(rows) -> dict`, `render(summary, meta) -> str`, `main()`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py`:

```python
from scripts.report import render, summarise


def _row(kind, group, dino=None, whole=None, pres=None):
    return {"case_id": "x", "kind": kind, "bucket": group, "selected": "d",
            "dino_cropped": dino, "dino_whole": whole, "preservation": pres,
            "clip": 0.3, "siglip": 0.4}


def test_summary_splits_targets_from_controls():
    rows = [_row("target", "repaired", dino=0.8, pres=1.0),
            _row("control", "healthy", whole=0.9)]
    got = summarise(rows)

    assert got["target"]["n"] == 1
    assert got["control"]["n"] == 1
    assert got["target"]["dino_cropped"] == pytest.approx(0.8)


def test_whole_image_dino_never_enters_the_cropped_mean():
    rows = [_row("target", "repaired", dino=0.8, pres=1.0),
            _row("target", "healthy", whole=0.2)]
    got = summarise(rows)

    #: 0.8, not 0.5. Averaging a whole-scene score into an object-crop mean
    #: compares two different measurements.
    assert got["target"]["dino_cropped"] == pytest.approx(0.8)
    assert got["target"]["dino_whole"] == pytest.approx(0.2)


def test_render_states_the_mechanism_and_refuses_a_pass_rate_headline():
    text = render(summarise([_row("target", "repaired", dino=0.8, pres=1.0)]),
                  {"mechanism": "stitch", "arm": "oracle", "run": "r",
                   "heldout_refs": 1})
    assert "stitch" in text
    #: The pipeline optimises against the verifier, so its pass-rate is a
    #: development signal only (RUNBOOK §5.2).
    assert "pass-rate" not in text.lower() or "development signal" in text.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_report.py -k "summar or render" -v`
Expected: FAIL — `ImportError: cannot import name 'summarise'`

- [ ] **Step 3: Implement**

Add to `scripts/report.py`:

```python
def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarise(rows) -> dict:
    """Per-stratum means. Cropped and whole-image DINO stay separate."""
    out = {}
    for kind in ("target", "control"):
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        out[kind] = {
            "n": len(group),
            "healthy": sum(1 for r in group if r["bucket"] == "healthy"),
            "repaired": sum(1 for r in group if r["bucket"] == "repaired"),
            "unrepairable": sum(1 for r in group
                                if r["bucket"] == "unrepairable"),
            "repair_rate": repair_rate([r["bucket"] for r in group]),
            "dino_cropped": _mean(r["dino_cropped"] for r in group),
            "dino_whole": _mean(r["dino_whole"] for r in group),
            "preservation": _mean(r["preservation"] for r in group),
            "clip": _mean(r["clip"] for r in group),
            "siglip": _mean(r["siglip"] for r in group),
        }
    return out


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def render(summary: dict, meta: dict) -> str:
    """The human-readable table, with its own caveats attached."""
    lines = [
        f"# Results — {meta['run']}",
        "",
        f"Arm: **{meta['arm']}**  ·  mechanism: **{meta['mechanism']}**  ·  "
        f"held-out refs: **{meta['heldout_refs']}**",
        "",
        "| stratum | n | healthy | repaired | unrepairable | repair rate | "
        "DINO (crop) | DINO (whole) | preservation | CLIP | SigLIP |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, s in summary.items():
        rate = "—" if s["repair_rate"] is None else f"{s['repair_rate']:.0%}"
        lines.append(
            f"| {kind} | {s['n']} | {s['healthy']} | {s['repaired']} | "
            f"{s['unrepairable']} | {rate} | {_fmt(s['dino_cropped'])} | "
            f"{_fmt(s['dino_whole'])} | {_fmt(s['preservation'])} | "
            f"{_fmt(s['clip'])} | {_fmt(s['siglip'])} |")

    lines += [
        "",
        "## How to read this",
        "",
        "- **DINO (crop) is the claim.** CLIP and SigLIP are context: prompt "
        "alignment cannot tell a rare concept from its coarse term, which is "
        "the premise of the project. A flat CLIP delta beside a clear DINO "
        "gain is a success.",
        "- **Read preservation next to DINO.** Identity bought by repainting "
        "the canvas is not a repair.",
        "- **DINO (whole) is not comparable to DINO (crop).** It covers "
        "healthy cases, which never ran `mask` and so have no box to crop to.",
        "- **Verifier pass-rate is absent on purpose.** The pipeline "
        "optimises against the verifier, so it is a development signal only.",
        f"- Controls should start high and move little. If they gain as much "
        f"as targets, the metric is reading 'an edit happened', not identity.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    from ragregen import config, trace

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="an outputs/pipeline_<TS>/ to report on")
    ap.add_argument("--screen-run", type=Path,
                    default=Path("outputs/screen_latest"))
    ap.add_argument("--dataset", type=Path,
                    default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path,
                    default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    run_info = json.loads((args.run / "run.json").read_text())

    encoders = metrics.EvalEncoders.load(pipe_cfg, args.device)
    rows, skipped = [], []
    try:
        for case in ds.cases:
            d = args.run / case.id
            if not (d / "scores.json").is_file():
                skipped.append({"case_id": case.id, "why": "no output"})
                continue
            held_path = d / "heldout_refs.json"
            if held_path.is_file():
                held = [Path(p) for p in json.loads(held_path.read_text())]
            else:
                _, held = metrics.split_refs(case.gt_refs,
                                             pipe_cfg.eval_heldout_refs)
            draft = Image.open(
                args.screen_run / case.id / "draft.png").convert("RGB")
            rows.append(score_case(case, d, draft, encoders,
                                   heldout_paths=held))
    finally:
        encoders.free()

    summary = summarise(rows)
    meta = {"run": args.run.name,
            "arm": run_info.get("args", {}).get("arm", "?"),
            "mechanism": run_info.get("args", {}).get("mechanism", "?"),
            "heldout_refs": pipe_cfg.eval_heldout_refs}

    (args.run / "report.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "rows": rows, "skipped": skipped},
        indent=2, default=str))
    (args.run / "report.md").write_text(render(summary, meta))
    print(f"[report] {len(rows)} cases, {len(skipped)} skipped -> "
          f"{args.run / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Move the `import argparse` / `import numpy as np` / `from PIL import Image` lines
to the module header if they are not already there.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_report.py -v`
Expected: 11 passed

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat: the results table, stratified, with its caveats attached"
```

---

### Task 10: Wire it in, run it, document it

**Files:**
- Modify: `scripts/run.sh`, `docs/RUNBOOK.md`
- Test: none — exercised by running it

**Interfaces:**
- Consumes: `scripts/report.py` (Tasks 7–9)
- Produces: no new API

- [ ] **Step 1: Add the stage to `run.sh`**

In the usage block, after the `pipeline` line:

```
  report       metrics over a finished run       (GPU, ~5 min)
```

In the `case` statement, after the `pipeline` line:

```bash
  report)      exec "$PY" scripts/report.py "$@" ;;
```

- [ ] **Step 2: Document it**

In `docs/RUNBOOK.md` §4, after the `pipeline` row:

```markdown
| `report` | yes, light | ~5 min | Metrics over a finished pipeline run: DINO identity against **held-out** references, preservation, CLIP/SigLIP. Writes `report.md` and `report.json` into the run directory. Needs `--run outputs/pipeline_<TS>`. |
```

In `docs/RUNBOOK.md` §5.2, add below the metrics table:

```markdown
**The oracle arm's references are split.** `eval.heldout_refs` (default 1) reserves the last
reference of every case for DINO. The repair never sees it. Without the split, DINO would score
each output against an image the repair was handed — the same objection §4 makes about
`--proto-arm ceiling`.
```

- [ ] **Step 3: Check the card**

Run: `nvidia-smi`
Expected: ≥8 GB free for the three eval encoders. If not, stop and wait — do not lower anything to make it fit.

- [ ] **Step 4: Run the report on a real run**

```bash
./scripts/run.sh pipeline --limit 4 --arm oracle --mechanism stitch --tag doc4 \
  --screen-run outputs/screen_latest
./scripts/run.sh report --run outputs/doc4_<TS>
```

Read `outputs/doc4_<TS>/report.md`. Confirm: targets and controls appear as separate rows; healthy
cases show a whole-image DINO and no cropped one; preservation is exactly 1.000 for `stitch`, whose
compositor guarantees bit-identical pixels outside the mask. A preservation below 1.0 there is a
real defect — stop and investigate rather than recording the number.

- [ ] **Step 5: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green, 11 `@pytest.mark.gpu` deselected.

- [ ] **Step 6: Commit**

```bash
git add scripts/run.sh docs/RUNBOOK.md
git commit -m "feat: report is a run.sh stage, and the hold-out is documented"
```

---

## Notes for the executor

- **The held-out references are never edited with.** If a test seems to want DINO scored against the edit pool, the test is wrong — that is the exact self-marking this doc exists to remove.
- **Never compute repair rate from `queue.json`'s `status`.** It means two opposite things (`findings/2026-07-29-orchestration-result.md` §2). `scores.json`'s draft verdict is the only correct source.
- **Healthy cases legitimately have no `mask.png`.** Do not "fix" this by grounding at report time; that would load the detector and change what is being measured. A missing mask on a *failed* draft is a different thing and must raise.
- **Preservation must be exactly 1.0 for `stitch` composites.** Not 0.999. If it is not, the compositor changed and the metric is telling you so.
- **Do not lower `eval.heldout_refs` to 0** to make a thin dataset fit. Add a reference instead — 0 reintroduces the leakage the whole doc is about.
- `sushi` masks the plate, not the food (`findings/2026-07-28-masking-result.md` §3). Expected, not a bug.
