# Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a draft, a mask and a reference into a repaired image, by two mechanisms — FLUX.1-Kontext inpainting, and diffusion-free compositing.

**Architecture:** `ragregen/composite.py` holds pure pixel work — feathered alpha, compositing, RGBA cutouts — with no models and no torch, so it is fully testable on CPU. `ragregen/regen.py` holds the mechanisms: `stitch` (pure, but returns the shared result type) and `Inpainter` (owns 12 GB of FLUX weights, injected, never loaded at import). The split is what makes the spec's §7 constraint structural: everything except the single diffusion call is verifiable without a GPU.

**Tech Stack:** Python 3.11, numpy, Pillow, opencv (`cv2`), diffusers `FluxKontextInpaintPipeline`, pytest. No new dependencies.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Bare `python` is conda 3.13 and fails on faiss.
- **Dilation is owned by `mask_draft`.** `KontextConfig.dilate_px` defaults to **0** here. Chaining the two would grow the repaint region to ~24px of slack.
- **`largest_cc` defaults to `False`.** A mask with several components is a fact about the concept, not noise to clean up.
- **`padding_mask_crop` must be `None` whenever a reference is passed** — the pipeline crops the reference using the *source's* crop region (`kontext_engine.py:9-13`). Raise, never warn.
- **Both mechanisms share `feather_alpha`.** Two blending rules would make them incomparable on the preservation metric that exists to compare them.
- **Pixels outside the mask must be bit-identical to the draft.** Asserted with `==`, not `allclose`.
- **Dependency injection is mandatory.** No module-level model loading, no network at import.
- The 286 existing tests must stay green.
- `git add <exact paths>` — never `git add -A`. `outputs/` is gitignored.
- **GPU etiquette:** `nvidia-smi` before anything that loads FLUX. `preflight` refuses below 12 GB free and that refusal is correct behaviour, not a bug to work around.
- Port sources are **reference only** (`RUNBOOK.md` §7). Never modify anything under `../ImageRAG/` or `../rag-edit/`.

---

### Task 1: Feathered alpha and compositing

**Files:**
- Create: `ragregen/composite.py`
- Test: `tests/test_composite.py`

**Interfaces:**
- Produces: `feather_alpha(mask_binary, size, feather_px=6) -> np.ndarray` float `HxW` in `[0,1]`; `composite_back(original, raw, mask_binary, feather_px=6) -> tuple[Image, np.ndarray]`
- Consumed by Tasks 3, 4 and 7. Ported from `../ImageRAG/scripts/kontext_engine.py:473-518`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_composite.py`:

```python
import numpy as np
import pytest
from PIL import Image

from ragregen import composite


def _mask(size=(64, 64), box=(16, 16, 48, 48)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


def test_feather_alpha_produces_three_zones():
    a = composite.feather_alpha(_mask(), (64, 64), feather_px=4)
    assert a[0, 0] == 0.0, "outside must be exactly zero"
    assert a[32, 32] == pytest.approx(1.0), "interior must be exactly one"
    band = a[(a > 0.0) & (a < 1.0)]
    assert band.size > 0, "a feathered edge must exist between the two"


def test_feather_alpha_with_no_feather_is_hard():
    a = composite.feather_alpha(_mask(), (64, 64), feather_px=0)
    assert set(np.unique(a)) == {0.0, 1.0}


def test_composite_back_leaves_outside_pixels_bit_identical():
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    raw = Image.new("RGB", (64, 64), (200, 0, 0))
    out, alpha = composite.composite_back(original, raw, _mask(), feather_px=0)

    o = np.asarray(original)
    got = np.asarray(out)
    outside = alpha == 0.0
    assert (got[outside] == o[outside]).all()
    assert tuple(got[32, 32]) == (200, 0, 0)


def test_composite_back_resizes_a_raw_of_a_different_size():
    """Kontext returns a preferred resolution, almost never the input's."""
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    raw = Image.new("RGB", (128, 128), (200, 0, 0))
    out, _ = composite.composite_back(original, raw, _mask(), feather_px=0)
    assert out.size == (64, 64)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_composite.py -v`
Expected: FAIL — `ImportError: cannot import name 'composite' from 'ragregen'`

- [ ] **Step 3: Implement**

Create `ragregen/composite.py`:

```python
"""Pure pixel work shared by both regeneration mechanisms.

No models, no torch, no network. Everything here runs on CPU in a unit test,
which is what lets doc 2 be validated while FLUX cannot fit on the shared card
(specs/2026-07-28-regeneration-design.md §7).

Ported from ../ImageRAG/scripts/kontext_engine.py:473-518. The arithmetic is
load-bearing and reproduced exactly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def feather_alpha(mask_binary: Image.Image, size: tuple[int, int],
                  feather_px: int = 6) -> np.ndarray:
    """Hard mask -> float alpha in [0, 1] at `size`.

    Three zones fall out and the preservation metric depends on all three:
      outside  alpha == 0   must stay bit-identical to the original
      band     0 < alpha < 1
      inside   alpha == 1
    """
    m = mask_binary.convert("L")
    if m.size != size:
        # NEAREST only: bilinear invents grey values along the boundary, which
        # would put pixels in the feather band that the mask never marked.
        m = m.resize(size, Image.NEAREST)
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_px))
    return np.array(m).astype(np.float32) / 255.0


def composite_back(original: Image.Image, raw: Image.Image,
                   mask_binary: Image.Image, feather_px: int = 6
                   ) -> tuple[Image.Image, np.ndarray]:
    """Paste the edited region onto the untouched original, feathered.

    Mandatory, not cosmetic: FLUX blends in latent space at 1/8 resolution and
    then VAE-decodes the whole frame, so every unmasked pixel has already
    changed by the time we see `raw`. This is the collateral-damage guarantee.

    Also absorbs the resolution change -- `raw` comes back at a preferred
    Kontext resolution, which is almost never the original's size.
    """
    if raw.size != original.size:
        raw = raw.resize(original.size, Image.LANCZOS)

    o = np.array(original.convert("RGB")).astype(np.float32)
    r = np.array(raw.convert("RGB")).astype(np.float32)
    a = feather_alpha(mask_binary, original.size, feather_px)[..., None]

    out = np.rint(o * (1 - a) + r * a).astype(np.uint8)

    # Belt and braces: force exact equality outside the feather. The float
    # blend already gives this, but the preservation metric asserts == 0.0
    # exactly and should be testing the compositing rule, not float rounding.
    outside = a[..., 0] == 0.0
    out[outside] = np.array(original.convert("RGB"))[outside]

    return Image.fromarray(out, "RGB"), a[..., 0]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_composite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/composite.py tests/test_composite.py
git commit -m "feat: feathered alpha and the three-zone composite"
```

---

### Task 2: RGBA cutouts from a reference

**Files:**
- Modify: `ragregen/composite.py`
- Test: `tests/test_composite.py`

**Interfaces:**
- Consumes: `ragregen.mask.MaskResult` (doc 1) — fields `mask: np.ndarray` float `HxW`, `box`, `score`, `n_candidates`, `selected_by`
- Produces: `cutout_from(reference, mask_result) -> Image.Image | None` — RGBA, cropped to the silhouette's bounding box, alpha = the silhouette
- Consumed by Tasks 3 and 9.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_composite.py`:

```python
class _FakeMaskResult:
    """Stands in for mask.MaskResult; only `.mask` is read here."""

    def __init__(self, arr):
        self.mask = arr


def test_cutout_is_rgba_cropped_to_the_silhouette_box():
    ref = Image.new("RGB", (64, 64), (5, 6, 7))
    arr = np.zeros((64, 64), dtype=float)
    arr[16:48, 20:40] = 1.0
    out = composite.cutout_from(ref, _FakeMaskResult(arr))
    assert out.mode == "RGBA"
    assert out.size == (20, 32), "cropped to the silhouette bbox, not the frame"


def test_cutout_alpha_is_the_silhouette_not_a_rectangle():
    ref = Image.new("RGB", (32, 32), (5, 6, 7))
    arr = np.zeros((32, 32), dtype=float)
    arr[8:24, 8:24] = 1.0
    arr[8:12, 8:12] = 0.0            # bite a corner out of the square
    out = composite.cutout_from(ref, _FakeMaskResult(arr))
    alpha = np.asarray(out)[..., 3]
    assert alpha[0, 0] == 0, "the bitten corner must stay transparent"
    assert alpha[-1, -1] == 255


def test_cutout_of_an_empty_mask_is_none():
    ref = Image.new("RGB", (32, 32))
    assert composite.cutout_from(ref, _FakeMaskResult(np.zeros((32, 32)))) is None


def test_cutout_of_no_mask_result_is_none():
    assert composite.cutout_from(Image.new("RGB", (8, 8)), None) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_composite.py -k cutout -v`
Expected: FAIL — `AttributeError: module 'ragregen.composite' has no attribute 'cutout_from'`

- [ ] **Step 3: Implement**

Append to `ragregen/composite.py`:

```python
def cutout_from(reference: Image.Image, mask_result) -> Image.Image | None:
    """The reference's object as RGBA, cropped to its bounding box.

    `stitch` needs pixels plus a silhouette, not a MaskResult -- keeping the
    conversion here means stitch can be tested on synthetic arrays with no
    masker in sight.

    None when there is nothing to cut out, which is the same signal doc 1's
    masker gives: no object located means do not attempt this repair.
    """
    if mask_result is None:
        return None
    m = np.asarray(mask_result.mask) > 0.5
    ys, xs = np.where(m)
    if xs.size == 0:
        return None

    rgb = np.array(reference.convert("RGB"))
    rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(rgba, "RGBA").crop(box)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_composite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/composite.py tests/test_composite.py
git commit -m "feat: RGBA cutouts from a reference silhouette"
```

---

### Task 3: `stitch` — place and paste

**Files:**
- Create: `ragregen/regen.py`
- Test: `tests/test_regen.py`

**Interfaces:**
- Consumes: `composite.feather_alpha`, `composite.cutout_from`
- Produces: `RegenResult` dataclass (`image`, `raw`, `alpha`, `mechanism`, `meta`); `stitch(draft, mask, cutout, *, feather_px=6, fill_residual=True) -> RegenResult`
- Consumed by Tasks 4, 7 and 9. **Task 4 implements `fill_residual`; this task accepts the parameter and ignores it**, so the signature never changes under Task 4's feet.

- [ ] **Step 1: Write the failing test**

Create `tests/test_regen.py`:

```python
import numpy as np
import pytest
from PIL import Image

from ragregen import regen


def _draft(size=(64, 64), colour=(10, 20, 30)):
    return Image.new("RGB", size, colour)


def _mask(size=(64, 64), box=(16, 16, 48, 48)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


def _cutout(size=(16, 16), colour=(200, 0, 0)):
    c = Image.new("RGBA", size, colour + (255,))
    return c


def test_stitch_records_its_mechanism():
    out = regen.stitch(_draft(), _mask(), _cutout(), fill_residual=False)
    assert out.mechanism == "stitch"
    assert out.raw is None, "stitch runs no model, so there is no raw output"


def test_stitch_leaves_pixels_outside_the_mask_bit_identical():
    draft = _draft()
    out = regen.stitch(draft, _mask(), _cutout(), feather_px=0,
                       fill_residual=False)
    d, g = np.asarray(draft), np.asarray(out.image)
    outside = out.alpha == 0.0
    assert (g[outside] == d[outside]).all()


def test_stitch_puts_the_object_inside_the_mask_box():
    out = regen.stitch(_draft(), _mask(box=(16, 16, 48, 48)), _cutout(),
                       feather_px=0, fill_residual=False)
    ys, xs = np.where(out.alpha > 0)
    assert xs.min() >= 16 and xs.max() < 48
    assert ys.min() >= 16 and ys.max() < 48


def test_stitch_scales_to_fit_inside_without_cropping_the_object():
    """A cutout larger than its hole must shrink, never be clipped."""
    big = _cutout(size=(200, 100))
    out = regen.stitch(_draft(), _mask(box=(16, 16, 48, 48)), big,
                       feather_px=0, fill_residual=False)
    ys, xs = np.where(out.alpha > 0)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert w <= 32 and h <= 32
    assert w / h == pytest.approx(200 / 100, rel=0.15), "aspect preserved"


def test_stitch_uses_the_shared_feather_function():
    """One compositing rule for both mechanisms, so preservation is comparable."""
    from ragregen import composite

    calls = []
    real = composite.feather_alpha

    def spy(mask_binary, size, feather_px=6):
        calls.append(size)
        return real(mask_binary, size, feather_px)

    composite.feather_alpha = spy
    try:
        regen.stitch(_draft(), _mask(), _cutout(), fill_residual=False)
    finally:
        composite.feather_alpha = real
    assert calls == [(64, 64)]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: FAIL — `ImportError: cannot import name 'regen' from 'ragregen'`

- [ ] **Step 3: Implement**

Create `ragregen/regen.py`:

```python
"""The two repair mechanisms.

`stitch` pastes the reference's object into the hole and runs no model.
`Inpainter` conditions FLUX.1-Kontext on the reference and owns ~12 GB of
weights. The pilot picks one (parent spec §6); until it runs, neither is the
default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ragregen import composite

MECHANISMS = ("inpaint", "stitch")


@dataclass(frozen=True)
class RegenResult:
    image: Image.Image
    raw: Image.Image | None
    alpha: np.ndarray
    mechanism: str
    meta: dict = field(default_factory=dict)


def _mask_bounds(mask: Image.Image) -> tuple[np.ndarray, int, int, int, int]:
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(m)
    if xs.size == 0:
        raise ValueError("stitch: the mask is entirely black -- nothing to fill")
    return m, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def stitch(draft: Image.Image, mask: Image.Image, cutout: Image.Image, *,
           feather_px: int = 6, fill_residual: bool = True) -> RegenResult:
    """Paste the reference's object into the masked region. No diffusion.

    Fit-inside rather than fill: the object must never be cropped by its own
    hole. Placement is centred on the mask's centroid, which is where the
    object being replaced actually sat.
    """
    m, x0, x1, y0, y1 = _mask_bounds(mask)
    tw, th = x1 - x0 + 1, y1 - y0 + 1

    cw, ch = cutout.size
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cutout.convert("RGBA").resize((nw, nh), Image.LANCZOS)

    ys, xs = np.where(m)
    ox = int(round(xs.mean() - nw / 2))
    oy = int(round(ys.mean() - nh / 2))

    canvas = Image.new("RGBA", draft.size, (0, 0, 0, 0))
    canvas.paste(resized, (ox, oy), resized)

    pasted_alpha = Image.fromarray(np.asarray(canvas)[..., 3], "L")
    a = composite.feather_alpha(pasted_alpha, draft.size, feather_px)[..., None]

    d = np.array(draft.convert("RGB")).astype(np.float32)
    c = np.asarray(canvas.convert("RGB")).astype(np.float32)
    out = np.rint(d * (1 - a) + c * a).astype(np.uint8)

    outside = a[..., 0] == 0.0
    out[outside] = np.array(draft.convert("RGB"))[outside]

    return RegenResult(image=Image.fromarray(out, "RGB"), raw=None,
                       alpha=a[..., 0], mechanism="stitch",
                       meta={"scale": round(scale, 4),
                             "placed_at": [ox, oy],
                             "cutout_size": [nw, nh],
                             "mask_box": [x0, y0, x1, y1],
                             "residual_filled": False})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/regen.py tests/test_regen.py
git commit -m "feat: stitch places and pastes a reference cutout"
```

---

### Task 4: `stitch` — fill the residual

**Files:**
- Modify: `ragregen/regen.py`
- Test: `tests/test_regen.py`

**Interfaces:**
- Produces: `fill_residual=True` (the default) now fills `mask AND NOT pasted_alpha` via `cv2.inpaint`, and `meta["residual_filled"]` reports whether any pixel needed it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regen.py`:

```python
DRAFT_RGB = np.array([10, 20, 30], dtype=np.uint8)

#: 1:2 aspect against a square hole, so fit-inside leaves margins either side.
#: A SQUARE cutout in a square hole scales to fit exactly and leaves no
#: residual at all -- which would make every test below vacuously green.
THIN = (8, 16)


def _kept_draft_pixels(image, mask_img) -> int:
    m = np.asarray(mask_img) > 127
    a = np.asarray(image)
    return int(((a == DRAFT_RGB).all(axis=-1) & m).sum())


def test_residual_fill_reduces_the_draft_pixels_left_inside_the_mask():
    """The cutout's shape differs from the hole's, so a crescent of the
    ORIGINAL object would otherwise show around the pasted one -- fragments of
    a generic leopard around a pasted Amur leopard, which reads as a generation
    artifact rather than as what it is.

    Asserted as a reduction, not an absence: cv2.inpaint propagates inward from
    the residual's boundary, and part of that boundary is the draft itself, so
    some filled pixels legitimately land back on the draft's colour.
    """
    draft, mask = _draft(colour=(10, 20, 30)), _mask(box=(16, 16, 48, 48))
    unfilled = regen.stitch(draft, mask, _cutout(size=THIN), feather_px=0,
                            fill_residual=False)
    filled = regen.stitch(draft, mask, _cutout(size=THIN), feather_px=0,
                          fill_residual=True)

    assert _kept_draft_pixels(filled.image, mask) < \
           _kept_draft_pixels(unfilled.image, mask)
    assert filled.meta["residual_filled"] is True


def test_residual_fill_can_be_switched_off():
    out = regen.stitch(_draft(), _mask(), _cutout(size=THIN),
                       feather_px=0, fill_residual=False)
    assert out.meta["residual_filled"] is False


def test_a_cutout_that_fills_the_hole_needs_no_residual_fill():
    """Square cutout, square hole: fit-inside covers it exactly."""
    out = regen.stitch(_draft(), _mask(box=(16, 16, 48, 48)),
                       _cutout(size=(32, 32)), feather_px=0, fill_residual=True)
    assert out.meta["residual_filled"] is False


def test_residual_fill_still_leaves_outside_pixels_bit_identical():
    """cv2.inpaint must not bleed across the mask boundary."""
    draft = _draft()
    out = regen.stitch(draft, _mask(), _cutout(size=THIN), feather_px=0,
                       fill_residual=True)
    d, g = np.asarray(draft), np.asarray(out.image)
    outside = np.asarray(_mask()) <= 127
    assert (g[outside] == d[outside]).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_regen.py -k residual -v`
Expected: FAIL — `test_residual_fill_reduces_the_draft_pixels_left_inside_the_mask` fails because with no fill the two counts are equal (not `<`), and `meta["residual_filled"]` is hard-coded `False`.

- [ ] **Step 3: Implement**

In `ragregen/regen.py`, replace the `return RegenResult(...)` at the end of `stitch` with:

```python
    filled = False
    if fill_residual:
        # mask minus what the paste actually covered. cv2.inpaint diffuses
        # surrounding colour inward: classical, milliseconds, and no diffusion
        # model -- which is the entire point of stitch being an arm.
        residual = m & (a[..., 0] <= 0.0)
        if residual.any():
            import cv2

            out = cv2.inpaint(out, residual.astype(np.uint8), 3,
                              cv2.INPAINT_TELEA)
            filled = True

    # Restore outside the MASK, not outside the paste. The residual sits
    # INSIDE the mask with alpha 0, so restoring by `a == 0` would overwrite
    # every pixel cv2.inpaint just filled and silently nullify the fill. `~m`
    # is also the guarantee as actually stated: pixels outside the mask are
    # bit-identical; inside it, we are free to paste and fill.
    out[~m] = np.array(draft.convert("RGB"))[~m]

    return RegenResult(image=Image.fromarray(out, "RGB"), raw=None,
                       alpha=a[..., 0], mechanism="stitch",
                       meta={"scale": round(scale, 4),
                             "placed_at": [ox, oy],
                             "cutout_size": [nw, nh],
                             "mask_box": [x0, y0, x1, y1],
                             "residual_filled": filled})
```

Delete the earlier `outside = ...` / `out[outside] = ...` pair that Task 3 placed
before the return — the restore must happen *after* the inpaint, or `cv2.inpaint`
will bleed colour across the mask boundary into pixels the guarantee protects.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest --ignore=tests/test_draft.py -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add ragregen/regen.py tests/test_regen.py
git commit -m "feat: fill the stitch residual classically, no diffusion"
```

---

### Task 5: `KontextConfig` and input validation

**Files:**
- Modify: `ragregen/regen.py`
- Test: `tests/test_regen.py`

**Interfaces:**
- Produces: `KontextConfig` dataclass; `PreparedInputs` dataclass; `validate_inputs(prompt, image, mask_image, image_reference=None, cfg=None) -> PreparedInputs`
- **`dilate_px` defaults to 0 and `largest_cc` to `False`** — the two deviations from the port source, both load-bearing (design §3).
- Consumed by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regen.py`:

```python
def test_validate_inputs_does_not_dilate_a_mask_mask_draft_already_dilated():
    """Chaining mask_draft's 12px with the port's 12px would grow the repaint
    region to ~24px of slack, silently (design §3)."""
    mask = _mask(box=(16, 16, 48, 48))
    prepared = regen.validate_inputs("repaint it", _draft(), mask)
    before = (np.asarray(mask) > 127).sum()
    after = (np.asarray(prepared.mask_binary) > 127).sum()
    assert after == before


def test_validate_inputs_keeps_every_component_of_a_multi_part_mask():
    """A plate of sushi is many pieces. That is the concept, not noise."""
    m = Image.new("L", (64, 64), 0)
    m.paste(255, (4, 4, 12, 12))
    m.paste(255, (40, 40, 60, 60))
    prepared = regen.validate_inputs("repaint it", _draft(), m)
    arr = np.asarray(prepared.mask_binary) > 127
    assert arr[4:12, 4:12].any() and arr[40:60, 40:60].any()


def test_validate_inputs_rejects_an_empty_prompt():
    with pytest.raises(ValueError, match="prompt"):
        regen.validate_inputs("  ", _draft(), _mask())


def test_validate_inputs_rejects_an_all_black_mask():
    with pytest.raises(ValueError, match="entirely black"):
        regen.validate_inputs("repaint it", _draft(), Image.new("L", (64, 64), 0))


def test_validate_inputs_rejects_an_all_white_mask():
    with pytest.raises(ValueError, match="entirely white"):
        regen.validate_inputs("repaint it", _draft(),
                              Image.new("L", (64, 64), 255))


def test_validate_inputs_notes_a_suspiciously_large_mask():
    m = Image.new("L", (64, 64), 0)
    m.paste(255, (0, 0, 64, 50))          # 78% of the frame
    prepared = regen.validate_inputs("repaint it", _draft(), m)
    assert any("inverted" in n for n in prepared.notes)


def test_validate_inputs_keeps_the_original_untouched():
    draft = _draft()
    prepared = regen.validate_inputs("repaint it", draft, _mask())
    assert np.asarray(prepared.original).shape == np.asarray(draft).shape
    assert (np.asarray(prepared.original) == np.asarray(draft)).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_regen.py -k validate_inputs -v`
Expected: FAIL — `AttributeError: module 'ragregen.regen' has no attribute 'validate_inputs'`

- [ ] **Step 3: Implement**

Append to `ragregen/regen.py`:

```python
@dataclass(frozen=True)
class KontextConfig:
    """Everything tunable in one place, so the trace can record the whole state.

    Two defaults deviate from the port source, both deliberately:

    `dilate_px = 0` -- mask_draft already dilated by the operator's
    `mask_dilate_px`. Dilating again would grow the repaint region to ~24px of
    slack without saying so.

    `largest_cc = False` -- keeping only the biggest connected component
    discards most of a multi-piece mask. A plate of sushi is many pieces; that
    is a fact about the concept, not noise to clean up.
    """
    model_id: str = "black-forest-labs/FLUX.1-Kontext-dev"
    device: str = "cuda"
    quantize: str = "nf4"
    offload: str | None = None
    dilate_px: int = 0
    blur_factor: int = 12
    largest_cc: bool = False
    steps: int = 28
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    strength: float = 1.0
    max_area: int = 1024 ** 2
    seed: int = 0
    ref_prep: str = "crop"
    feather_px: int = 6


@dataclass
class PreparedInputs:
    prompt: str
    original: Image.Image
    image: Image.Image
    mask_binary: Image.Image
    mask_blurred: Image.Image
    reference: Image.Image | None
    notes: list = field(default_factory=list)


def _to_binary(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    m = mask.convert("L")
    if m.size != size:
        # NEAREST only: bilinear invents grey along the boundary.
        m = m.resize(size, Image.NEAREST)
    return np.array(m) > 127


def _dilate_binary(binary: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return binary
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(binary.astype(np.uint8), k, iterations=1) > 0


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2

    n, labels = cv2.connectedComponents(binary.astype(np.uint8))
    n_comp = n - 1                       # label 0 is background
    if n_comp <= 1:
        return binary, max(n_comp, 0)
    sizes = [int((labels == i).sum()) for i in range(1, n_comp + 1)]
    return labels == (int(np.argmax(sizes)) + 1), n_comp


def _blur(mask: Image.Image, blur_factor: int) -> Image.Image:
    from PIL import ImageFilter

    if blur_factor <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(blur_factor))


def validate_inputs(prompt: str, image, mask_image, image_reference=None,
                    cfg: KontextConfig | None = None) -> PreparedInputs:
    """Everything the pipeline call needs, validated.

    Order matters: binarise -> (largest component) -> dilate -> blur. The hard
    mask is kept alongside the blurred one because compositing and the
    preservation metric need a crisp boundary, not the soft one the model eats.
    """
    cfg = cfg or KontextConfig()
    notes: list = []

    if not (prompt or "").strip():
        raise ValueError("validate_inputs: empty prompt. Kontext is "
                         "instruction-driven and ignores an empty one.")

    image = image.convert("RGB")
    original = image.copy()

    binary = _to_binary(mask_image, image.size)
    frac = float(binary.mean())
    if frac == 0.0:
        raise ValueError("validate_inputs: mask is entirely black -- nothing "
                         "to edit. White = the region to edit.")
    if frac == 1.0:
        raise ValueError("validate_inputs: mask is entirely white -- the whole "
                         "frame would be regenerated. White = the region to edit.")
    if frac > 0.5:
        notes.append(f"WARNING: mask covers {frac:.0%} of the frame -- white "
                     f"must be the EDIT region; this looks inverted")

    if cfg.largest_cc:
        binary, n_comp = _largest_component(binary)
        if n_comp > 1:
            notes.append(f"mask had {n_comp} components; kept the largest")

    binary = _dilate_binary(binary, cfg.dilate_px)
    mask_binary = Image.fromarray((binary * 255).astype(np.uint8), "L")

    # Blur softens the seam the model renders. It preserves nothing -- that is
    # composite_back's job.
    mask_blurred = _blur(mask_binary, cfg.blur_factor)

    reference = (image_reference.convert("RGB")
                 if image_reference is not None else None)

    return PreparedInputs(prompt=prompt.strip(), original=original, image=image,
                          mask_binary=mask_binary, mask_blurred=mask_blurred,
                          reference=reference, notes=notes)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/regen.py tests/test_regen.py
git commit -m "feat: kontext config and input validation, dilation owned upstream"
```

---

### Task 6: Reference preparation

**Files:**
- Modify: `ragregen/regen.py`
- Test: `tests/test_regen.py`

**Interfaces:**
- Produces: `prepare_reference(reference, concept, mode="crop", masker=None, pad_frac=0.08) -> tuple[Image, dict]`
- `masker` is `callable(image, phrase) -> (float mask HxW, info)` — the contract doc 1's `mask_reference` already satisfies, pinned by `tests/test_mask.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regen.py`:

```python
def _masker_for(box, size=(64, 64)):
    def masker(image, phrase):
        arr = np.zeros(size[::-1], dtype=float)
        x0, y0, x1, y1 = box
        arr[y0:y1, x0:x1] = 1.0
        return arr, {}
    return masker


def test_prepare_reference_crops_to_the_padded_object_box():
    ref = Image.new("RGB", (64, 64), (5, 6, 7))
    out, info = regen.prepare_reference(ref, "durian", mode="crop",
                                        masker=_masker_for((16, 16, 32, 32)))
    # xs 16..31 -> pw = int(15 * 0.08) + 4 = 5 -> box (11, 11, 37, 37)
    assert info["applied"] == "crop"
    assert out.size == (26, 26)


def test_prepare_reference_without_a_masker_returns_the_reference_unchanged():
    ref = Image.new("RGB", (64, 64))
    out, info = regen.prepare_reference(ref, "durian", mode="crop", masker=None)
    assert out is ref
    assert info["applied"] == "none"


def test_prepare_reference_of_an_ungrounded_concept_returns_the_whole_image():
    ref = Image.new("RGB", (64, 64))
    empty = lambda image, phrase: (np.zeros((64, 64)), {})
    out, info = regen.prepare_reference(ref, "wombat", mode="crop", masker=empty)
    assert out is ref
    assert "wombat" in info["reason"]


def test_prepare_reference_matte_greys_the_background_not_the_object():
    ref = Image.new("RGB", (64, 64), (250, 250, 250))
    out, info = regen.prepare_reference(ref, "durian", mode="crop_matte",
                                        masker=_masker_for((16, 16, 32, 32)))
    arr = np.asarray(out)
    assert info["applied"] == "crop_matte"
    # The object is global [16, 32); the crop origin is 11, so local = global
    # - 11 and the object occupies local [5, 21). Local (2,2) is global
    # (13,13) -- in the padding, outside the object. Local (13,13) is global
    # (24,24) -- squarely inside it.
    assert tuple(arr[2, 2]) == (127, 127, 127), "the padding is matted grey"
    assert tuple(arr[13, 13]) == (250, 250, 250), "the object survives"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_regen.py -k prepare_reference -v`
Expected: FAIL — `AttributeError: module 'ragregen.regen' has no attribute 'prepare_reference'`

- [ ] **Step 3: Implement**

Append to `ragregen/regen.py`:

```python
def prepare_reference(reference: Image.Image, concept: str, mode: str = "crop",
                      masker=None, pad_frac: float = 0.08
                      ) -> tuple[Image.Image, dict]:
    """Crop / matte the reference before it is encoded.

    The pipeline auto-resizes the reference to a ~1MP preferred resolution and
    encodes it to ~4k latent tokens, so subject scale in the reference frame
    buys token budget directly: a subject filling 10% of the frame gets 10% of
    the budget. Cropping to the subject is the cheap lever and is the default.

    `crop_matte` is offered but not default: the hard alpha edge it leaves is
    encoded by the VAE as a real object boundary. Which default is right is
    settled by the ref-prep ablation, not by assertion.

    `masker` is callable(image, phrase) -> (float mask HxW, info) -- exactly
    what doc 1's mask_reference produces.
    """
    info: dict = {"ref_prep": mode}
    if mode == "none" or masker is None:
        info["applied"] = "none"
        if mode != "none":
            info["reason"] = "no masker supplied"
        return reference, info

    m, minfo = masker(reference, concept)
    info["mask_info"] = {k: v for k, v in (minfo or {}).items() if k != "bbox"}
    ys, xs = np.where(np.asarray(m) > 0.5)
    if xs.size == 0:
        info["applied"] = "none"
        info["reason"] = f"{concept!r} not grounded in the reference"
        return reference, info

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pw = int((x1 - x0) * pad_frac) + 4
    ph = int((y1 - y0) * pad_frac) + 4
    box = (max(0, x0 - pw), max(0, y0 - ph),
           min(reference.width, x1 + pw + 1),
           min(reference.height, y1 + ph + 1))

    out = reference
    if mode == "crop_matte":
        # Neutral grey, not white or black: a white cut-out on a dark scene
        # reads to the model as a bright object, and it renders one.
        arr = np.array(reference.convert("RGB")).astype(np.float32)
        alpha = np.clip(np.asarray(m), 0, 1)[..., None]
        out = Image.fromarray(
            (arr * alpha + 127.0 * (1 - alpha)).astype(np.uint8), "RGB")

    out = out.crop(box)
    info["applied"] = mode
    info["bbox"] = list(box)
    return out, info
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/regen.py tests/test_regen.py
git commit -m "feat: reference preparation, crop and matte"
```

---

### Task 7: The `Inpainter`

**Files:**
- Modify: `ragregen/regen.py`
- Test: `tests/test_regen.py`

**Interfaces:**
- Produces: `Inpainter(cfg=None)` with `.load()`, `.regen(draft, mask, reference, prompt, *, seed=None, use_reference=True, padding_mask_crop=None) -> RegenResult`, `.free()`
- The pipeline object is injectable via `Inpainter(cfg, pipe=<fake>)` so every non-diffusion path is testable on CPU.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regen.py`:

```python
class FakePipe:
    """Stands in for FluxKontextInpaintPipeline.

    Returns a solid red frame at a DIFFERENT size than the input, which is what
    Kontext really does -- output resolution is a preferred one, not the input's.
    """

    def __init__(self, out_size=(128, 128)):
        self.out_size = out_size
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        img = Image.new("RGB", self.out_size, (200, 0, 0))
        return type("R", (), {"images": [img]})()

    def set_progress_bar_config(self, **kw):
        pass


def test_inpainter_records_its_mechanism_and_composites_to_input_size():
    pipe = FakePipe()
    inp = regen.Inpainter(pipe=pipe)
    out = inp.regen(_draft(), _mask(), Image.new("RGB", (32, 32)), "repaint it")
    assert out.mechanism == "inpaint"
    assert out.image.size == (64, 64), "composited back to the ORIGINAL size"
    assert out.raw.size == (128, 128), "raw keeps the model's own resolution"


def test_inpainter_leaves_pixels_outside_the_mask_bit_identical():
    draft = _draft()
    inp = regen.Inpainter(pipe=FakePipe())
    out = inp.regen(draft, _mask(), None, "repaint it")
    d, g = np.asarray(draft), np.asarray(out.image)
    outside = out.alpha == 0.0
    assert (g[outside] == d[outside]).all()


def test_use_reference_false_runs_the_identical_call_without_the_reference():
    pipe = FakePipe()
    inp = regen.Inpainter(pipe=pipe)
    inp.regen(_draft(), _mask(), Image.new("RGB", (32, 32)), "repaint it",
              use_reference=False)
    assert pipe.calls[0]["image_reference"] is None


def test_padding_mask_crop_with_a_reference_raises():
    """The pipeline preprocesses the reference with the SOURCE's crop region,
    cropping it to an unrelated rectangle (kontext_engine.py:9-13)."""
    inp = regen.Inpainter(pipe=FakePipe())
    with pytest.raises(ValueError, match="padding_mask_crop"):
        inp.regen(_draft(), _mask(), Image.new("RGB", (32, 32)), "repaint it",
                  padding_mask_crop=32)


def test_padding_mask_crop_without_a_reference_is_allowed():
    pipe = FakePipe()
    inp = regen.Inpainter(pipe=pipe)
    inp.regen(_draft(), _mask(), None, "repaint it", padding_mask_crop=32)
    assert pipe.calls[0]["padding_mask_crop"] == 32


def test_regen_before_load_is_a_clear_error():
    with pytest.raises(RuntimeError, match="load"):
        regen.Inpainter().regen(_draft(), _mask(), None, "repaint it")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_regen.py -k inpainter -v`
Expected: FAIL — `AttributeError: module 'ragregen.regen' has no attribute 'Inpainter'`

- [ ] **Step 3: Implement**

Append to `ragregen/regen.py`:

```python
class Inpainter:
    """FLUX.1-Kontext, conditioned on a reference image.

    `pipe` is injectable so every path except the diffusion call itself is
    testable on CPU -- which matters because FLUX needs ~12 GB and the card is
    shared (design §7).
    """

    def __init__(self, cfg: KontextConfig | None = None, pipe=None):
        self.cfg = cfg or KontextConfig()
        self.pipe = pipe

    def load(self) -> "Inpainter":
        import torch
        from diffusers import FluxKontextInpaintPipeline

        from ragregen import env

        cfg = self.cfg
        kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": str(env.HF_CACHE)}
        if cfg.quantize == "nf4":
            from diffusers import PipelineQuantizationConfig

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True,
                              "bnb_4bit_quant_type": "nf4",
                              "bnb_4bit_compute_dtype": torch.bfloat16,
                              "bnb_4bit_use_double_quant": True},
                # T5-XXL is 9.5GB in bf16 and is the difference between fitting
                # a 24GB card and not.
                components_to_quantize=["transformer", "text_encoder_2"])

        self.pipe = FluxKontextInpaintPipeline.from_pretrained(cfg.model_id,
                                                               **kwargs)
        self.pipe.set_progress_bar_config(disable=True)
        if cfg.offload == "on":
            self.pipe.enable_model_cpu_offload(device=cfg.device)
        else:
            self.pipe.to(cfg.device)
        # The VAE decode is the peak allocation on a 1MP frame and the first
        # thing to OOM on a shared card.
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()
        return self

    def regen(self, draft, mask, reference, prompt: str, *, seed=None,
              use_reference: bool = True, padding_mask_crop=None) -> RegenResult:
        if self.pipe is None:
            raise RuntimeError("Inpainter.load() has not been called")

        cfg = self.cfg
        ref = reference if use_reference else None
        if padding_mask_crop is not None and ref is not None:
            raise ValueError(
                "padding_mask_crop cannot be combined with a reference: the "
                "pipeline preprocesses the reference with the SOURCE's crop "
                "region, cropping it to an unrelated rectangle. Leave it None.")

        prepared = validate_inputs(prompt, draft, mask, ref, cfg)
        call = {"prompt": prepared.prompt, "image": prepared.image,
                "mask_image": prepared.mask_blurred, "image_reference": ref,
                "strength": cfg.strength, "num_inference_steps": cfg.steps,
                "guidance_scale": cfg.guidance_scale,
                "true_cfg_scale": cfg.true_cfg_scale, "max_area": cfg.max_area}
        if padding_mask_crop is not None:
            call["padding_mask_crop"] = padding_mask_crop

        import time
        t0 = time.time()
        raw = self.pipe(**call).images[0]

        image, alpha = composite.composite_back(prepared.original, raw,
                                                prepared.mask_binary,
                                                cfg.feather_px)
        meta = {"seed": cfg.seed if seed is None else seed,
                "steps": cfg.steps, "strength": cfg.strength,
                "guidance_scale": cfg.guidance_scale,
                "used_reference": ref is not None,
                "input_size": list(prepared.original.size),
                "raw_output_size": list(raw.size),
                "feather_px": cfg.feather_px,
                "notes": prepared.notes,
                "seconds": round(time.time() - t0, 1)}
        return RegenResult(image=image, raw=raw, alpha=alpha,
                           mechanism="inpaint", meta=meta)

    def free(self) -> None:
        from ragregen import env

        self.pipe = None
        env.reclaim_gpu()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 5: Refuse to load when the card cannot hold it**

Append to `tests/test_regen.py`:

```python
def test_free_vram_below_the_floor_refuses_rather_than_thrashing():
    """A shared 4090. Refusing is correct behaviour, not a bug to work around."""
    with pytest.raises(RuntimeError, match="12.0 GB"):
        regen.check_vram(free_gb=9.6)


def test_free_vram_above_the_floor_passes():
    regen.check_vram(free_gb=18.0)          # must not raise
```

Run: `$PY -m pytest tests/test_regen.py -k vram -v`
Expected: FAIL — `AttributeError: module 'ragregen.regen' has no attribute 'check_vram'`

Add to `ragregen/regen.py`, above `class Inpainter`:

```python
#: Below this, FLUX-nf4 does not fit beside another researcher's job and the
#: run thrashes or OOMs mid-generation. ../rag-edit died exactly that way.
MIN_VRAM_GB = 12.0


def free_vram_gb() -> float | None:
    """Free VRAM, or None when there is no CUDA device to ask."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def check_vram(free_gb: float | None = None) -> None:
    """Raise unless the card can hold the pipeline. Never lower this floor."""
    free_gb = free_vram_gb() if free_gb is None else free_gb
    if free_gb is None:
        return
    if free_gb < MIN_VRAM_GB:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free; FLUX-nf4 needs {MIN_VRAM_GB} GB. "
            f"Another job is holding the card -- wait rather than thrash. "
            f"`nvidia-smi` shows who.")
```

and call it as the first line of `Inpainter.load`:

```python
        check_vram()
```

Run: `$PY -m pytest tests/test_regen.py -v`
Expected: PASS.

- [ ] **Step 6: Add the GPU-marked real call**

Append to `tests/test_regen.py`:

```python
@pytest.mark.gpu
def test_real_inpainter_returns_the_input_size_and_preserves_outside():
    """The only test that needs FLUX. Run it first when the card frees up."""
    draft = Image.new("RGB", (512, 512), (10, 20, 30))
    inp = regen.Inpainter().load()
    try:
        out = inp.regen(draft, _mask((512, 512), (128, 128, 384, 384)),
                        None, "a red sphere")
    finally:
        inp.free()
    assert out.image.size == (512, 512)
    d, g = np.asarray(draft), np.asarray(out.image)
    outside = out.alpha == 0.0
    assert (g[outside] == d[outside]).all()
```

- [ ] **Step 7: Commit**

```bash
git add ragregen/regen.py tests/test_regen.py
git commit -m "feat: the Kontext inpainter, with an injectable pipeline"
```

---

### Task 8: `mask_dilate_px` as an operator knob

**Files:**
- Modify: `ragregen/config.py:50-57` (`PipelineConfig`), `ragregen/config.py:141-150` (`load_pipeline`), `configs/pipeline.yaml`, `configs/dataset.example.yaml` is **not** touched
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `PipelineConfig.mask_dilate_px: int` (default 12)
- Closes a gap: the masking spec promised this knob and it was never added.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_pipeline_carries_the_mask_dilation_knob(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nmask_dilate_px: 20\n")
    assert config.load_pipeline(p).mask_dilate_px == 20


def test_mask_dilate_px_defaults_to_twelve(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\n")
    assert config.load_pipeline(p).mask_dilate_px == 12
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_config.py -k mask_dilat -v`
Expected: FAIL — `AttributeError: 'PipelineConfig' object has no attribute 'mask_dilate_px'`

- [ ] **Step 3: Implement**

In `ragregen/config.py`, add to `PipelineConfig` after `retriever`:

```python
    #: Slack around the draft mask, in pixels. A pixel-tight mask leaves a halo
    #: of the ORIGINAL object's edge which the inpainter rebuilds, reintroducing
    #: what we asked it to replace. Owned here, never applied twice: regen's
    #: KontextConfig.dilate_px is 0.
    mask_dilate_px: int = 12
```

In `load_pipeline`, add before the closing paren:

```python
        mask_dilate_px=int(data.get("mask_dilate_px", 12)),
```

In `configs/pipeline.yaml`, add after the `seed` line:

```yaml
mask_dilate_px: 12     # slack around the draft mask (doc 1 §5)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/config.py configs/pipeline.yaml tests/test_config.py
git commit -m "feat: mask_dilate_px as an operator knob"
```

---

### Task 9: The end-to-end repair script

**Files:**
- Create: `scripts/repair_case.py`
- Modify: `scripts/run.sh`, `docs/RUNBOOK.md` §4
- Test: none — this is a smoke harness, exercised by running it

**Interfaces:**
- Consumes: everything above, plus `ragregen.mask` from doc 1
- Produces: no new API

- [ ] **Step 1: Write the script**

Create `scripts/repair_case.py`:

```python
#!/usr/bin/env python
"""Repair one case, oracle arm. The first image in this project to look at.

Deliberately no retrieval, no queue, no VLM: the oracle arm needs none of them,
and each is a separate failure surface between the input and the output.

**Drafts do not exist yet.** Until a screen run produces them, --draft accepts
any image -- including a gt_ref, which repairs a correct image and should
therefore change little. A weak end-to-end check, and honest about being one.

Usage:
  ./scripts/repair_case.py --case boston_bull --mechanism stitch
  ./scripts/repair_case.py --case boston_bull --mechanism inpaint --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import composite, config, env, mask, models, regen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--mechanism", choices=regen.MECHANISMS, default="stitch")
    ap.add_argument("--draft", type=Path, default=None,
                    help="defaults to the case's first gt_ref")
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--out", type=Path, default=Path("outputs/repair"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    cases = {c.id: c for c in ds.cases}
    if args.case not in cases:
        print(f"[error] no case '{args.case}'. Known: {', '.join(sorted(cases))}")
        return 2
    case = cases[args.case]

    draft_path = args.draft or Path(case.gt_refs[0])
    if not draft_path.is_file():
        print(f"[error] draft not found: {draft_path}")
        return 2

    from PIL import Image

    draft = Image.open(draft_path).convert("RGB")
    ref_path = Path(case.gt_refs[-1])
    reference = Image.open(ref_path).convert("RGB")
    print(f"[repair] {case.id}  draft={draft_path.name}  ref={ref_path.name}  "
          f"mechanism={args.mechanism}", flush=True)

    masker = mask.Masker(models.DinoDetector(device=args.device),
                         sam=models.SamSegmenter(device=args.device))
    draft_mask = mask.mask_draft(draft, case.coarse, masker,
                                 score_phrase=case.concept,
                                 dilate_px=pipe_cfg.mask_dilate_px)
    if draft_mask is None:
        print(f"[repair] '{case.coarse}' not grounded in the draft -- "
              f"nothing to repair. The draft stands.")
        return 0

    ref_mask = mask.mask_reference(reference, case.concept, masker)
    cutout = composite.cutout_from(reference, ref_mask)
    masker.free()
    env.reclaim_gpu()

    from PIL import Image as _I
    mask_pil = _I.fromarray((draft_mask.mask > 0.5).astype("uint8") * 255, "L")

    if args.mechanism == "stitch":
        if cutout is None:
            print(f"[error] '{case.concept}' not grounded in the reference; "
                  f"stitch needs a segmented object.")
            return 2
        result = regen.stitch(draft, mask_pil, cutout,
                              feather_px=regen.KontextConfig().feather_px)
    else:
        prompt = f"replace the {case.coarse} with the {case.concept}"
        cfg = regen.KontextConfig(device=args.device, steps=pipe_cfg.steps,
                                  seed=pipe_cfg.seed)
        inpainter = regen.Inpainter(cfg).load()
        try:
            result = inpainter.regen(draft, mask_pil, reference, prompt)
        finally:
            inpainter.free()

    out_dir = args.out / f"{case.id}_{args.mechanism}"
    out_dir.mkdir(parents=True, exist_ok=True)
    draft.save(out_dir / "before.png")
    result.image.save(out_dir / "after.png")
    mask_pil.save(out_dir / "mask.png")
    if result.raw is not None:
        result.raw.save(out_dir / "raw.png")
    (out_dir / "meta.json").write_text(json.dumps(
        {"case": case.id, "mechanism": result.mechanism,
         "draft": str(draft_path), "reference": str(ref_path),
         "selected_by": draft_mask.selected_by,
         "n_candidates": draft_mask.n_candidates,
         "meta": result.meta}, indent=2, default=str))

    print(f"[repair] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it on the mechanism that needs no GPU weights for diffusion**

Run: `$PY scripts/repair_case.py --case boston_bull --mechanism stitch`
Expected: exit 0, and `outputs/repair/boston_bull_stitch/` containing `before.png`,
`after.png`, `mask.png`, `meta.json`. **Open `after.png` and look at it.** The
pasted dog should sit inside the masked region with no fragment of the original
animal around it.

- [ ] **Step 3: Wire it into `run.sh`**

In `scripts/run.sh`, add to the usage block after the `score-b` line:

```
  repair       repair ONE case, oracle arm      (GPU, ~1 min stitch)
```

and to the `case` statement after the `score-b` line:

```bash
  repair)      exec "$PY" scripts/repair_case.py "$@" ;;
```

- [ ] **Step 4: Document it**

In `docs/RUNBOOK.md` §4, add after the `score-b` row:

```markdown
| `repair` | yes | ~1 min (stitch) / ~8 min (inpaint) | Repairs one case on the oracle arm and writes before/after PNGs. The smoke test for the regeneration half; not part of the main table. |
```

- [ ] **Step 5: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green, `@pytest.mark.gpu` tests deselected.

- [ ] **Step 6: Commit**

```bash
git add scripts/repair_case.py scripts/run.sh docs/RUNBOOK.md
git commit -m "feat: repair one case end to end on the oracle arm"
```

---

## Notes for the executor

- **Never modify `../ImageRAG/` or `../rag-edit/`.** They are reference-only (`RUNBOOK.md` §7). Port by copying into `ragregen/`, not by importing across.
- **`dilate_px=0` and `largest_cc=False` are not typos.** Both differ from the port source deliberately (design §3), and both are pinned by tests in Task 5.
- **Check `nvidia-smi` before Task 7 Step 5 or Task 9's inpaint path.** FLUX needs ~12 GB; at the time of writing ~10 GB was free. If it will not fit, run the stitch path and say the inpaint path is unverified — do not lower the guard to make it fit.
- **`sushi` masks the plate, not the food** (`findings/2026-07-28-masking-result.md` §3). If you smoke-test that case, the repair will be meaningless and that is expected, not a bug in this doc.
- Expect `stitch` to look worse than `inpaint` where lighting differs. That is R3, it is predicted, and it is a finding to report rather than a defect to fix here.
