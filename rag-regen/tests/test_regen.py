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


def test_prepare_reference_accepts_the_real_mask_result_shape():
    from ragregen.mask import MaskResult

    arr = np.zeros((64, 64), dtype=float)
    arr[16:32, 16:32] = 1.0
    found = MaskResult(mask=arr, box=(16, 16, 32, 32), score=0.8,
                       n_candidates=2, selected_by="confidence")
    ref = Image.new("RGB", (64, 64), (5, 6, 7))
    out, info = regen.prepare_reference(
        ref, "durian", mode="crop", masker=lambda image, phrase: found)
    assert out.size == (26, 26)
    assert info["mask_info"]["selected_by"] == "confidence"
    assert info["subject_frac_before"] == pytest.approx(0.0625)


def test_edit_prompt_forbids_reference_background_transfer():
    prompt = regen.edit_prompt("an axolotl in an aquarium", "salamander",
                               "axolotl")
    assert "preserve" in prompt.lower()
    assert "do not copy" in prompt.lower()
    assert "background" in prompt.lower()
    assert "an axolotl in an aquarium" in prompt


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


def test_inpainter_passes_the_declared_seed_to_diffusers():
    pipe = FakePipe()
    inp = regen.Inpainter(regen.KontextConfig(seed=123), pipe=pipe)
    out = inp.regen(_draft(), _mask(), None, "repaint it")
    assert pipe.calls[0]["generator"].initial_seed() == 123
    assert out.meta["seed"] == 123


def test_per_call_seed_overrides_the_config_seed():
    pipe = FakePipe()
    inp = regen.Inpainter(regen.KontextConfig(seed=123), pipe=pipe)
    out = inp.regen(_draft(), _mask(), None, "repaint it", seed=456)
    assert pipe.calls[0]["generator"].initial_seed() == 456
    assert out.meta["seed"] == 456


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


def test_free_vram_below_the_floor_refuses_rather_than_thrashing():
    """A shared 4090. Refusing is correct behaviour, not a bug to work around."""
    with pytest.raises(RuntimeError, match="12.0 GB"):
        regen.check_vram(free_gb=9.6)


def test_free_vram_above_the_floor_passes():
    regen.check_vram(free_gb=18.0)          # must not raise


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
