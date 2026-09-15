"""Metric arithmetic, proved without a model.

Every test here runs on synthetic arrays. The GPU on this machine is routinely
held by someone else (findings/2026-07-28-regeneration-result.md §4), and a
metric you cannot check on CPU is a metric you stop checking.
"""
import numpy as np
import pytest
from PIL import Image

from ragregen import metrics
from ragregen.metrics import (EvalEncoders, crop_to_mask, dino_identity,
                              preservation, prompt_alignment, split_refs)


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


def test_a_clearly_positive_sample_gives_a_positive_lower_bound():
    deltas = [0.10, 0.12, 0.09, 0.11, 0.13, 0.10, 0.12, 0.11]
    lo, hi = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert lo > 0
    assert hi > lo


def test_the_interval_brackets_the_sample_mean():
    deltas = [0.10, -0.02, 0.04, 0.08, 0.00, 0.06]
    lo, hi = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert lo <= float(np.mean(deltas)) <= hi


def test_the_ci_is_deterministic_for_a_seed():
    #: A headline that moves between report runs is not a headline.
    deltas = [0.05, -0.01, 0.03, 0.00, 0.02]
    a = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=3)
    b = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=3)
    assert a == b


def test_an_empty_sample_has_no_interval():
    #: Not (0.0, 0.0). An interval over nothing is not a number, and printing
    #: zeros would read as a measured null result.
    assert metrics.paired_bootstrap_ci([], n_boot=100, seed=0) is None


def test_a_single_case_has_no_interval_either():
    #: Every resample of one value is that value, so the interval collapses
    #: to zero width: "+0.070 [+0.070, +0.070] no_harm" reads as
    #: extraordinary precision when it is an artifact of resampling a
    #: degenerate sample. Reported honestly means declining to report it --
    #: below MIN_CI_N the answer is None and the table renders "—".
    assert metrics.paired_bootstrap_ci([0.07], n_boot=100, seed=0) is None


def test_the_floor_is_the_boundary_the_constant_names():
    #: One below the floor gets nothing; exactly the floor gets an interval.
    #: With 11 bridge controls mostly expected healthy, a 1-2 case repair
    #: stratum is a likely outcome, not a corner case.
    assert metrics.MIN_CI_N == 5
    short = [0.01] * (metrics.MIN_CI_N - 1)
    assert metrics.paired_bootstrap_ci(short, n_boot=200, seed=0) is None
    exact = [0.01] * metrics.MIN_CI_N
    assert metrics.paired_bootstrap_ci(exact, n_boot=200, seed=0) is not None


def test_the_margin_is_the_value_the_spec_fixed():
    #: Doc 5 §2 fixes -0.02 before any data exists. Changing it after seeing
    #: deltas is the failure this constant prevents.
    assert metrics.MARGIN == -0.02


def test_a_lower_bound_above_the_margin_is_no_harm():
    assert metrics.delta_verdict(-0.01, 0.05) == "no_harm"


def test_an_upper_bound_below_zero_is_harm():
    assert metrics.delta_verdict(-0.09, -0.03) == "harm"


def test_an_interval_spanning_the_margin_is_inconclusive():
    assert metrics.delta_verdict(-0.06, 0.04) == "inconclusive"


def test_a_lower_bound_exactly_on_the_margin_is_no_harm():
    #: The boundary is decided here, not by whoever reads the table.
    assert metrics.delta_verdict(-0.02, 0.01) == "no_harm"
