import numpy as np
import pytest
from PIL import Image

from ragregen import mask


def _cand(box, conf, mask_arr=None, crop=None):
    """A candidate with a mask covering its own box, unless one is given."""
    if mask_arr is None:
        mask_arr = np.zeros((32, 32), dtype=float)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask_arr[y0:y1, x0:x1] = 1.0
    if crop is None:
        crop = Image.new("RGB", (8, 8))
    return mask.Candidate(box=box, dino_conf=conf, mask=mask_arr, crop=crop)


def test_select_box_without_a_bank_takes_the_highest_dino_confidence():
    lo = _cand((0.0, 0.0, 4.0, 4.0), 0.3)
    hi = _cand((0.0, 0.0, 8.0, 8.0), 0.9)
    assert mask.select_box([lo, hi]) is hi


class FakeClassifier:
    """Scores a crop by its area, standing in for a prototype bank.

    A 'stack of books' prototype outranks any single-book crop, so area is a
    faithful stand-in for the whole-versus-part behaviour under test.
    """

    def __init__(self, known=("books",)):
        self.known = set(known)

    def score(self, crop, phrase):
        if phrase not in self.known:
            return None
        return float(crop.size[0] * crop.size[1])


def test_select_box_with_a_bank_prefers_the_whole_over_the_part():
    """stack_of_books: grounding 'books' returns one box per book.

    The single book wins on detector confidence; the whole stack is the thing
    the prompt is about. Repainting one book out of a stack is not a repair.
    """
    one_book = _cand((0.0, 0.0, 4.0, 4.0), 0.9, crop=Image.new("RGB", (4, 4)))
    whole_stack = _cand((0.0, 0.0, 16.0, 16.0), 0.4, crop=Image.new("RGB", (16, 16)))
    got = mask.select_box([one_book, whole_stack],
                          prototypes=FakeClassifier(), phrase="books")
    assert got is whole_stack


def test_a_single_candidate_ignores_the_bank_entirely():
    """One box is not a choice, and embedding it would be wasted work."""
    only = _cand((0.0, 0.0, 4.0, 4.0), 0.2, crop=Image.new("RGB", (4, 4)))

    class Exploding:
        def score(self, crop, phrase):
            raise AssertionError("must not classify a single candidate")

    assert mask.select_box([only], prototypes=Exploding(), phrase="books") is only


def test_an_unknown_phrase_falls_back_to_confidence():
    lo = _cand((0.0, 0.0, 16.0, 16.0), 0.3, crop=Image.new("RGB", (16, 16)))
    hi = _cand((0.0, 0.0, 4.0, 4.0), 0.9, crop=Image.new("RGB", (4, 4)))
    got = mask.select_box([lo, hi], prototypes=FakeClassifier(known=()),
                          phrase="wombat")
    assert got is hi


def test_selection_rule_is_reported_not_inferred():
    lo = _cand((0.0, 0.0, 16.0, 16.0), 0.3, crop=Image.new("RGB", (16, 16)))
    hi = _cand((0.0, 0.0, 4.0, 4.0), 0.9, crop=Image.new("RGB", (4, 4)))
    assert mask.selection_rule([lo, hi], FakeClassifier(), "books") == "prototype"
    assert mask.selection_rule([lo, hi], None, "books") == "confidence"
    assert mask.selection_rule([hi], FakeClassifier(), "books") == "confidence"
    assert mask.selection_rule([lo, hi], FakeClassifier(known=()),
                               "wombat") == "confidence"


class FakeDetector:
    """Returns pre-programmed (box, score) pairs per phrase."""

    def __init__(self, by_phrase):
        self.by_phrase = by_phrase
        self.asked = []

    def all_boxes(self, image, phrase, max_boxes=8):
        self.asked.append(phrase)
        return list(self.by_phrase.get(phrase, []))[:max_boxes]


class FakeSam:
    """Segments a box into a mask covering exactly that box."""

    def __init__(self, empty_for=()):
        self.empty_for = set(empty_for)
        self.calls = 0

    def mask_from_box(self, image, box):
        self.calls += 1
        arr = np.zeros(image.size[::-1], dtype=bool)
        if tuple(box) in self.empty_for:
            return arr
        x0, y0, x1, y1 = (int(v) for v in box)
        arr[y0:y1, x0:x1] = True
        return arr


IMG = Image.new("RGB", (64, 64), (10, 20, 30))
TWO = [((0.0, 0.0, 8.0, 8.0), 0.9), ((0.0, 0.0, 32.0, 32.0), 0.4)]


def _masker(by_phrase, **kw):
    return mask.Masker(FakeDetector(by_phrase), sam=FakeSam(**kw))


def test_segment_masks_every_candidate_then_selects_one():
    sam = FakeSam()
    m = mask.Masker(FakeDetector({"books": TWO}), sam=sam)
    got = m.segment(IMG, "books")
    assert sam.calls == 2, "SAM runs on every candidate, not only the winner"
    assert got.n_candidates == 2
    assert got.box == (0.0, 0.0, 8.0, 8.0)
    assert got.selected_by == "confidence"


def test_segment_returns_a_float_mask_covering_the_selected_box():
    got = _masker({"books": TWO}).segment(IMG, "books")
    assert got.mask.dtype == np.float64
    assert set(np.unique(got.mask)) <= {0.0, 1.0}
    assert got.mask[0:8, 0:8].all()
    assert not got.mask[8:, 8:].any()


def test_segment_with_a_classifier_reports_the_prototype_rule():
    m = mask.Masker(FakeDetector({"books": TWO}), sam=FakeSam())
    got = m.segment(IMG, "books", prototypes=FakeClassifier())
    assert got.selected_by == "prototype"
    assert got.box == (0.0, 0.0, 32.0, 32.0), "the whole stack, not one book"


ONE = [((16.0, 16.0, 32.0, 32.0), 0.9)]


def test_mask_draft_dilates_so_the_inpainter_gets_slack():
    """A pixel-tight mask leaves a halo of the ORIGINAL object's edge, which
    the inpainter reconstructs -- reintroducing what we asked it to replace."""
    m = _masker({"books": ONE})
    tight = m.segment(IMG, "books")
    grown = mask.mask_draft(IMG, "books", m, dilate_px=4)
    assert grown.mask.sum() > tight.mask.sum()


def test_mask_draft_with_zero_dilation_leaves_the_mask_alone():
    m = _masker({"books": ONE})
    tight = m.segment(IMG, "books")
    same = mask.mask_draft(IMG, "books", m, dilate_px=0)
    assert same.mask.sum() == tight.mask.sum()


def test_mask_reference_never_dilates():
    """Dilating a reference mask mattes background into the cutout."""
    m = _masker({"stack of books": ONE})
    tight = m.segment(IMG, "stack of books")
    ref = mask.mask_reference(IMG, "stack of books", m)
    assert ref.mask.sum() == tight.mask.sum()


def test_each_wrapper_grounds_its_own_phrase():
    """Guard: mask_draft grounds the COARSE term, mask_reference the FINE one.

    The draft is the image that got the fine concept wrong, so grounding it on
    the fine phrase would make mask geometry vary with draft quality. A later
    refactor must not silently swap these (design §4).
    """
    det = FakeDetector({"books": ONE, "stack of books": ONE})
    m = mask.Masker(det, sam=FakeSam())
    mask.mask_draft(IMG, "books", m)
    mask.mask_reference(IMG, "stack of books", m)
    assert det.asked == ["books", "stack of books"]


def test_an_ungrounded_phrase_returns_none_not_an_empty_mask():
    assert _masker({}).segment(IMG, "wombat") is None


def test_an_empty_sam_mask_returns_none():
    """SAM can return nothing for a box DINO was confident about."""
    m = mask.Masker(FakeDetector({"books": ONE}),
                    sam=FakeSam(empty_for=[(16.0, 16.0, 32.0, 32.0)]))
    assert m.segment(IMG, "books") is None


def test_both_wrappers_propagate_none():
    m = _masker({})
    assert mask.mask_draft(IMG, "wombat", m) is None
    assert mask.mask_reference(IMG, "wombat", m) is None


def _kontext_engine():
    """The real prepare_reference, or skip.

    ImageRAG is reference-only (RUNBOOK §7) and not a package, so this binds
    across a sys.path insert rather than an import dependency. Torch and
    diffusers are lazy there, so the import costs ~0.2s and touches no GPU.
    Doc 2 ports this function into ragregen/regen.py, at which point the skip
    goes away.
    """
    import sys
    from pathlib import Path

    scripts = Path("/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/ImageRAG/scripts")
    if not (scripts / "kontext_engine.py").is_file():
        pytest.skip("ImageRAG/scripts/kontext_engine.py not present")
    sys.path.insert(0, str(scripts))
    try:
        import kontext_engine
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"kontext_engine not importable: {exc}")
    return kontext_engine


def test_mask_reference_output_drives_the_real_prepare_reference():
    """The downstream consumer derives its own bbox from our mask.

    Pins the geometry end to end: mask_reference's mask, unpadded and
    undilated, produces exactly the crop the real consumer computes. It does
    NOT pin dtype -- prepare_reference thresholds at `mask > 0.5` and mattes
    with `clip(mask, 0, 1)`, both of which a bool or 0-255 array would also
    satisfy. Dtype is a call-site convenience, not a contract this can catch.
    """
    ke = _kontext_engine()
    ref = mask.mask_reference(IMG, "stack of books", _masker({"stack of books": ONE}))

    out, info = ke.prepare_reference(
        IMG, "stack of books", mode="crop",
        masker=lambda image, phrase: (ref.mask, {}))

    assert info["applied"] == "crop"
    # The box is (16,16)-(32,32) plus 8% padding and the +4px floor.
    assert info["bbox"] == [11, 11, 37, 37]
    assert out.size == (26, 26)


def test_prepare_reference_reports_none_applied_when_we_return_no_mask():
    """mask_reference returning None must not be dressed up as a crop."""
    ke = _kontext_engine()
    empty = np.zeros((64, 64), dtype=float)
    out, info = ke.prepare_reference(IMG, "wombat", mode="crop",
                                     masker=lambda image, phrase: (empty, {}))
    assert info["applied"] == "none"
    assert out is IMG


def test_free_drops_both_models_so_flux_can_load():
    """Stage-batching requires masking to be fully evictable (design §8)."""
    class Evictable:
        def __init__(self):
            self.freed = False

        def free(self):
            self.freed = True

    det, sam = Evictable(), Evictable()
    m = mask.Masker(det, sam=sam)
    m.free()
    assert (det.freed, sam.freed) == (True, True)
    assert m.detector is None and m.sam is None


def test_free_tolerates_models_without_a_free_method():
    """The fakes in this suite have none, and neither does DinoDetector."""
    m = mask.Masker(FakeDetector({}), sam=FakeSam())
    m.free()
    assert m.detector is None


def test_mask_draft_grounds_the_coarse_term_but_scores_the_fine_one():
    """boston_bull: grounding 'dog' returns the head and the whole dog.

    Scoring 'dog'-ness cannot separate them -- both crops are dog. The
    question that separates them is "which is the Boston bull", so the
    grounding phrase and the scoring phrase must be different terms. The fine
    prototype is also the one that exists: gt_refs give every concept, while
    coarse prototypes need coarse_refs, which the dataset does not carry.
    """
    det = FakeDetector({"dog": [((0.0, 0.0, 8.0, 8.0), 0.9),
                                ((0.0, 0.0, 32.0, 32.0), 0.4)]})
    m = mask.Masker(det, sam=FakeSam())
    got = mask.mask_draft(IMG, "dog", m,
                          prototypes=FakeClassifier(known=("Boston bull",)),
                          score_phrase="Boston bull", dilate_px=0)
    assert det.asked == ["dog"], "grounding still uses the coarse term"
    assert got.selected_by == "prototype"
    assert got.box == (0.0, 0.0, 32.0, 32.0), "the whole dog, not the head"


def test_crop_to_object_returns_the_padded_object_box():
    """Prototypes built from whole photographs encode framing, not the object.

    Cropping each reference to its object before embedding is what makes a
    prototype an OBJECT prototype. Padding mirrors prepare_reference exactly
    (pad_frac of the box, with a 4px floor) so both sides frame alike.
    """
    m = _masker({"durian": ONE})           # box (16, 16, 32, 32) on a 64x64
    out = mask.crop_to_object(IMG, "durian", m, pad_frac=0.08)
    # xs 16..31 -> pw = int(15 * 0.08) + 4 = 5; box (11, 11, 37, 37)
    assert out.size == (26, 26)


def test_crop_to_object_returns_none_when_the_phrase_is_not_grounded():
    assert mask.crop_to_object(IMG, "wombat", _masker({})) is None
