import pytest
from PIL import Image

from ragregen import models


def test_build_crop_scorer_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown crop scorer"):
        models.build_crop_scorer("nope", device="cpu")


def test_build_crop_scorer_refuses_fgclip_and_names_the_escape_hatch():
    """fgclip is a known SCORER_IDS key but cannot load under this
    environment's pinned transformers -- build_crop_scorer must refuse it
    up front with a message that names the working alternative, not fail
    deep inside model construction. See models.py's module docstring."""
    with pytest.raises(RuntimeError, match="siglip_so400m_384"):
        models.build_crop_scorer("fgclip", device="cpu")


def test_scorer_ids_cover_the_configured_choices():
    assert "siglip_so400m_384" in models.SCORER_IDS
    assert "fgclip" in models.SCORER_IDS


def test_scorer_ids_match_validate_encoder_dims_keys():
    """SCORER_IDS keys must line up with validate.ENCODER_DIMS where they overlap,
    or a scorer picked in pipeline.yaml can silently mismatch the dimension
    check in validate.py."""
    from ragregen.validate import ENCODER_DIMS

    for key in models.SCORER_IDS:
        assert key in ENCODER_DIMS, (
            f"SCORER_IDS has '{key}' but ENCODER_DIMS does not know it")


@pytest.mark.gpu
def test_dino_detector_finds_a_box_on_a_real_image():
    det = models.DinoDetector(device="cuda")
    img = Image.open("tests/fixtures/two_dogs.jpg")
    boxes = det.all_boxes(img, "dog", max_boxes=8)
    assert len(boxes) >= 2
    for box, conf in boxes:
        assert len(box) == 4
        assert 0.0 <= conf <= 1.0


@pytest.mark.gpu
def test_siglip_scorer_prefers_the_matching_phrase():
    sc = models.build_crop_scorer("siglip_so400m_384", device="cuda")
    dog = Image.open("tests/fixtures/dog_crop.jpg")
    assert sc.score(dog, "a dog") > sc.score(dog, "a violin")


@pytest.mark.gpu
def test_siglip_scorer_output_is_in_unit_interval():
    sc = models.build_crop_scorer("siglip_so400m_384", device="cuda")
    dog = Image.open("tests/fixtures/dog_crop.jpg")
    for phrase in ("a dog", "a violin"):
        s = sc.score(dog, phrase)
        assert 0.0 <= s <= 1.0


# FG-CLIP cannot run in this env at all (see models.py's module docstring
# and build_crop_scorer's guard) -- these are bake-off-only tests, skipped
# here rather than deleted so Task 13's separate fgclip env can re-enable
# them (drop the skip marker, and call FGCLIPScorer directly since
# build_crop_scorer("fgclip") is unconditionally blocked in this codebase).
FGCLIP_SKIP = (
    "FG-CLIP's remote code targets transformers ~4.12 and cannot load under the "
    "5.14.1 this env pins for Qwen3-VL/FLUX-Kontext. It is a bake-off-only "
    "candidate, run from a separate conda env (see Task 13)."
)


@pytest.mark.gpu
@pytest.mark.skip(reason=FGCLIP_SKIP)
def test_fgclip_scorer_prefers_the_matching_phrase():
    sc = models.build_crop_scorer("fgclip", device="cuda")
    dog = Image.open("tests/fixtures/dog_crop.jpg")
    assert sc.score(dog, "a dog") > sc.score(dog, "a violin")


@pytest.mark.gpu
@pytest.mark.skip(reason=FGCLIP_SKIP)
def test_fgclip_scorer_output_is_in_unit_interval():
    sc = models.build_crop_scorer("fgclip", device="cuda")
    dog = Image.open("tests/fixtures/dog_crop.jpg")
    for phrase in ("a dog", "a violin"):
        s = sc.score(dog, phrase)
        assert 0.0 <= s <= 1.0


@pytest.mark.gpu
def test_sam_segmenter_masks_inside_the_box_it_was_given():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), (240, 240, 240))
    ImageDraw.Draw(img).ellipse((64, 64, 192, 192), fill=(200, 40, 40))

    seg = models.SamSegmenter()
    m = seg.mask_from_box(img, (64.0, 64.0, 192.0, 192.0))

    assert m.shape == (256, 256)
    assert m.dtype == bool
    assert m.any(), "SAM returned nothing for a box around an obvious object"
    # The disc is inside the box, so nothing far outside it should be masked.
    assert not m[:32, :32].any()
