import numpy as np
import pytest

from ragregen.verify import prototype


class FakeEmbedder:
    """Returns a preset row per image, in call order."""

    def __init__(self, rows):
        self.rows = [np.asarray(r, dtype=np.float32) for r in rows]
        self.calls = 0

    def encode_pil(self, images, batch_size=32):
        n = len(list(images))
        out = np.stack(self.rows[self.calls:self.calls + n])
        self.calls += n
        return out


def test_build_prototype_normalises_before_and_after_the_mean():
    # Two vectors of very different magnitude pointing 90 degrees apart. If the
    # magnitudes were not divided out first, the long one would dominate and
    # the prototype would sit near the x axis instead of the diagonal.
    emb = FakeEmbedder([[100.0, 0.0], [0.0, 1.0]])
    p = prototype.build_prototype([object(), object()], emb)
    assert p == pytest.approx([2 ** -0.5, 2 ** -0.5], abs=1e-6)


def test_build_prototype_returns_unit_norm():
    emb = FakeEmbedder([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    p = prototype.build_prototype([object()] * 3, emb)
    assert float(np.linalg.norm(p)) == pytest.approx(1.0, abs=1e-6)


def test_build_prototype_of_a_single_reference_is_that_reference_normalised():
    emb = FakeEmbedder([[0.0, 5.0]])
    assert prototype.build_prototype([object()], emb) == pytest.approx([0.0, 1.0])


def test_build_prototype_rejects_an_empty_reference_set():
    with pytest.raises(ValueError, match="at least one reference"):
        prototype.build_prototype([], FakeEmbedder([]))


def test_bank_returns_none_for_an_unknown_phrase():
    bank = prototype.PrototypeBank({"parrot": np.array([1.0, 0.0])})
    assert bank.get("wombat") is None
    assert bank.score(np.array([1.0, 0.0]), "wombat") is None


def test_bank_scores_cosine_against_the_named_prototype():
    bank = prototype.PrototypeBank({"parrot": np.array([1.0, 0.0])})
    assert bank.score(np.array([1.0, 0.0]), "parrot") == pytest.approx(1.0)
    assert bank.score(np.array([0.0, 1.0]), "parrot") == pytest.approx(0.0)


def test_cosine_is_invariant_to_magnitude():
    a, b = np.array([2.0, 0.0]), np.array([9.0, 0.0])
    assert prototype.cosine(a, b) == pytest.approx(1.0)


from pathlib import Path

from ragregen.config import Case, DatasetConfig


class RecordingEmbedder:
    """Embeds by filename stem, so a prototype's provenance is inspectable."""

    def __init__(self):
        self.seen = []

    def encode_pil(self, images, batch_size=32):
        images = list(images)
        self.seen.extend(images)
        return np.stack([np.array([1.0, 0.0]) for _ in images])


class ExplodingRefs(list):
    def __iter__(self):
        raise AssertionError("the retrieved arm must not read gt_refs")


def _ds(tmp_path, coarse_refs=None):
    (tmp_path / "f.png").write_bytes(b"")
    return DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=[tmp_path / "f.png"])],
        coarse_refs=coarse_refs or {},
    )


def test_ceiling_bank_covers_the_fine_and_coarse_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    ds = _ds(tmp_path, {"parrot": [tmp_path / "f.png"]})
    bank = prototype.bank_from_gt_refs(ds, RecordingEmbedder())
    assert bank.phrases == ["African grey parrot", "parrot"]


def test_ceiling_bank_omits_a_coarse_term_with_no_references(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    bank = prototype.bank_from_gt_refs(_ds(tmp_path), RecordingEmbedder())
    assert bank.phrases == ["African grey parrot"]
    assert bank.get("parrot") is None


def test_ceiling_bank_never_builds_a_prototype_from_an_excluded_image(
        tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    ds = _ds(tmp_path)
    bank = prototype.bank_from_gt_refs(
        ds, RecordingEmbedder(), exclude={tmp_path / "f.png"})
    assert bank.get("African grey parrot") is None


def test_retrieved_bank_does_not_touch_gt_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())

    class FakeRetriever:
        seen = []

        def search(self, query, k):
            self.seen.append(query)
            return [type("H", (), {"path": tmp_path / "f.png"})() for _ in range(k)]

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=ExplodingRefs())],
    )
    retriever = FakeRetriever()
    bank = prototype.bank_from_retrieval(ds, retriever, RecordingEmbedder(), k=2)
    assert bank.phrases == ["African grey parrot", "parrot"]
    assert retriever.seen == [
        "a real photograph of African grey parrot, a type of parrot, as the main subject",
        "a real photograph of parrot as the main subject",
    ]


class _Emb:
    def encode_pil(self, images, batch_size=32):
        return np.stack([np.array([1.0, 0.0]) for _ in list(images)])


def test_crop_classifier_embeds_the_crop_then_scores_it():
    bank = prototype.PrototypeBank({"durian": np.array([1.0, 0.0])})
    clf = prototype.CropClassifier(bank, _Emb())
    assert clf.score(object(), "durian") == pytest.approx(1.0)


def test_crop_classifier_returns_none_for_an_unknown_phrase_without_embedding():
    class Exploding:
        def encode_pil(self, images, batch_size=32):
            raise AssertionError("must not embed for an unknown phrase")

    clf = prototype.CropClassifier(prototype.PrototypeBank({}), Exploding())
    assert clf.score(object(), "wombat") is None


def test_bank_from_gt_refs_applies_a_prepare_hook_per_reference(tmp_path, monkeypatch):
    """Prototypes built from whole photographs encode framing, not the object.

    `prepare` is the seam that lets a caller crop each reference to its object
    first, without prototype.py taking a dependency on the masker.
    """
    monkeypatch.setattr(prototype, "_open", lambda p: "WHOLE")
    seen = []

    def prepare(image, phrase):
        seen.append((image, phrase))
        return "CROPPED"

    class Recorder:
        def encode_pil(self, images, batch_size=32):
            self.got = list(images)
            return np.stack([np.array([1.0, 0.0]) for _ in self.got])

    rec = Recorder()
    ds = _ds(tmp_path)
    prototype.bank_from_gt_refs(ds, rec, prepare=prepare)
    assert seen == [("WHOLE", "African grey parrot")]
    assert rec.got == ["CROPPED"]


def test_prepare_returning_none_falls_back_to_the_whole_image(tmp_path, monkeypatch):
    """An ungrounded reference still has to contribute something."""
    monkeypatch.setattr(prototype, "_open", lambda p: "WHOLE")

    class Recorder:
        def encode_pil(self, images, batch_size=32):
            self.got = list(images)
            return np.stack([np.array([1.0, 0.0]) for _ in self.got])

    rec = Recorder()
    prototype.bank_from_gt_refs(_ds(tmp_path), rec,
                                prepare=lambda image, phrase: None)
    assert rec.got == ["WHOLE"]
