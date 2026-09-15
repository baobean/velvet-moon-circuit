import types

import numpy as np
import pytest

from ragregen import encoders, validate


def test_build_encoder_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown encoder"):
        encoders.build_encoder("nope", device="cpu")


def test_every_encoder_name_has_a_declared_dim():
    for name in encoders.ENCODER_IDS:
        assert name in validate.ENCODER_DIMS, (
            f"{name} is buildable but has no entry in validate.ENCODER_DIMS")


def test_siglip_id_is_the_transformers_checkpoint_not_the_open_clip_one():
    """`timm/ViT-SO400M-14-SigLIP-384` is an open_clip-format repo with no
    config.json; AutoModel/AutoProcessor cannot load it at all (Task 7)."""
    assert (encoders.ENCODER_IDS["siglip_so400m_384"]
            == "google/siglip-so400m-patch14-384")


def test_fgclip_is_refused_as_a_live_encoder_here():
    """FG-CLIP's remote code targets transformers ~4.12 and cannot load under
    5.14.1. It stays a bake-off-only candidate, run from its own env, and the
    refusal must say so rather than failing deep inside from_pretrained."""
    with pytest.raises(RuntimeError, match="bake-off"):
        encoders.build_encoder("fgclip", device="cpu")


def test_fgclip_keeps_its_repo_id_for_the_bakeoff_env():
    assert encoders.ENCODER_IDS["fgclip"] == "qihoo360/fg-clip-base"


# --- embedding width resolution -------------------------------------------
# SiglipConfig has no `projection_dim` (verified against the real cached
# config: AttributeError). Reading it directly would break `dim` for the
# primary encoder, so resolution falls back to the tower hidden size.

def _cfg(**kw):
    return types.SimpleNamespace(**kw)


def test_dim_uses_projection_dim_when_the_model_has_one():
    cfg = _cfg(projection_dim=512,
               text_config=_cfg(hidden_size=768),
               vision_config=_cfg(hidden_size=768))
    assert encoders._resolve_dim(cfg) == 512


def test_dim_falls_back_to_the_text_tower_when_projection_dim_is_absent():
    """The real SigLIP-so400m shape: no projection_dim, hidden_size 1152."""
    cfg = _cfg(text_config=_cfg(hidden_size=1152),
               vision_config=_cfg(hidden_size=1152))
    assert encoders._resolve_dim(cfg) == 1152
    assert encoders._resolve_dim(cfg) == validate.ENCODER_DIMS["siglip_so400m_384"]


def test_dim_reports_an_actionable_error_when_nothing_declares_a_width():
    with pytest.raises(AttributeError, match="embedding width"):
        encoders._resolve_dim(_cfg())


# --- pooled-embedding extraction ------------------------------------------
# transformers 5.x returns the tower's BaseModelOutputWithPooling from both
# SiglipModel and CLIPModel get_image_features/get_text_features; 4.x returned
# a bare tensor. Reading the 4.x contract under 5.x raises AttributeError deep
# inside encode_images, which only the GPU tests reach.

class _FakeOut:
    """Stands in for BaseModelOutputWithPooling."""

    def __init__(self, pooler_output):
        self.pooler_output = pooler_output
        self.last_hidden_state = "not the embedding"


def test_pooled_unwraps_the_transformers_5_output_object():
    assert encoders._pooled(_FakeOut("EMB")) == "EMB"


def test_pooled_passes_a_bare_tensor_through():
    torch = pytest.importorskip("torch")
    t = torch.zeros(2, 4)
    assert encoders._pooled(t) is t


def test_pooled_refuses_an_unrecognised_return():
    with pytest.raises(TypeError, match="pooled embedding"):
        encoders._pooled(object())


class _FakeTorch:
    class no_grad:
        def __enter__(self): return self
        def __exit__(self, *a): return False


class _FakeInputs(dict):
    def to(self, device): return self


class _FakeProcessor:
    def __call__(self, images=None, text=None, **kw):
        self.n = len(images if images is not None else text)
        return _FakeInputs()


class _FakeTower:
    """Returns the 5.x output object, like the real SigLIP/CLIP towers."""

    def __init__(self, dim=3):
        self.dim = dim

    def _out(self, n):
        np = pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")
        return _FakeOut(torch.from_numpy(
            np.arange(n * self.dim, dtype="float32").reshape(n, self.dim)))


class _FakeModel(_FakeTower):
    def get_image_features(self, **kw): return self._out(kw.get("_n", 2))
    def get_text_features(self, **kw): return self._out(kw.get("_n", 2))


def _encoder_without_loading(model):
    """An HFEncoder with its models injected, skipping from_pretrained."""
    enc = object.__new__(encoders.HFEncoder)
    enc.device = "cpu"
    enc.model = model
    enc.processor = _FakeProcessor()
    enc._torch = _FakeTorch()
    return enc


def test_encode_images_returns_the_pooled_embedding_array(tmp_path):
    """Would raise AttributeError('BaseModelOutputWithPooling' has no
    attribute 'float') if the 4.x bare-tensor contract were assumed."""
    from PIL import Image

    paths = []
    for i in range(2):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (8, 8), (i, i, i)).save(p)
        paths.append(p)

    out = _encoder_without_loading(_FakeModel()).encode_images(paths)
    assert out.shape == (2, 3)


def test_encode_text_returns_the_pooled_embedding_array():
    out = _encoder_without_loading(_FakeModel()).encode_text(["a", "b"])
    assert out.shape == (2, 3)


@pytest.mark.gpu
def test_siglip_encoder_dim_matches_the_declared_dim():
    enc = encoders.build_encoder("siglip_so400m_384", device="cuda")
    assert enc.dim == validate.ENCODER_DIMS["siglip_so400m_384"]


@pytest.mark.gpu
def test_siglip_encoder_ranks_the_matching_image_first(tmp_path):
    from PIL import Image

    from ragregen import retrieve

    dog = tmp_path / "dog.jpg"
    Image.open("tests/fixtures/dog_crop.jpg").save(dog)
    noise = tmp_path / "noise.jpg"
    Image.new("RGB", (256, 256), (3, 200, 7)).save(noise)

    enc = encoders.build_encoder("siglip_so400m_384", device="cuda")
    retrieve.build_index([dog, noise], enc, tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", enc)
    assert r.search("a photograph of an animal", k=2)[0].path.stem == "dog"


def test_encode_images_delegates_to_encode_pil(monkeypatch, tmp_path):
    from PIL import Image
    from ragregen import encoders

    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(p)

    enc = encoders.HFEncoder.__new__(encoders.HFEncoder)
    seen = {}

    def fake_encode_pil(images, batch_size=32):
        seen["n"] = len(images)
        seen["mode"] = images[0].mode
        return np.zeros((len(images), 4), dtype=np.float32)

    enc.encode_pil = fake_encode_pil
    out = enc.encode_images([p])
    assert out.shape == (1, 4)
    assert seen == {"n": 1, "mode": "RGB"}


def test_encode_images_never_holds_more_than_one_batch_in_memory(tmp_path):
    """The corpus must not be decoded into RAM all at once.

    Regression test for the build_index failure of 2026-08-06: the LAION
    corpus is 56,657 images averaging 0.61 MB decoded, so materialising every
    one of them before batching needs ~35 GB on a 23 GB box. Four campaign
    attempts died -- SIGKILL from the OOM killer, a SIGABRT, and a swap-thrash
    -- without ever encoding a single batch. Peak RAM must be O(batch_size),
    not O(corpus).
    """
    from PIL import Image
    from ragregen import encoders

    paths = []
    for i in range(10):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i, i, i)).save(p)
        paths.append(p)

    enc = encoders.HFEncoder.__new__(encoders.HFEncoder)
    handed = []

    def fake_encode_pil(images, batch_size=32):
        images = list(images)
        handed.append(len(images))
        return np.zeros((len(images), 4), dtype=np.float32)

    enc.encode_pil = fake_encode_pil
    out = enc.encode_images(paths, batch_size=3)

    assert out.shape == (10, 4)
    assert sum(handed) == 10, "every image must be encoded exactly once"
    assert max(handed) <= 3, (
        f"encode_images decoded {max(handed)} images at once with "
        f"batch_size=3 -- it is loading the whole corpus into RAM")


def test_encode_images_preserves_corpus_order_across_batches(tmp_path):
    """Batch boundaries must not permute rows.

    The FAISS index and its .paths.json sidecar are positional: row i must be
    the embedding of paths[i]. A reordering here silently mislabels every
    retrieval hit rather than failing.
    """
    from PIL import Image
    from ragregen import encoders

    paths = []
    for i in range(7):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i, i, i)).save(p)
        paths.append(p)

    enc = encoders.HFEncoder.__new__(encoders.HFEncoder)

    def fake_encode_pil(images, batch_size=32):
        # Row value = the red channel, i.e. the image's index in `paths`.
        return np.array([[im.getpixel((0, 0))[0]] for im in images],
                        dtype=np.float32)

    enc.encode_pil = fake_encode_pil
    out = enc.encode_images(paths, batch_size=3)
    assert out.ravel().tolist() == [float(i) for i in range(7)]


def test_encode_images_reports_progress_against_the_corpus(tmp_path, capsys):
    """Progress must count the corpus, not the batch.

    Once encode_images drives the batching itself it hands encode_pil one
    batch at a time, so encode_pil's own counter sees start=0, len=32 every
    call and prints "32/32" forever. On a 56,657-image corpus that is 1,771
    identical lines and no way to tell 5% from 95% -- which matters because
    these runs are watched only through the log.
    """
    from PIL import Image
    from ragregen import encoders

    paths = []
    for i in range(100):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(p)
        paths.append(p)

    enc = encoders.HFEncoder.__new__(encoders.HFEncoder)
    enc.encode_pil = lambda images, batch_size=32: np.zeros(
        (len(list(images)), 4), dtype=np.float32)

    enc.encode_images(paths, batch_size=10)

    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "[encode]" in ln]
    assert lines, "encode_images reported no progress at all"
    assert all("/100" in ln for ln in lines), (
        f"progress is not measured against the 100-image corpus: {lines}")
    assert not any("10/10" == ln.split()[-1] for ln in lines), (
        f"per-batch counter leaked into the log: {lines}")
