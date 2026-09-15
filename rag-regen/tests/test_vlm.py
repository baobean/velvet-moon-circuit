import pytest
from PIL import Image

from ragregen import vlm

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeProcessor:
    """Mirrors Qwen2_5_VLProcessor, verified with inspect.signature:

        __call__(self, images, text, videos, audio, **kwargs)   <- images FIRST
        apply_chat_template(self, conversation, chat_template, tools,
                            documents, add_generation_prompt, ..., tokenize, ...)

    The parameter order matters. Plan 1 shipped a green suite that crashed on
    the first real case because its fake declared `prompt` first where
    FluxKontextPipeline declared `image` first. This processor has the same
    shape, so the fake reproduces it rather than a convenient reordering.
    """

    def __init__(self):
        self.conversation = None
        self.called_with = None

    def apply_chat_template(self, conversation, chat_template=None, tools=None,
                            documents=None, add_generation_prompt=False,
                            tokenize=True, **kwargs):
        self.conversation = conversation
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "TEMPLATED"

    def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if text is None:
            raise ValueError("text must be provided")
        if images is None:
            raise ValueError("images must be provided")
        self.called_with = {"text": text, "images": images}
        return FakeInputs()

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["  the reply  "]


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeIds())

    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class FakeIds:
    shape = (1, 7)


class FakeModel:
    def __init__(self):
        self.gen_kwargs = None

    def generate(self, **kw):
        self.gen_kwargs = kw
        return [[0] * 12]


def _vlm(model=None, processor=None, max_new_tokens=64):
    v = object.__new__(vlm.QwenVLM)
    v.device = "cpu"
    v.model = model or FakeModel()
    v.processor = processor or FakeProcessor()
    v.max_new_tokens = max_new_tokens
    v.max_image_pixels = None
    v._torch = _FakeTorch()
    return v


class _FakeTorch:
    class no_grad:
        def __enter__(self): return self
        def __exit__(self, *a): return False


def test_ask_returns_the_stripped_reply():
    assert _vlm().ask(IMG, "a fox") == "the reply"


def test_ask_passes_the_image_and_prompt_through_the_chat_template():
    proc = FakeProcessor()
    _vlm(processor=proc).ask(IMG, "an Amur leopard")
    content = proc.conversation[0]["content"]
    assert {"type": "image"} in content
    assert any(c.get("text") == "an Amur leopard" for c in content)
    assert proc.called_with["images"] == [IMG]
    assert proc.called_with["text"] == ["TEMPLATED"]


def test_ask_images_preserves_candidate_then_reference_order():
    proc = FakeProcessor()
    ref = Image.new("RGB", (16, 16), "white")
    _vlm(processor=proc).ask_images([IMG, ref], "compare them")
    assert proc.called_with["images"] == [IMG, ref]
    assert [c["type"] for c in proc.conversation[0]["content"]] == [
        "image", "image", "text"]


def test_multi_image_pixel_cap_preserves_order_and_aspect_ratio():
    proc = FakeProcessor()
    model = _vlm(processor=proc)
    model.max_image_pixels = 100
    wide = Image.new("RGB", (40, 20), "red")
    tall = Image.new("RGB", (20, 40), "blue")

    model.ask_images([wide, tall], "compare")

    resized = proc.called_with["images"]
    assert resized[0].width > resized[0].height
    assert resized[1].height > resized[1].width
    assert all(im.width * im.height <= 100 for im in resized)


def test_chat_template_is_untokenised_with_a_generation_prompt():
    """The processor tokenises on the next call; asking it to tokenise here
    would hand generate() a string instead of tensors."""
    proc = FakeProcessor()
    _vlm(processor=proc).ask(IMG, "a fox")
    assert proc.tokenize is False
    assert proc.add_generation_prompt is True


def test_generation_is_deterministic():
    """A judge that samples would give a different verdict on a re-run, which
    would make the whole C1 measurement irreproducible."""
    model = FakeModel()
    _vlm(model=model).ask(IMG, "a fox")
    assert model.gen_kwargs["do_sample"] is False


def test_max_new_tokens_is_forwarded():
    model = FakeModel()
    _vlm(model=model, max_new_tokens=123).ask(IMG, "a fox")
    assert model.gen_kwargs["max_new_tokens"] == 123


def test_vlm_id_is_the_cached_qwen_2_5():
    """Qwen3-VL is deliberately NOT used: it is reserved as the held-out judge
    (parent spec §5), and this checkpoint is already cached."""
    assert vlm.VLM_ID == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_qwen_vlm_satisfies_the_semantic_vlm_protocol():
    from ragregen.verify.semantic import SemanticVerifier
    verdict = SemanticVerifier(_vlm()).judge(IMG, "a fox")
    assert verdict.raw == "the reply"


@pytest.mark.gpu
def test_real_qwen_answers_a_trivial_question():
    v = vlm.QwenVLM(device="cuda")
    reply = v.ask(Image.new("RGB", (64, 64), (255, 0, 0)),
                  "Reply with exactly one word: what colour is this image?")
    assert isinstance(reply, str) and reply.strip()
    assert "red" in reply.lower()
