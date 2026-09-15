import pytest
from PIL import Image
from graft.models import SdxlIpGenerator


@pytest.mark.gpu
def test_generate_multi_runs_and_is_deterministic():
    gen = SdxlIpGenerator(ip_scale=0.6)
    refs = [Image.new("RGB", (224, 224), c) for c in ((0, 120, 0), (10, 90, 10))]
    a = gen.generate_multi("a photo of a plant", refs, [0.6, 0.4], seed=0, steps=6)
    b = gen.generate_multi("a photo of a plant", refs, [0.6, 0.4], seed=0, steps=6)
    assert a.size == (1024, 1024)
    assert list(a.getdata()) == list(b.getdata())   # same seed -> identical
