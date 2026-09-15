import pytest
from PIL import Image, ImageDraw
import numpy as np
from graft.models import SdxlIpGenerator


@pytest.mark.gpu
def test_inpaint_multi_repaints_box_only():
    gen = SdxlIpGenerator(ip_scale=0.6)
    base = Image.new("RGB", (1024, 1024), (180, 140, 90))
    mask = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(mask).rectangle([300, 300, 600, 600], fill=255)
    ref = Image.new("RGB", (256, 256), (20, 160, 40))
    out = gen.inpaint_multi("a photo of a plant leaf", base, mask, [ref], [1.0], seed=0, steps=10)
    assert out.size == base.size
    a, o = np.asarray(base), np.asarray(out)
    assert np.abs(a[:200, :200].astype(int) - o[:200, :200].astype(int)).mean() < 8
    assert np.abs(a[300:600, 300:600].astype(int) - o[300:600, 300:600].astype(int)).mean() > 5
