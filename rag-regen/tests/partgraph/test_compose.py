import math
import pytest
from PIL import Image
from ragregen.partgraph.compose import compose_reference


def _imgs():
    return {
        "leaf": Image.new("RGB", (100, 50), (0, 128, 0)),
        "bark": Image.new("RGB", (40, 90), (128, 64, 0)),
        "flower": Image.new("RGB", (60, 60), (200, 0, 120)),
    }


def test_single_canvas_rgb_grid_size():
    out = compose_reference(_imgs(), tile=384)
    assert out.mode == "RGB"
    cols = math.ceil(math.sqrt(3))          # 2
    rows = math.ceil(3 / cols)              # 2
    assert out.size == (cols * 384, rows * 384)


def test_deterministic_default_order():
    a = compose_reference(_imgs(), tile=128)
    b = compose_reference(_imgs(), tile=128)
    assert list(a.getdata()) == list(b.getdata())


def test_empty_raises():
    with pytest.raises(ValueError):
        compose_reference({}, tile=128)
