import numpy as np
from PIL import Image
from graft.baselines import b0_vanilla, b1_imagerag

class _CaptureGen:
    def __init__(self): self.prompt = self.ip_image = None
    def generate(self, prompt, ip_image, seed, negative, steps):
        self.prompt, self.ip_image = prompt, ip_image
        return Image.new("RGB", (8, 8))

class _Siglip:
    # first two pool images cluster, third is an outlier -> medoid index 0 or 1
    def embed_image(self, images):
        m = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])[: len(images)]
        return m / np.linalg.norm(m, axis=1, keepdims=True)

class _Models:
    def __init__(self): self.generator, self.siglip = _CaptureGen(), _Siglip()
    def unload(self, name): pass

class _Cfg:
    def __init__(self, neutralize): self.neutralize_name = neutralize

def test_b0_neutralized_head():
    m = _Models(); b0_vanilla("Avocado", m, _Cfg(True), seed=0)
    assert m.generator.prompt == "a photo of a plant"

# One distinct flat color per pool image, so the captured ip_image identifies
# WHICH pool path B1 actually conditioned on (a marker, not just "some image").
_POOL_COLORS = [(10, 0, 0), (20, 0, 0), (30, 0, 0)]

def test_b1_uses_medoid_and_neutral_scaffold(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"; Image.new("RGB", (8, 8), _POOL_COLORS[i]).save(p); paths.append(str(p))
    m = _Models(); b1_imagerag("Avocado", paths, m, _Cfg(True), seed=0)
    assert m.generator.prompt == "a photo of a plant"          # no "According to this image"
    assert "according to this image" not in m.generator.prompt.lower()
    # _Siglip's rows make index 1 the medoid (rows 0/1 cluster, mean sits nearer
    # row 1); B1 must condition on THAT pool image, not simply pool_paths[0].
    assert m.generator.ip_image.getpixel((0, 0)) == _POOL_COLORS[1]
