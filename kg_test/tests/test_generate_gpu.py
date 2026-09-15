import pytest
from PIL import Image

pytestmark = pytest.mark.gpu


def test_generate_lever_a_produces_image():
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.generate import generate_lever_a
    from graft.models import Models
    from graft.schema import ConceptKG

    kg = ConceptKG.from_json("tests/fixtures/sample_kg.json")
    cfg = GraftConfig(neutralize_name=False)
    models = Models(cfg)

    image, prompt = generate_lever_a(kg, models, cfg, seed=0)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.width >= 768 and image.height >= 768
    assert kg.concept in prompt
