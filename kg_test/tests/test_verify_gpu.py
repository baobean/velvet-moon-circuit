import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.gpu


def test_loaders_smoke():
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.models import Models

    m = Models(GraftConfig())
    img = Image.new("RGB", (384, 384), (120, 160, 90))

    e = m.siglip.embed_image([img])
    assert e.shape[0] == 1
    assert abs(float(np.linalg.norm(e[0])) - 1.0) < 1e-3  # normalized

    boxes = m.detector.detect(img, "leaf")
    assert isinstance(boxes, list)

    m.unload("siglip")


def test_verify_returns_report():
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.generate import generate_lever_a
    from graft.models import Models
    from graft.schema import ConceptKG
    from graft.verify import VerifyReport, verify

    kg = ConceptKG.from_json("tests/fixtures/sample_kg.json")
    cfg = GraftConfig()
    models = Models(cfg)

    gen, _prompt = generate_lever_a(kg, models, cfg, seed=0)

    report = verify(gen, kg, models, cfg)

    assert isinstance(report, VerifyReport)
    assert len(report.part_scores) <= len(kg.parts)  # only box-derived parts (with stored embeddings) are scored now
    assert 0.0 <= report.attr_pass <= 1.0
