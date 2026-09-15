# tests/test_generate_many_gpu.py
"""1-species real-weights smoke for worker_generate_many: SDXL+IP-Adapter is
loaded once, every planned cell across {ours, b0, b1} is generated and saved.

Run manually on the shared 4090:
    python -m pytest tests/test_generate_many_gpu.py -q -m gpu
"""
import json
import os
import shutil

import pytest
from PIL import Image

pytestmark = pytest.mark.gpu

KG_FIXTURE = "tests/fixtures/sample_kg.json"


def test_generate_many_creates_every_planned_png(tmp_path):
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.eval_common import build_manifest, id_to_filename
    from graft.schema import ConceptKG
    from graft.worker_generate_many import main

    kg = ConceptKG.from_json(KG_FIXTURE)
    species = kg.concept  # "Ashok"

    manifest = build_manifest(
        [species], methods=["ours", "b0", "b1"], name_modes=["neutral"],
        ip_scales=[0.6], seeds=1,
    )

    cfg_yaml = tmp_path / "cfg.yaml"
    GraftConfig(outputs_dir=str(tmp_path)).to_yaml(str(cfg_yaml))

    manifest_json = tmp_path / "manifest.json"
    manifest_json.write_text(json.dumps(manifest))

    kg_dir = tmp_path / "kg"
    (kg_dir / species).mkdir(parents=True)
    shutil.copy(KG_FIXTURE, kg_dir / species / "kg.json")

    b2_json = tmp_path / "b2.json"
    b2_json.write_text(json.dumps({species: []}))

    # b1 selects its exemplar as the medoid of this pool
    pool_json = tmp_path / "pool.json"
    pool_json.write_text(json.dumps({species: kg.ref_paths}))

    out_dir = tmp_path / "out"

    rc = main([str(cfg_yaml), str(manifest_json), str(kg_dir), str(b2_json),
               str(pool_json), str(out_dir)])
    assert rc == 0

    assert len(manifest) == 3  # ours, b0, b1 -- one seed each
    for row in manifest:
        expected = out_dir / species / "gen_fast" / id_to_filename(row["image_id"])
        assert os.path.exists(expected), f"missing {expected}"
        with Image.open(expected) as im:
            assert im.width >= 768 and im.height >= 768
