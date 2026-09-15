# tests/test_score_many_gpu.py
"""1-species real-weights smoke for worker_score_many: for each embedder in
{dino, siglip, clip, vlm}, exactly ONE model is loaded and every generated PNG
is scored with only that model's metrics. Verifies the no-cross-model-churn
safety envelope and that every scored image lands in the output table with
in-range values.

Run manually on the shared 4090:
    python -m pytest tests/test_score_many_gpu.py -q -m gpu
"""
import json
import os
import shutil

import pytest

pytestmark = pytest.mark.gpu

KG_FIXTURE = "tests/fixtures/sample_kg.json"

_METRIC_KEYS = {
    "dino": {"dino"},
    "siglip": {"siglip2"},
    "clip": {"clip_i", "clip_t"},
    "vlm": {"attribute_accuracy"},
}


def test_score_many_covers_every_image_per_embedder(tmp_path):
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.eval_common import build_manifest, id_to_filename
    from graft.schema import ConceptKG
    from graft.worker_generate_many import main as generate_main
    from graft.worker_score_many import main as score_main
    from graft.worker_score_many import score_plan

    kg = ConceptKG.from_json(KG_FIXTURE)
    species = kg.concept  # "Ashok"

    manifest = build_manifest(
        [species], methods=["ours", "b0"], name_modes=["neutral"],
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

    pool_json = tmp_path / "pool.json"
    pool_json.write_text(json.dumps({species: kg.ref_paths}))

    out_dir = tmp_path / "out"

    # Phase 2: generate the images to be scored.
    assert generate_main([str(cfg_yaml), str(manifest_json), str(kg_dir),
                          str(b2_json), str(pool_json), str(out_dir)]) == 0

    # held-out real references: the concept's own ref images.
    heldout_json = tmp_path / "heldout.json"
    heldout_json.write_text(json.dumps({species: kg.ref_paths[:3]}))

    plan = score_plan(manifest, out_dir=str(out_dir), exists_fn=os.path.exists)
    assert len(plan) == 2, "expected both planned cells generated"
    scored_ids = {p["image_id"] for p in plan}

    for embedder in ("dino", "siglip", "clip", "vlm"):
        out_table = tmp_path / f"table_{embedder}.json"
        rc = score_main([embedder, str(cfg_yaml), str(manifest_json), str(kg_dir),
                         str(heldout_json), str(out_dir), str(out_table)])
        assert rc == 0, f"{embedder} worker failed"

        table = json.loads(out_table.read_text())
        assert set(table) == scored_ids, f"{embedder}: table missed images"

        for iid, row in table.items():
            assert set(row) == _METRIC_KEYS[embedder], f"{embedder}: wrong metric keys for {iid}"
            for key, val in row.items():
                if key == "attribute_accuracy":
                    assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
                else:
                    assert -1.0 <= val <= 1.0, f"{key}={val} out of [-1,1]"

        # filenames on disk match the plan's ids
        for iid in scored_ids:
            assert (out_dir / species / "gen_fast" / id_to_filename(iid)).exists()
