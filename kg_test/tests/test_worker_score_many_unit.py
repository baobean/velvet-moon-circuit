# tests/test_worker_score_many_unit.py
from graft.worker_score_many import score_plan

def test_score_plan_only_includes_generated_images():
    manifest = [
        {"image_id": "A|ours|neutral|ip0.6|seed0", "species": "A"},
        {"image_id": "A|ours|neutral|ip0.6|seed1", "species": "A"},
    ]
    on_disk = {"A_ours_neutral_ip0.6_seed0.png"}  # seed1 never generated
    plan = score_plan(manifest, out_dir="OUT",
                      exists_fn=lambda p: p.split("/")[-1] in on_disk)
    assert [p["image_id"] for p in plan] == ["A|ours|neutral|ip0.6|seed0"]
    assert plan[0]["img_path"].endswith("A/gen_fast/A_ours_neutral_ip0.6_seed0.png")
