# tests/test_worker_generate_many_unit.py
from graft.worker_generate_many import generate_plan
from graft.schema import ConceptKG


def _kg(concept):
    return ConceptKG(concept, [], [], {}, "", [], ["a.jpg", "b.jpg"],
                     ref_embeddings=[[1.0, 0.0], [0.9, 0.2]])


def test_generate_plan_resolves_prompts_and_skips_existing():
    manifest = [
        {"image_id": "A|ours|neutral|ip0.6|seed0", "species": "A", "method": "ours",
         "name_mode": "neutral", "ip_scale": 0.6, "seed": 0},
        {"image_id": "A|b0|neutral|ipNone|seed0", "species": "A", "method": "b0",
         "name_mode": "neutral", "ip_scale": None, "seed": 0},
    ]
    done = {"A_b0_neutral_ipNone_seed0.png"}  # b0 already generated
    plan = generate_plan(
        manifest, kg_by_species={"A": _kg("A")}, b2_attrs={"A": []},
        species_pool={"A": ["a.jpg", "b.jpg"]}, out_dir="OUT",
        exists_fn=lambda p: p.split("/")[-1] in done,
    )
    ids = [p["image_id"] for p in plan]
    assert ids == ["A|ours|neutral|ip0.6|seed0"]        # b0 skipped as existing
    item = plan[0]
    assert item["prompt"].startswith("a photo of a plant")
    assert item["ip_path"] in ("a.jpg", "b.jpg")        # medoid exemplar
    assert item["out_path"].endswith("A_ours_neutral_ip0.6_seed0.png")


def test_generate_plan_b1_resolves_medoid_from_kg_ref_embeddings():
    # b1's pool IS the KG's build refs, so b1 resolves its exemplar from the
    # KG's stored ref_embeddings (same as ours) -- not by re-embedding the pool.
    # species_pool paths deliberately differ from kg.ref_paths here: if b1 ever
    # fell back to blindly taking pool[0] this assertion would catch it.
    manifest = [
        {"image_id": "A|b1|neutral|ip0.6|seed0", "species": "A", "method": "b1",
         "name_mode": "neutral", "ip_scale": 0.6, "seed": 0},
    ]
    plan = generate_plan(
        manifest, kg_by_species={"A": _kg("A")}, b2_attrs={"A": []},
        species_pool={"A": ["p0.jpg", "p1.jpg"]}, out_dir="OUT",
        exists_fn=lambda p: False,
    )
    from graft.generate import select_exemplar
    assert plan[0]["ip_path"] == select_exemplar(_kg("A"))  # KG medoid, non-None
    assert plan[0]["ip_path"] == "a.jpg"                    # not pool[0] ("p0.jpg")
