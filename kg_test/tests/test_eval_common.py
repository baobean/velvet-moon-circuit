import json
from graft.eval_common import image_id, id_to_filename, kg_is_usable, build_manifest, SWEPT_METHODS, prompt_for, exemplar_for, rows_from_tables
from graft.schema import AttributeNode, ConceptKG

def test_image_id_and_filename():
    iid = image_id("Egyptian lotus", "ours", "neutral", 0.6, 1)
    assert iid == "Egyptian lotus|ours|neutral|ip0.6|seed1"
    assert id_to_filename(iid) == "Egyptian lotus_ours_neutral_ip0.6_seed1.png"
    # None ip_scale (b0/b2) stays literal
    assert image_id("Guava", "b0", "named", None, 0) == "Guava|b0|named|ipNone|seed0"

def test_kg_is_usable(tmp_path):
    good = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"], ref_embeddings=[[0.1], [0.2]])
    p = tmp_path / "kg.json"; good.to_json(str(p))
    assert kg_is_usable(str(p)) is True
    bad = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"])  # no ref_embeddings
    q = tmp_path / "bad.json"; bad.to_json(str(q))
    assert kg_is_usable(str(q)) is False
    assert kg_is_usable(str(tmp_path / "missing.json")) is False

def test_build_manifest_swept_vs_unswept_and_seeds():
    m = build_manifest(
        species_names=["A"], methods=["ours", "b0", "b1"],
        name_modes=["neutral", "named"], ip_scales=[0.4, 0.6], seeds=2,
    )
    # ours,b1 swept over 2 ip x 2 name x 2 seeds = 8 each; b0 unswept: 2 name x 2 seeds = 4
    n = lambda meth: sum(1 for r in m if r["method"] == meth)
    assert n("ours") == 8 and n("b1") == 8 and n("b0") == 4
    # b0 rows carry ip_scale None
    assert all(r["ip_scale"] is None for r in m if r["method"] == "b0")
    # ids are unique and self-describing
    ids = [r["image_id"] for r in m]
    assert len(ids) == len(set(ids))
    assert "A|b0|neutral|ipNone|seed1" in ids

def _kg():
    return ConceptKG("monkey puzzle tree",
                     [AttributeNode("silhouette", "symmetric candelabra", "vision")],
                     [], {}, "", [], ["a.jpg", "b.jpg", "c.jpg"],
                     ref_embeddings=[[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])

def test_prompt_for_heads():
    kg = _kg()
    assert prompt_for("b0", "Avocado", kg, [], neutralize=True) == "a photo of a plant"
    assert prompt_for("b0", "Avocado", kg, [], neutralize=False) == "a photo of a Avocado"
    assert prompt_for("ours", "Avocado", kg, [], neutralize=True).startswith("a photo of a plant")
    assert "symmetric candelabra" in prompt_for("ours", "Avocado", kg, [], neutralize=True)
    assert prompt_for("b2", "durian", kg, ["spiky husk"], neutralize=True) == "a photo of a plant, spiky husk"

def test_exemplar_for_medoid_and_none():
    kg = _kg()  # medoid of the 3 rows is index 0 or 1
    assert exemplar_for("ours", kg) in ("a.jpg", "b.jpg")
    assert exemplar_for("b0", kg) is None
    # b1 selects via the KG's stored ref_embeddings too (same medoid rule) --
    # b1's pool IS the KG's build refs, so this is exemplar parity by construction.
    from graft.generate import select_exemplar
    assert exemplar_for("b1", kg) in ("a.jpg", "b.jpg")
    assert exemplar_for("b1", kg) == select_exemplar(kg)

def test_rows_from_tables_picks_best_seed_and_flags_missing():
    man = build_manifest(["A", "B"], ["ours"], ["neutral"], [0.6], seeds=2)
    def metr(dino, aa):
        return {"dino": dino, "siglip2": 0.5, "clip_i": 0.5, "clip_t": 0.1, "attribute_accuracy": aa}
    tables = {
        "A|ours|neutral|ip0.6|seed0": metr(0.30, 0.4),
        "A|ours|neutral|ip0.6|seed1": metr(0.90, 0.8),  # higher attr -> this seed wins
        # species B: no images at all -> its cell is missing
    }
    rows, missing = rows_from_tables(man, tables, {"A": 16, "B": 10}, use_part_tree=False)
    assert len(rows) == 1
    r = rows[0]
    assert r["species"] == "A" and r["method"] == "ours" and r["ip_scale"] == 0.6
    assert r["n_heldout"] == 16
    assert abs(r["dino"] - 0.90) < 1e-9  # winning seed's metrics
    assert {"species": "B", "method": "ours", "name_mode": "neutral", "ip_scale": 0.6} in missing
