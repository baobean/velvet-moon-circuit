from graft.schema import AttributeNode, ConceptKG, PartNode


def _kg():
    return ConceptKG(
        concept="monkey puzzle tree",
        global_attrs=[AttributeNode("silhouette", "symmetric candelabra", "vision")],
        parts=[
            PartNode(
                "leaf",
                [AttributeNode("shape", "stiff triangular scales", "vision")],
                "crops/leaf.png",
                {"siglip2": [0.1, 0.2], "dino": [0.3]},
            )
        ],
        concept_embeddings={"siglip2": [0.5, 0.5], "dino": [0.9]},
        anchor="conifer",
        delta=["leaves are stiff triangular scales"],
        ref_paths=["refs/1.jpg"],
    )


def test_roundtrip(tmp_path):
    kg = _kg()
    p = tmp_path / "kg.json"
    kg.to_json(str(p))
    kg2 = ConceptKG.from_json(str(p))
    assert kg2.concept == "monkey puzzle tree"
    assert kg2.parts[0].attributes[0].value == "stiff triangular scales"
    assert kg2.anchor == "conifer"
    assert kg2.parts[0].embeddings["siglip2"] == [0.1, 0.2]


def test_attribute_texts_orders_vision_first():
    kg = _kg()
    kg.global_attrs.append(AttributeNode("age", "old", "llm"))
    texts = kg.attribute_texts()
    assert texts[0].endswith("symmetric candelabra")  # vision before llm
    assert any("leaf" in t and "triangular" in t for t in texts)


def test_ref_embeddings_roundtrip_and_backcompat(tmp_path):
    kg = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"], ref_embeddings=[[0.1, 0.2], [0.3, 0.4]])
    p = tmp_path / "kg.json"
    kg.to_json(str(p))
    back = ConceptKG.from_json(str(p))
    assert back.ref_embeddings == [[0.1, 0.2], [0.3, 0.4]]

    # Phase-1 kg.json without the field still deserializes to [].
    import json
    d = json.loads(p.read_text()); d.pop("ref_embeddings")
    (tmp_path / "old.json").write_text(json.dumps(d))
    assert ConceptKG.from_json(str(tmp_path / "old.json")).ref_embeddings == []
