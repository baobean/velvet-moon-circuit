from graft.prompt import NEUTRAL_TOKEN, compose_prompt, compose_ravel_prompt, concept_token
from graft.schema import AttributeNode, ConceptKG, PartNode


def test_prompt_mentions_concept_and_vision_attrs_first():
    kg = ConceptKG(
        "monkey puzzle tree",
        [AttributeNode("silhouette", "symmetric candelabra", "vision")],
        [PartNode("leaf", [AttributeNode("shape", "stiff triangular scales", "vision")], None, {})],
        {},
        "conifer",
        [],
        ["r.jpg"],
    )
    p = compose_prompt(kg, neutralize=False)
    assert p.lower().startswith("a photo of a monkey puzzle tree")
    assert "stiff triangular scales" in p
    assert "symmetric candelabra" in p


def test_prompt_respects_max_attrs():
    attrs = [AttributeNode(f"attr{i}", f"val{i}", "vision") for i in range(10)]
    kg = ConceptKG("x", attrs, [], {}, "y", [], [])
    p = compose_prompt(kg, max_attrs=3)
    assert p.count("val") == 3


def test_ravel_prompt_uses_given_attrs():
    p = compose_ravel_prompt("durian", ["spiky husk", "strong odor"], neutralize=False)
    assert p.lower().startswith("a photo of a durian")
    assert "spiky husk" in p and "strong odor" in p


def test_neutralize_replaces_head_but_keeps_attrs():
    kg = ConceptKG("monkey puzzle tree",
                   [AttributeNode("silhouette", "symmetric candelabra", "vision")],
                   [], {}, "", [], ["r.jpg"])
    p = compose_prompt(kg, neutralize=True)
    assert p.lower().startswith(f"a photo of a {NEUTRAL_TOKEN}")
    assert "monkey puzzle tree" not in p.lower()
    assert "symmetric candelabra" in p            # ref-derived attrs stay


def test_concept_token():
    assert concept_token("Avocado", True) == "plant"
    assert concept_token("Avocado", False) == "Avocado"
