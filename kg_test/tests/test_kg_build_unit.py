from graft.kg_build import _attr_values, parse_vlm_schema, should_reask
from graft.schema import AttributeNode

RAW = (
    '{"global":{"silhouette":"symmetric candelabra"},'
    '"parts":{"leaf":{"shape":"stiff triangular scales"},"bark":{"texture":"rough"}}}'
)


def test_parse_builds_vision_nodes():
    g, parts = parse_vlm_schema(RAW)
    assert g[0].name == "silhouette" and g[0].source == "vision"
    assert parts["leaf"][0].value == "stiff triangular scales"


def test_parse_tolerates_surrounding_text_and_missing_parts():
    raw = 'Sure, here is the schema:\n{"global":{"color":"green"},"parts":{}}\nHope that helps!'
    g, parts = parse_vlm_schema(raw)
    assert g[0].value == "green"
    assert parts == {}


def test_should_reask_when_mostly_generic():
    assert should_reask(["green", "full", "tree", "stiff scales"], threshold=0.5) is True   # 3/4 generic
    assert should_reask(["stiff triangular scales", "reddish flaking bark"], threshold=0.5) is False


def _vision(*values):
    return [AttributeNode(name=f"a{i}", value=v, source="vision") for i, v in enumerate(values)]


def test_not_visible_does_not_trip_reask():
    """A sanctioned "not visible" is honesty, not vagueness: re-asking would
    pressure the VLM into inventing attributes for parts it cannot see."""
    global_attrs = _vision("not visible", "stiff triangular scales")
    part_attrs = {"leaf": _vision("N/A"), "bark": _vision("reddish flaking bark")}

    values = _attr_values(global_attrs, part_attrs)
    assert sorted(values) == ["reddish flaking bark", "stiff triangular scales"]
    assert should_reask(values) is False


def test_generic_values_still_trip_reask_alongside_not_visible():
    values = _attr_values(_vision("not visible", "green"), {"leaf": _vision("tree")})
    assert values == ["green", "tree"]  # the honest answer is excluded, the vague ones are not
    assert should_reask(values) is True
