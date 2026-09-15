from ragregen.mmkg_store import schema as s

def test_ids_namespaced():
    assert s.global_id("inat","04486") == "inat:04486"
    assert s.hub_id("treevill","genus","Certhia") == "treevill:genus:Certhia"

def test_attribute_record_targets_only():
    cons = {"bark.texture":{"value":"rough","support":4,"visible_count":5,"is_target":True},
            "leaf.shape":{"value":"ovate","support":2,"visible_count":3,"is_target":False}}
    out = s.attribute_record(cons, source="inat-consensus")
    assert out == {"bark.texture":{"value":"rough","support":4,"visible_count":5,"source":"inat-consensus"}}

def test_build_record_relations_and_validation():
    rec = s.build_record(dataset="inat", species_key="04486", scientific_name="Phoebastria immutabilis",
        common_name=None, taxonomy={"family":"Diomedeidae","genus":"Phoebastria"},
        medoid={"image_path":"m.jpg","embedding_ref":0,"k_images":7,"selection":"siglip-centroid-nearest"},
        candidates=[{"image_path":"m.jpg","embedding_ref":0}], part_crops=[], attributes={}, provenance={})
    assert rec["global_id"] == "inat:04486" and rec["common_name"] is None
    assert rec["dataset"] == "inat" and rec["species_key"] == "04486"
    assert {"type":"instance_of","target":"inat:family:Diomedeidae"} in rec["relations"]
    assert {"type":"instance_of","target":"inat:genus:Phoebastria"} in rec["relations"]
    import pytest
    with pytest.raises(ValueError):
        s.build_record(dataset="inat", species_key="x", scientific_name="", common_name=None,
            taxonomy={"family":"F","genus":"G"}, medoid={"image_path":""}, candidates=[], part_crops=[],
            attributes={}, provenance={})
