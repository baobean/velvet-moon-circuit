from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph

def test_graph_roundtrips_through_json(tmp_path):
    inst = PartInstanceRec(id="Bamboo::leaf::0", concept="Bamboo", part="leaf",
                           ref_path="a.jpg", crop_path="a_leaf.png", siglip2=[0.1, 0.2])
    hub = PartTypeRec(id=3, part="leaf", member_ids=["Bamboo::leaf::0"],
                      centroid=[0.1, 0.2], coherence=0.71, label="")
    g = PartTypeGraph(part_instances=[inst], part_types=[hub])
    p = tmp_path / "graph.json"
    g.to_json(str(p))
    g2 = PartTypeGraph.from_json(str(p))
    assert g2.part_types[0].coherence == 0.71
    assert g2.instances_by_id()["Bamboo::leaf::0"].concept == "Bamboo"
