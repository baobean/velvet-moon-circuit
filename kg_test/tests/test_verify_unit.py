from graft.verify import PartScore, aggregate


def test_aggregate_flags_failing_parts_and_overall():
    ps = [PartScore("leaf", 0.7, True), PartScore("bark", 0.3, False)]
    r = aggregate(ps, attr_pass=0.8, sim_thresh=0.5, attr_thresh=0.6)
    assert r.failing_parts == ["bark"]
    assert r.ok is False  # a part failed

    r2 = aggregate([PartScore("leaf", 0.7, True)], 0.9, 0.5, 0.6)
    assert r2.ok is True


def test_aggregate_fails_on_low_attr_pass_even_if_parts_ok():
    ps = [PartScore("leaf", 0.9, True)]
    r = aggregate(ps, attr_pass=0.3, sim_thresh=0.5, attr_thresh=0.6)
    assert r.failing_parts == []
    assert r.ok is False


import numpy as np
from graft.verify import verify
from graft.schema import AttributeNode, ConceptKG, PartNode

class _FakeVLM:
    def ask(self, image, question): return "yes"
class _FakeDetector:
    def __init__(self, found): self.found = found
    def detect(self, image, phrase): return [(0, 0, 4, 4)] if self.found else []
class _FakeEmb:
    def embed_image(self, images): return np.ones((len(images), 3)) / np.sqrt(3)
class _FakeModels:
    def __init__(self, found):
        self.vlm, self.detector = _FakeVLM(), _FakeDetector(found)
        self.siglip = self.dino = _FakeEmb()
    def unload(self, name): pass

class _Cfg:
    part_sim_threshold = 0.5; attr_pass_threshold = 0.6
    def __init__(self, use_part_tree): self.use_part_tree = use_part_tree

def _kg_with_one_scored_part():
    part = PartNode("leaf", [AttributeNode("shape", "stiff scales", "vision")],
                    "crop.png", {"siglip2": [1/np.sqrt(3)]*3, "dino": [1/np.sqrt(3)]*3})
    unscored = PartNode("bark", [], None, {})  # no box at build -> empty embeddings
    return ConceptKG("x", [], [part, unscored], {}, "", [], ["r.jpg"])

def test_use_part_tree_false_skips_part_scoring():
    r = verify(object(), _kg_with_one_scored_part(), _FakeModels(found=True), _Cfg(False))
    assert r.part_scores == [] and r.failing_parts == []

def test_scores_only_box_derived_parts_and_no_whole_image_fallback():
    r = verify(object(), _kg_with_one_scored_part(), _FakeModels(found=False), _Cfg(True))
    names = [p.part for p in r.part_scores]
    assert names == ["leaf"]                         # 'bark' (empty embeddings) never scored
    leaf = r.part_scores[0]
    assert leaf.present is False and leaf.sim == 0.0  # no box in gen -> 0, not a whole-image sim
