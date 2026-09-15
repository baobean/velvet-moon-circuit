import numpy as np
from graft.parttype_graph import induce_parttypes, hub_coherence, valid_hubs

def _norm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)

def test_two_tight_groups_induce_two_hubs():
    rng = np.random.default_rng(0)
    a = _norm(np.array([1.,0,0]) + 0.01*rng.standard_normal((6,3)))
    b = _norm(np.array([0,1.,0]) + 0.01*rng.standard_normal((6,3)))
    labels = induce_parttypes(np.vstack([a,b]), min_cluster_size=3)
    assert len({l for l in labels if l != -1}) == 2

def test_valid_hubs_apply_concept_and_coherence_gates():
    labels = np.array([0,0,0, 1,1,1])
    concepts = ["X","X","X", "P","Q","R"]          # hub 0 = 1 concept; hub 1 = 3 concepts
    coh = {0: 0.9, 1: 0.9}
    assert valid_hubs(labels, concepts, coh) == {1}   # hub 0 fails >=2 concepts
    coh2 = {0: 0.9, 1: 0.4}
    assert valid_hubs(labels, concepts, coh2) == set()  # hub 1 now fails coherence 0.5
