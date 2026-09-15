import numpy as np
from graft.transfer import select_borrowed, reference_weights
from graft.parttype_graph import centroid_prototype

def _norm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)

def test_weights_cap_and_ratio():
    # 1 own + 4 borrowed: raw 2 vs 1each -> own 2/6=0.333, borrowed 0.667 (< cap 0.7)
    w = reference_weights(1, 4)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 2/6) < 1e-6
    # 1 own + 20 borrowed would be borrowed 20/22=0.909 -> capped to 0.7
    w2 = reference_weights(1, 20)
    assert abs(w2[1:].sum() - 0.7) < 1e-6
    assert abs(w2[0] - 0.3) < 1e-6

def test_selection_only_differs_by_kind_not_count():
    rng = np.random.default_rng(1)
    pool = _norm(rng.standard_normal((30, 8)))
    concepts = ["own"]*5 + [f"c{i}" for i in range(25)]
    labels = np.array([0]*15 + [1]*15)
    q = pool[0]
    kw = dict(query_embed=q, pool_embeds=pool, pool_concepts=concepts,
              pool_labels=labels, own_concept="own", hub_id=0, k=4, seed=7)
    hub = select_borrowed("hub", **kw)
    raw = select_borrowed("rawnn", **kw)
    rnd = select_borrowed("random", **kw)
    assert len(hub) == len(raw) == len(rnd) == 4          # same count (P4)
    for sel in (hub, raw, rnd):
        assert all(concepts[i] != "own" for i in sel)     # never borrow from own concept

def test_hubnn_is_hub_gated_but_query_ranked():
    # 2 hubs. Query is inside hub 0. hub 1 members are globally closer to the query than
    # hub 0 members -> a plain rawnn would reach into hub 1; hubnn must stay inside hub 0.
    rng = np.random.default_rng(3)
    hub0 = _norm(np.array([[1.0, 0.0, 0.0], [0.98, 0.10, 0.0], [0.97, -0.12, 0.0],
                           [0.95, 0.20, 0.0]], dtype=float))          # own-hub candidates
    hub1 = _norm(np.array([[0.999, 0.01, 0.0], [0.995, 0.02, 0.0]], dtype=float))  # closer to q
    embeds = np.vstack([hub0, hub1])
    concepts = ["own", "b", "c", "d", "e", "f"]
    labels = np.array([0, 0, 0, 0, 1, 1])
    q = embeds[0]
    kw = dict(query_embed=q, pool_embeds=embeds, pool_concepts=concepts,
              pool_labels=labels, own_concept="own", hub_id=0, k=3, seed=0)
    hubnn = select_borrowed("hubnn", **kw)
    rawnn = select_borrowed("rawnn", **kw)
    # hubnn stays inside hub 0 (never picks the globally-closer hub-1 crops)
    assert all(labels[i] == 0 for i in hubnn), "hubnn must not leave the query's hub"
    # rawnn DOES reach the closer hub-1 crops -> the two genuinely differ
    assert any(labels[i] == 1 for i in rawnn), "rawnn should reach outside the hub here"
    # within hub 0, hubnn ranks by query similarity: b(0.98)>c/d over the far one
    assert hubnn[0] == 1  # concept 'b', the query-nearest in-hub candidate
    assert all(concepts[i] != "own" for i in hubnn)

def test_oraclehub_uses_taxonomy_labels_not_embedding():
    # 6 crops. Query 'own' is family-A. Its ONLY same-family sibling ('sibA') is embedding-FAR
    # from the query; the embedding-nearest crops belong to family B. RawNN grabs the B crops;
    # oraclehub must recover the distant sibling because taxonomy (labels) says it's relevant.
    embeds = _norm(np.array([[1.0, 0.0, 0.0],     # own   (query)
                             [0.2, 1.0, 0.0],     # sibA  (same family, embedding-FAR)
                             [0.99, 0.05, 0.0],   # nb1   (family B, embedding-near)
                             [0.98, 0.10, 0.0],   # nb2   (family B, embedding-near)
                             [0.97, 0.12, 0.0],   # nb3   (family B, embedding-near)
                             [0.96, 0.14, 0.0]], dtype=float))  # nb4 (family B)
    concepts = ["own", "sibA", "b1", "b2", "b3", "b4"]
    fam_labels = np.array([0, 0, 1, 1, 1, 1])   # family ids (the ORACLE signal, not embedding)
    kw = dict(query_embed=embeds[0], pool_embeds=embeds, pool_concepts=concepts,
              own_concept="own", k=3, seed=0)
    oracle = select_borrowed("oraclehub", pool_labels=fam_labels, hub_id=0, **kw)
    rawnn = select_borrowed("rawnn", pool_labels=fam_labels, hub_id=0, **kw)
    # oraclehub recovers the embedding-distant sibling that shares the family
    assert [concepts[i] for i in oracle] == ["sibA"]
    # rawnn cannot -- it grabs the embedding-near, taxonomically-unrelated family-B crops
    assert all(concepts[i] != "sibA" for i in rawnn)

def test_hubnn_respects_max_per_concept():
    rng = np.random.default_rng(9)
    # hub 0: concept 'b' has 3 near-query members; cap must hold them to 2
    embeds = _norm(np.array([[1.0, 0, 0], [0.99, 0.01, 0], [0.98, 0.02, 0],
                             [0.97, 0.03, 0], [0.5, 0.5, 0.1]], dtype=float))
    concepts = ["own", "b", "b", "b", "d"]
    labels = np.array([0, 0, 0, 0, 0])
    kw = dict(query_embed=embeds[0], pool_embeds=embeds, pool_concepts=concepts,
              pool_labels=labels, own_concept="own", hub_id=0, k=3, seed=0)
    sel = select_borrowed("hubnn", **kw)
    assert sum(1 for i in sel if concepts[i] == "b") <= 2

def test_weights_uniform_when_no_own():
    # reference_weights(0, 4): no own crops, so no cap applies; uniform borrowed weights
    w = reference_weights(0, 4)
    assert abs(w.sum() - 1.0) < 1e-9
    assert all(abs(wi - 0.25) < 1e-6 for wi in w)

def test_centroid_prototype_max_per_concept_enforced():
    # Build a hub where one concept has 3 members very close to centroid,
    # and multiple other concepts contribute members farther away
    rng = np.random.default_rng(42)
    # Hub 0 members: 3 from "a" (close), 2 from "b" (close), 2 from "c" (close)
    hub_members = np.array([
        [1.0, 0.0, 0.0],      # a - closest
        [0.99, 0.05, 0.0],    # a - close
        [0.98, 0.1, 0.0],     # a - close (3rd, would violate max_per_concept=2)
        [0.95, -0.1, 0.02],   # b - close
        [0.94, -0.15, 0.05],  # b - close
        [0.92, 0.2, -0.1],    # c - farther
        [0.90, 0.25, -0.15],  # c - farther
    ], dtype=float)
    hub_members = _norm(hub_members)

    # Other hubs/points
    others = _norm(rng.standard_normal((10, 3)))
    embeds = np.vstack([hub_members, others])

    labels = np.array([0]*7 + [1]*10)  # hub 0 has indices 0-6, hub 1 has 7-16
    concepts = ["a", "a", "a", "b", "b", "c", "c"] + [f"c{i}" for i in range(10)]

    # Get centroid_prototype with k=4, max_per_concept=2
    result = centroid_prototype(embeds, labels, hub_id=0, concepts=concepts,
                                 k=4, max_per_concept=2, exclude_concept=None)

    # Count from each concept in the result
    a_count = sum(1 for idx in result if concepts[idx] == "a")
    b_count = sum(1 for idx in result if concepts[idx] == "b")
    c_count = sum(1 for idx in result if concepts[idx] == "c")

    # Each concept should have at most 2
    assert a_count <= 2, f"Expected at most 2 from 'a', got {a_count}"
    assert b_count <= 2, f"Expected at most 2 from 'b', got {b_count}"
    assert c_count <= 2, f"Expected at most 2 from 'c', got {c_count}"
    # Should have selected exactly k=4 items
    assert len(result) == 4, f"Expected 4 results, got {len(result)}"
