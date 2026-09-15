from ragregen.detector_policy import build_policy_manifest, policy_sha256, DEGENERATE_RULE


def _manifest(**over):
    base = dict(
        thresholds={"text": 0.5, "retrieved_reference": 0.25048828125},
        semantic={"model_id": "qwen2.5-vl", "weight_sha256": "s1", "template": "T", "decoding": {"max_new_tokens": 256}},
        reranker={"model_id": "qwen3-vl-reranker-2b", "weight_sha256": "r1", "prompt": "P", "max_image_side": 448},
        retrieval={"encoder": "siglip_so400m_384", "encoder_weight_sha256": "e1",
                   "index_sha256": "i1", "corpus": "laion100k",
                   "query_builder": "ragregen.retrieve.reference_query", "k": 5,
                   "ref_prep": "crop_to_mask+thumbnail448"},
        generator={"model_id": "flux-kontext", "weight_sha256": "g1"},
        env_manifest={"torch": "2.6.0", "sentence-transformers": "5.4.0"},
        code_snapshot_sha256="c1",
        degenerate_behaviour=DEGENERATE_RULE)
    base.update(over)
    return build_policy_manifest(**base)


def test_manifest_captures_full_rule_and_semantic_branch():
    m = _manifest()
    assert m["protocol_status"] == "frozen_before_score"
    assert "semantic_failure OR" in m["rule"]
    assert m["thresholds"] == {"text": 0.5, "retrieved_reference": 0.25048828125}
    assert m["degenerate_behaviour"]["semantic_parse_failure"] == "fail_closed_route"
    assert m["semantic"]["weight_sha256"] == "s1"
    assert m["generator"]["weight_sha256"] == "g1"


def test_policy_sha256_is_order_independent():
    a = _manifest(); b = _manifest()
    assert policy_sha256(a) == policy_sha256(b)
