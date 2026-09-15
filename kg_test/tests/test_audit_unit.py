from graft.audit import generated_image_name, summarize_audit


def test_generated_image_name_matches_the_eval_driver():
    """run_eval writes eval_images/{method}_{name_mode}_ip{ip_scale}.png, so the
    audit must address the SAME file -- auditing a Phase-1 'ours.png' leftover
    would rest the prune decision on pre-sprint images."""
    assert generated_image_name("neutral", 0.6) == "ours_neutral_ip0.6.png"
    assert generated_image_name("named", 0.4) == "ours_named_ip0.4.png"


def test_summarize_audit_hit_rates():
    recs = [
        {"part": "leaf", "ref_box": True,  "gen_box": True,  "sim": 0.7},
        {"part": "leaf", "ref_box": True,  "gen_box": False, "sim": None},
        {"part": "bark", "ref_box": False, "gen_box": False, "sim": None},
    ]
    s = summarize_audit(recs)
    assert s["leaf"]["ref_hit_rate"] == 1.0
    assert s["leaf"]["gen_hit_rate"] == 0.5
    assert s["leaf"]["mean_sim"] == 0.7           # only scored (sim not None) count
    assert s["bark"]["ref_hit_rate"] == 0.0
