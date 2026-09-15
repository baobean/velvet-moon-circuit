from graft.refine import pick_best


from graft.refine import classify_returncode

def test_classify_returncode():
    assert classify_returncode(0) == "ok"
    assert classify_returncode(2) == "error"     # real usage/config bug -> raise
    assert classify_returncode(3) == "skip"      # OOM -> next seed
    assert classify_returncode(-11) == "skip"    # SIGSEGV crash -> next seed (was: fail case)
    assert classify_returncode(1) == "skip"      # generic crash -> next seed


def test_exhaustion_error_reports_the_last_observed_failure(monkeypatch):
    """Post-Task-12 every nonzero code except 2 is a 'skip', so the exhaustion
    message must report what was actually observed instead of asserting OOM."""
    import dataclasses

    import pytest

    from graft import refine as refine_mod
    from graft.config import GraftConfig
    from graft.schema import ConceptKG

    cfg = dataclasses.replace(GraftConfig(), n_refine=1)
    kg = ConceptKG("x", [], [], {}, "", [], ["a.jpg"], ref_embeddings=[[1.0]])

    def _always_skip(kg_json_path, cfg_yaml_path, seed, workdir):
        return None, None, (-11, "Segmentation fault (core dumped)\n")

    monkeypatch.setattr(refine_mod, "_run_attempt", _always_skip)
    with pytest.raises(RuntimeError) as exc:
        refine_mod.refine(kg, None, cfg)
    msg = str(exc.value)
    assert "last exit code=-11" in msg
    assert "Segmentation fault" in msg
    assert "likely repeated OOM" not in msg


def test_pick_best_prefers_ok_then_higher_score():
    cands = [("a", False, 0.4), ("b", True, 0.5), ("c", True, 0.9)]
    assert pick_best(cands) == "c"


def test_pick_best_falls_back_to_highest_score_when_none_ok():
    cands = [("a", False, 0.8), ("b", False, 0.6)]
    assert pick_best(cands) == "a"
