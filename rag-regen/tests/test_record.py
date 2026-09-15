import json
from pathlib import Path

import pytest

from ragregen.verify import record
from ragregen.verify.grounded import CandidateScore, ConceptScore


def _full() -> ConceptScore:
    cand = CandidateScore(box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.9, sim=0.7,
                          sim_coarse=0.2, sim_proto=0.55, sim_proto_coarse=0.11)
    return ConceptScore("Boston bull", "subject", "PRESENT",
                        box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.9, sim=0.7,
                        sim_coarse=0.2, sim_proto=0.55, sim_proto_coarse=0.11,
                        candidates=(cand,))


def test_round_trip_preserves_every_field():
    s = _full()
    assert record.from_dict(record.to_dict(s, with_state=True)) == s


def test_round_trip_preserves_nones_and_empty_candidates():
    s = ConceptScore("three", "count", "ABSTAIN")
    assert record.from_dict(record.to_dict(s, with_state=True)) == s


def test_with_state_false_omits_state_only():
    d = record.to_dict(_full(), with_state=False)
    assert "state" not in d
    assert d["sim"] == 0.7 and d["kind"] == "subject"


def test_a_bare_state_string_rehydrates_as_the_lossy_old_shape():
    """Old streams.json holds {'phrase': 'MISSING'}. A resume must not die."""
    s = record.from_dict("MISSING")
    assert s.state == "MISSING"
    assert s.sim is None and s.candidates == ()


def test_from_dict_rejects_an_unknown_state():
    with pytest.raises(ValueError, match="not a verifier state"):
        record.from_dict({"state": "BROKEN", "phrase": "x", "kind": "subject"})


def test_from_dict_defaults_a_missing_state_to_abstain_not_present():
    """IMPORTANT 3: the one file shape that legitimately has no `state` is a
    CURRENT stream_a.json -- score_a.py always calls
    record.to_dict(with_state=False), so every entry it writes omits the
    key. Defaulting a missing state to PRESENT would be the same class of
    error this branch exists to fix: when the file does not say, assume the
    most favourable value. ABSTAIN is the state that honestly means "no
    verdict was reached".
    """
    s = record.from_dict({"phrase": "fox", "kind": "subject", "sim": 0.7})
    assert s.state == "ABSTAIN"


def _score_a_dict_before_the_refactor(s):
    """scripts/score_a.py's body at commit 66db22c, preserved verbatim.

    The tau = 0.25 C1 baseline reproduces only if stream_a.json's bytes hold,
    and _write uses json.dumps(indent=2) with no sort_keys -- so key ORDER is
    part of the contract, not an implementation detail.
    """
    return {
        "sim": s.sim,
        "sim_coarse": s.sim_coarse,
        "sim_proto": s.sim_proto,
        "sim_proto_coarse": s.sim_proto_coarse,
        "candidates": [
            {"box": list(c.box), "dino_conf": c.dino_conf, "sim": c.sim,
             "sim_coarse": c.sim_coarse, "sim_proto": c.sim_proto,
             "sim_proto_coarse": c.sim_proto_coarse}
            for c in s.candidates
        ],
        "box": list(s.box) if s.box is not None else None,
        "dino_conf": s.dino_conf,
        "kind": s.kind,
    }


def test_score_a_serialisation_is_byte_identical():
    """record.to_dict(with_state=False) must dump to the exact same bytes
    the pre-refactor score_a.py comprehension produced, key order included.
    """
    cases = (
        _full(),
        ConceptScore("three", "count", "ABSTAIN"),
        ConceptScore("fox", "subject", "PRESENT",
                    box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.8, sim=0.42),
    )
    for s in cases:
        new = record.to_dict(s, with_state=False)
        del new["phrase"]
        assert (json.dumps(new, indent=2)
                == json.dumps(_score_a_dict_before_the_refactor(s), indent=2))


def test_from_dict_reads_pre_66db22c_stream_a_json_without_error():
    """`tests/fixtures/stream_a_golden.json` is a real stream_a.json written
    2026-07-27, before commit 66db22c added candidates/sim_proto persistence
    and dropped `state`. It is NOT a golden of current score_a.py output --
    its entries carry the six-key shape {box, dino_conf, kind, sim,
    sim_coarse, state} and nothing else. It is kept as back-compat coverage:
    `record.from_dict` must still read this exact shape without error, since
    a `--resume` over an old run directory will hand it exactly this.
    """
    golden = json.loads(
        Path("tests/fixtures/stream_a_golden.json").read_text())
    assert golden, "fixture is empty"
    for case_id, case in golden.items():
        for phrase, payload in case.items():
            assert set(payload) == {"box", "dino_conf", "kind", "sim",
                                    "sim_coarse", "state"}, (case_id, phrase)
            s = record.from_dict({**payload, "phrase": phrase})
            assert s.state == payload["state"], (case_id, phrase)
            assert s.sim == payload["sim"], (case_id, phrase)
            assert s.candidates == ()
            assert s.sim_proto is None and s.sim_proto_coarse is None
