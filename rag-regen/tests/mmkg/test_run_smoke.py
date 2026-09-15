"""Offline smoke test for the Stage-1 orchestrator: FAKE VLM + FAKE Retriever
over a tiny synthetic 4-concept / 4-case set (concept<->case is 1:1, matching
the real labels_holdout48.csv). No GPU, no faiss, no torch -- only PIL
(already required by ragregen.mmkg.vlm_read) and the frozen mmkg modules
under test.
"""
import json
from collections import namedtuple

from PIL import Image

from ragregen.mmkg import run_stage1

Hit = namedtuple("Hit", "path score rank")


class FakeVLM:
    """Implements the ``asker`` protocol: .ask(image, prompt) -> str.

    ``vlm_read.read_slots`` calls ``.convert("RGB")`` on the opened image
    before handing it to the asker, which drops PIL's ``.filename``
    attribute -- so read prompts are answered by decoding the image's
    solid fill color (each fixture image is one flat color) against a
    canned reads table keyed by RGB tuple, not by path. Judge prompts are
    answered "yes" unless the deliberately-mismatched sentinel value
    "clashcolor" appears in the prompt, so exactly one attribute in the
    fixture is engineered to contradict.
    """

    def __init__(self, reads_by_color):
        self.reads_by_color = reads_by_color
        self.model_id = "fake-vlm-1"
        self.n_calls = 0

    def ask(self, image, prompt):
        self.n_calls += 1
        if "essentially the same" in prompt:
            return "no" if "clashcolor" in prompt else "yes"
        pixel = image.getpixel((0, 0))
        return self.reads_by_color.get(pixel, "{}")


class FakeRetriever:
    """Implements Retriever.search(query, k) -> [Hit] over a fixed pool per query."""

    def __init__(self, pool_by_query):
        self.pool_by_query = pool_by_query

    def search(self, query, k):
        items = self.pool_by_query.get(query, [])[:k]
        return [Hit(path=p, score=s, rank=i) for i, (p, s) in enumerate(items)]


def _mk_image(path, color):
    Image.new("RGB", (8, 8), color).save(path)


def test_run_stage1_smoke(tmp_path):
    data = tmp_path / "data"
    data.mkdir()

    # Each fixture image is a solid fill color; FakeVLM decodes its canned
    # read reply from the color (opened+converted images lose .filename).
    # 4 concepts x 3 slice images + 4 drafts = 16 distinct colors.
    COLORS = {
        "a_slice0": (255, 0, 0), "a_slice1": (0, 255, 0), "a_slice2": (0, 0, 255),
        "c_slice0": (255, 255, 0), "c_slice1": (255, 0, 255), "c_slice2": (0, 255, 255),
        "b_slice0": (128, 0, 0), "b_slice1": (0, 128, 0), "b_slice2": (0, 0, 128),
        "d_slice0": (128, 128, 0), "d_slice1": (128, 0, 128), "d_slice2": (0, 128, 128),
        "a_draft": (255, 165, 0), "c_draft": (255, 192, 203),
        "b_draft": (192, 192, 192), "d_draft": (64, 64, 64),
    }
    assert len(set(COLORS.values())) == len(COLORS)  # every color must be unique

    def _mk_slices(prefix, n=3):
        paths = []
        for i in range(n):
            p = str(data / f"{prefix}_slice{i}.png")
            _mk_image(p, COLORS[f"{prefix}_slice{i}"])
            paths.append(p)
        return paths

    # concept "cat_a" (rare cohort): DISPLAY concept "cat a" (space) diverges
    # from its case_id / artifact key "cat_a" (underscore) -- this is the
    # regression sentinel for the case_id-join fix. consensus crimson/round.
    a_slices = _mk_slices("a")
    # concept "cat_c" (rare cohort, no divergence): consensus violet/jagged,
    # draft clashes on one slot -> FAIL.
    c_slices = _mk_slices("c")
    # concept "cat_b" (control cohort, no divergence): consensus azure/square.
    b_slices = _mk_slices("b")
    # concept "cat_d" (control cohort, no divergence): consensus teal/hex,
    # draft has nothing visible on either target slot -> ABSTAIN.
    d_slices = _mk_slices("d")

    a_draft = str(data / "a_draft.png"); _mk_image(a_draft, COLORS["a_draft"])
    c_draft = str(data / "c_draft.png"); _mk_image(c_draft, COLORS["c_draft"])
    b_draft = str(data / "b_draft.png"); _mk_image(b_draft, COLORS["b_draft"])
    d_draft = str(data / "d_draft.png"); _mk_image(d_draft, COLORS["d_draft"])

    reads = {}
    for i in range(3):
        reads[COLORS[f"a_slice{i}"]] = json.dumps(
            {"primary_color": "crimson", "secondary_color": "round"})
        reads[COLORS[f"c_slice{i}"]] = json.dumps(
            {"primary_color": "violet", "secondary_color": "jagged"})
        reads[COLORS[f"b_slice{i}"]] = json.dumps(
            {"primary_color": "azure", "secondary_color": "square"})
        reads[COLORS[f"d_slice{i}"]] = json.dumps(
            {"primary_color": "teal", "secondary_color": "hex"})
    # cat_a: matches the concept exactly on both target slots -> PASS
    reads[COLORS["a_draft"]] = json.dumps({"primary_color": "crimson", "secondary_color": "round"})
    # cat_c: secondary_color deliberately contradicts (sentinel "clashcolor") -> FAIL
    reads[COLORS["c_draft"]] = json.dumps(
        {"primary_color": "violet", "secondary_color": "clashcolor"})
    # cat_b: matches exactly -> PASS
    reads[COLORS["b_draft"]] = json.dumps({"primary_color": "azure", "secondary_color": "square"})
    # cat_d: nothing visible on either target slot -> ABSTAIN
    reads[COLORS["d_draft"]] = json.dumps(
        {"primary_color": "not visible", "secondary_color": "not visible"})

    query_a = "a real photograph of cat_a as the main subject"
    query_c = "a real photograph of cat_c as the main subject"
    query_b = "a real photograph of cat_b as the main subject"
    query_d = "a real photograph of cat_d as the main subject"

    # gt-ref artifacts for the divergent case, keyed by case_id "cat_a" --
    # never resolvable via the space-form display concept "cat a".
    ref_path = str(data / "cat_a_reference.png"); _mk_image(ref_path, (10, 20, 30))
    reserve_path = str(data / "cat_a_reserve.png"); _mk_image(reserve_path, (40, 50, 60))

    holdout = {
        "cohorts": {"rare": {"cases": ["cat_a", "cat_c"]}, "control": {"cases": ["cat_b", "cat_d"]}},
        "references": {"cat_a": {"path": ref_path}},
        "dino_reserves": {"cat_a": [reserve_path]},
    }
    retrieval = {"cases": {
        "cat_a": {"query": query_a, "hits": [
            {"path": p, "score": 0.9 - 0.01 * i, "rank": i} for i, p in enumerate(a_slices)]},
        "cat_c": {"query": query_c, "hits": [
            {"path": p, "score": 0.85 - 0.01 * i, "rank": i} for i, p in enumerate(c_slices)]},
        "cat_b": {"query": query_b, "hits": [
            {"path": p, "score": 0.8 - 0.01 * i, "rank": i} for i, p in enumerate(b_slices)]},
        "cat_d": {"query": query_d, "hits": [
            {"path": p, "score": 0.75 - 0.01 * i, "rank": i} for i, p in enumerate(d_slices)]},
    }}
    stream_b = {"cat_a": {"ok": True}, "cat_c": {"ok": True}, "cat_b": {"ok": False}, "cat_d": {"ok": False}}
    reranker = {"cases": {
        "cat_a": {"scores": {"draft": {"text_relevance": 0.5, "reference_relevance": 0.4}}},
        "cat_c": {"scores": {"draft": {"text_relevance": 0.45, "reference_relevance": 0.35}}},
        "cat_b": {"scores": {"draft": {"text_relevance": 0.3, "reference_relevance": 0.2}}},
        "cat_d": {"scores": {"draft": {"text_relevance": 0.25, "reference_relevance": 0.15}}},
    }}

    holdout_p = tmp_path / "holdout.json"; holdout_p.write_text(json.dumps(holdout))
    retrieval_p = tmp_path / "retrieval.json"; retrieval_p.write_text(json.dumps(retrieval))
    stream_b_p = tmp_path / "stream_b.json"; stream_b_p.write_text(json.dumps(stream_b))
    reranker_p = tmp_path / "retrieved_reranker.json"; reranker_p.write_text(json.dumps(reranker))

    labels_p = tmp_path / "labels.csv"
    with open(labels_p, "w", newline="") as f:
        f.write("case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n")
        # cat_a's DISPLAY concept "cat a" (space) diverges from its case_id
        # "cat_a" (underscore) -- the sentinel for the case_id-join fix.
        f.write(f"cat_a,a cat a,cat a,{a_draft},,pass,\n")
        f.write(f"cat_c,a cat c,cat_c,{c_draft},,fail,\n")
        f.write(f"cat_b,a cat b,cat_b,{b_draft},,pass,\n")
        f.write(f"cat_d,a cat d,cat_d,{d_draft},,fail,\n")

    out_dir = tmp_path / "out"
    pool_by_query = {
        query_a: [(p, 0.5) for p in a_slices],
        query_c: [(p, 0.5) for p in c_slices],
        query_b: [(p, 0.5) for p in b_slices],
        query_d: [(p, 0.5) for p in d_slices],
    }
    fake_vlm = FakeVLM(reads)

    result = run_stage1.main(
        ["--out", str(out_dir), "--holdout", str(holdout_p), "--retrieval", str(retrieval_p),
         "--labels", str(labels_p), "--stream-b", str(stream_b_p), "--reranker", str(reranker_p),
         "--k-slice", "40"],
        vlm_factory=lambda: fake_vlm,
        retriever_factory=lambda: FakeRetriever(pool_by_query),
    )

    # result.json written with gate/decision keys
    result_path = out_dir / "result.json"
    assert result_path.is_file()
    on_disk = json.loads(result_path.read_text())
    assert on_disk == result
    assert "gate" in result and {"recall_ok", "fp_ok", "decision"} <= set(result["gate"])
    assert result["decision"] in {"PASS", "NEGATIVE"}
    assert result["n_cases"] == 4
    assert result["n_concepts"] == 4

    # per-cohort feasibility breakdown (Step-3 stop condition needs this)
    by_cohort = result["feasibility"]["by_cohort"]
    assert set(by_cohort) == {"rare", "control"}
    for coh in ("rare", "control"):
        assert {"decidable", "total"} <= set(by_cohort[coh])
    assert by_cohort["rare"] == {"decidable": 2, "total": 2}       # cat_a, cat_c: decidable
    assert by_cohort["control"] == {"decidable": 2, "total": 2}    # cat_b, cat_d: decidable

    # The join-key regression sentinel: cat_a's per-concept artifacts (slice
    # re-retrieval, gt-ref paths, cohort) must resolve via its case_id
    # "cat_a", NOT via its space-form display concept "cat a" -- otherwise
    # retrieval_cases.get("cat a") misses, the slice re-search query is "",
    # the slice comes back empty, and the cohort/gt-refs are lost too.
    assert result["slice_sizes"]["cat_a"] == 3                     # non-empty slice
    assert "cat_a" not in result["undersized_slices"]

    # at least one trace file exists -- in fact one per case
    trace_dir = out_dir / "trace"
    trace_files = sorted(trace_dir.glob("*.json"))
    assert len(trace_files) == 4

    traces = {f.stem: json.loads(f.read_text()) for f in trace_files}
    verdicts = {cid: t["verifier"]["verdict"] for cid, t in traces.items()}
    assert verdicts == {
        "cat_a": "PASS", "cat_c": "FAIL", "cat_b": "PASS", "cat_d": "ABSTAIN",
    }

    cat_a_trace = traces["cat_a"]
    assert cat_a_trace["concept"] == "cat a"                        # display form preserved
    assert cat_a_trace["cohort"] == "rare"                          # correct cohort via case_id
    assert cat_a_trace["mmkg_concept"]["slice_size"] == 3           # non-empty slice
    assert cat_a_trace["mmkg_concept"]["built_from_slice"] == a_slices
    # gt-ref paths resolved via case_id, not the space-form concept
    assert cat_a_trace["scores_reused"]["dino_reserve_paths"] == [reserve_path]
    assert run_stage1._gt_ref_paths(holdout, "cat_a") == [ref_path, reserve_path]
    assert run_stage1._gt_ref_paths(holdout, "cat a") == []         # space form must NOT resolve

    assert fake_vlm.n_calls > 0  # the fake asker was actually driven
