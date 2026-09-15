"""Stage-1 MMKG verifier orchestrator: one batch, offline eval, one trace.

Flow (parent spec §7 -- one batch, no re-splicing after scoring):

1. Load the frozen holdout artifacts read-only: ``holdout.json`` (concept
   lists per cohort + ground-truth reference paths for the contamination
   guard), ``retrieval.json`` (the stored per-concept query -- reused
   verbatim, never re-templated -- plus its 5 stored hits, which serve as
   the flat arm's candidate pool), ``labels_holdout48.csv`` (human
   identity labels via :func:`ragregen.mmkg.trace.load_labels`), and two
   sources reused for the ``scores_reused`` cross-tab only, never as
   ground truth: ``stream_b.json`` (the prior semantic-verifier branch's
   per-case ``ok`` verdict) and ``retrieved_reranker.json`` (its
   text/reference relevance scores).
2. **Slice + feasibility (CPU, no VLM yet):** for each concept, re-search
   the index with the *stored* query at ``K_SLICE``, then
   :func:`ragregen.mmkg.build.dedup_and_guard`. Slice sizes are logged;
   any concept whose kept slice is smaller than 3 is flagged in the
   result -- the true decidable count is only known after the read pass,
   so this is a heads-up, not a gate.
3. **VLM read pass (one GPU load):** one ``QwenVLM`` instance reads every
   kept slice image (:func:`ragregen.mmkg.vlm_read.read_slots`) via
   :func:`ragregen.mmkg.build.build_concept`, and every case's draft once.
   Reads are cached by path so the feasibility pass below reuses them
   instead of re-reading.
4. **Retrieval arms (CPU):** :func:`ragregen.mmkg.retrieval_arms.flat_arm`
   over the stored hits vs. :func:`ragregen.mmkg.retrieval_arms.mmkg_arm`
   over the kept slice; divergence + coverage per concept.
5. **Verdict (CPU, judge is VLM):** per case,
   :func:`ragregen.mmkg.verifier.case_verdict` against the concept's
   target attributes and the draft's read, with
   ``judge_fn = lambda x, y, slot: vlm_read.judge_match(x, y, slot, vlm)``.
6. **Trace:** one :func:`ragregen.mmkg.trace.build_trace_record` +
   :func:`ragregen.mmkg.trace.write_trace` per case. Provenance records
   the sha256 of every reused source file plus the VLM id.
7. **Offline eval (CPU):** :func:`ragregen.mmkg.evaluate.confusion`,
   ``gate``, ``crosstab`` and ``stratify`` (by cohort and by
   target-attribute-count bucket) over
   ``{"verdict","label","semantic_ok","cohort","n_target_attributes"}``
   per case. Written to ``<out>/result.json`` alongside Role-1
   divergence/coverage aggregates, the feasibility summary, and slice-size
   flags.

All thresholds (gate recall/fp, slot consensus, verdict rule) live in the
frozen modules this file imports -- nothing is re-declared here.

``main()`` takes injectable factories for the VLM asker and the Retriever
so an offline smoke test can drive the whole flow with fakes and never
import torch or faiss (see ``tests/mmkg/test_run_smoke.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from ragregen.eval_manifest import sha256_file
from ragregen.mmkg import build, evaluate, retrieval_arms, trace, verifier, vlm_read

#: Slice size re-retrieved per concept from the index using the stored query.
K_SLICE = 40


# --------------------------------------------------------------------------
# Default (real) factories -- imported lazily so the offline smoke test never
# needs torch or faiss just because it imported this module.
# --------------------------------------------------------------------------

def _default_vlm_factory():
    from ragregen.vlm import VLM_ID, QwenVLM

    vlm = QwenVLM()
    if not hasattr(vlm, "model_id"):
        vlm.model_id = VLM_ID
    return vlm


def _default_retriever_factory(index_path, encoder_name):
    from ragregen.encoders import build_encoder
    from ragregen.retrieve import Retriever

    encoder = build_encoder(encoder_name)
    return Retriever.from_index(index_path, encoder)


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_json_or_empty(path):
    if not path or not os.path.isfile(path):
        return {}
    return _load_json(path)


def _read_case_rows(csv_path):
    """case_id/concept/draft_path rows, in file order (labels come from trace.load_labels)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _cohort_map(holdout):
    """case_id -> cohort ("rare"/"control"). Holdout cohort case lists are
    already in case_id (underscore) form."""
    out = {}
    for cohort, info in holdout.get("cohorts", {}).items():
        for case_id in info.get("cases", []):
            out[case_id] = cohort
    return out


def _gt_ref_paths(holdout, case_id):
    """Ground-truth reference paths for the contamination guard, keyed by
    case_id (underscore form) -- matches holdout.json's references/
    dino_reserves keys, NOT the space-form display concept."""
    paths = []
    ref = holdout.get("references", {}).get(case_id)
    if ref and ref.get("path"):
        paths.append(ref["path"])
    paths.extend(holdout.get("dino_reserves", {}).get(case_id, []))
    return paths


def _n_target_bucket(n):
    return str(n) if n <= 4 else "5+"


def _aggregate_role1(per_concept_div, per_concept_cov):
    n = len(per_concept_div)
    if n == 0:
        return {"n_concepts": 0}
    return {
        "n_concepts": n,
        "mean_jaccard": sum(d["jaccard"] for d in per_concept_div.values()) / n,
        "frac_selected_ref_identical": sum(
            1 for d in per_concept_div.values() if d["selected_ref_identical"]) / n,
        "mean_mmkg_unit_size": sum(c["mmkg_unit_size"] for c in per_concept_cov.values()) / n,
        "mean_mmkg_part_types": sum(c["mmkg_part_types"] for c in per_concept_cov.values()) / n,
        "frac_flat_top1_in_mmkg_pool": sum(
            1 for c in per_concept_cov.values() if c["flat_top1_in_mmkg_pool"]) / n,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/mmkg_stage1")
    p.add_argument("--holdout", default="outputs/phase_b/holdout.json")
    p.add_argument("--retrieval", default="outputs/phase_b/retrieval.json")
    p.add_argument("--labels", default="outputs/phase_b/labels_holdout48.csv")
    p.add_argument("--stream-b", default="outputs/phase_b/stream_b.json")
    p.add_argument("--reranker", default="outputs/phase_b/retrieved_reranker.json")
    p.add_argument("--index", default="data/laion100k/index.faiss")
    p.add_argument("--encoder", default="siglip_so400m_384")
    p.add_argument("--k-slice", type=int, default=K_SLICE)
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None, *, vlm_factory=None, retriever_factory=None):
    args = _parse_args(argv)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. load frozen sources (read-only) ---------------------------------
    holdout = _load_json(args.holdout)
    retrieval_cases = _load_json(args.retrieval).get("cases", {})
    labels = trace.load_labels(args.labels)
    rows = _read_case_rows(args.labels)
    semantic = _load_json_or_empty(args.stream_b)
    reranker = _load_json_or_empty(args.reranker).get("cases", {})

    cohort_of = _cohort_map(holdout)
    # Every per-concept artifact (holdout cohorts/references/dino_reserves,
    # retrieval.json cases, stream_b.json, retrieved_reranker.json) is keyed
    # by the CSV's ``case_id`` column (underscore form, e.g. "snow_leopard"),
    # never by the human-readable ``concept`` column (space form, e.g. "snow
    # leopard") -- concept<->case is 1:1 here, so ``case_id`` is the join key
    # for every artifact lookup below. ``concept`` is kept only for display
    # (trace record, eval_cases) via ``concept_of``.
    case_ids = sorted({row["case_id"] for row in rows})
    concept_of = {row["case_id"]: row["concept"] for row in rows}

    source_sha256 = {}
    for name, path in (("holdout", args.holdout), ("retrieval", args.retrieval),
                        ("labels", args.labels), ("stream_b", args.stream_b),
                        ("reranker", args.reranker)):
        if path and os.path.isfile(path):
            source_sha256[name] = sha256_file(path)

    vlm = (vlm_factory or _default_vlm_factory)()
    retriever = (retriever_factory or
                 (lambda: _default_retriever_factory(args.index, args.encoder)))()
    vlm_id = getattr(vlm, "model_id", type(vlm).__name__)

    reads_cache = {}

    def read_fn(path):
        if path not in reads_cache:
            reads_cache[path] = vlm_read.read_slots(path, vlm)
        return reads_cache[path]

    def judge_fn(x, y, slot):
        return vlm_read.judge_match(x, y, slot, vlm)

    # --- 2/3/4. per-concept slice, build, retrieval arms ---------------------
    mmkg_concepts = {}
    retrieval_records = {}
    div_by_concept, cov_by_concept = {}, {}
    slice_sizes = {}

    for case_id in case_ids:
        stored = retrieval_cases.get(case_id, {})
        query = stored.get("query", "")
        hits = stored.get("hits", [])

        slice_hits = retriever.search(query, args.k_slice)
        slice_paths = [str(h.path) for h in slice_hits]
        score_by_path = {str(h.path): h.score for h in slice_hits}
        rank_by_path = {str(h.path): h.rank for h in slice_hits}

        mmkg_concept = build.build_concept(
            concept_of[case_id], query, slice_paths, _gt_ref_paths(holdout, case_id), read_fn)
        mmkg_concepts[case_id] = mmkg_concept
        slice_sizes[case_id] = mmkg_concept["slice_size"]

        instances = [{"path": p, "score": score_by_path.get(p), "rank": rank_by_path.get(p),
                      "part_type": None} for p in mmkg_concept["built_from_slice"]]

        flat = retrieval_arms.flat_arm(hits)
        mmkg = retrieval_arms.mmkg_arm(instances, mmkg_concept["target_attributes"])
        div = retrieval_arms.divergence(flat, mmkg)
        cov = retrieval_arms.coverage(flat, mmkg)
        div_by_concept[case_id] = div
        cov_by_concept[case_id] = cov

        retrieval_records[case_id] = {
            "query": query, "k_slice": args.k_slice,
            "flat": flat, "mmkg": mmkg, "divergence": div, "coverage": cov,
        }

    undersized = sorted([c for c, n in slice_sizes.items() if n < 3])

    # feasibility from the (cached) reads actually used to build each concept
    concept_reads = {cid: [reads_cache[p] for p in mmkg_concepts[cid]["built_from_slice"]]
                      for cid in case_ids}
    feasibility = build.feasibility(concept_reads)

    # Per-cohort decidable/total breakdown so the brief's Step-3 stop
    # condition (rare-cohort decidable < 18) is directly readable from
    # result.json without re-deriving it from by_concept. Decidability is
    # reused verbatim from build_concept's own "decidable" flag (itself
    # schema.decidable(target_attributes)) -- the >=2 threshold is never
    # re-declared here.
    by_cohort = {}
    for coh in ("rare", "control"):
        coh_case_ids = [cid for cid in case_ids if cohort_of.get(cid) == coh]
        decidable = sum(1 for cid in coh_case_ids if mmkg_concepts[cid]["decidable"])
        by_cohort[coh] = {"decidable": decidable, "total": len(coh_case_ids)}
    feasibility["by_cohort"] = by_cohort

    # --- 5/6. per-case verdict + trace ---------------------------------------
    eval_cases = []
    trace_paths = []
    for row in rows:
        case_id, concept, draft_path = row["case_id"], row["concept"], row["draft_path"]
        targets = mmkg_concepts[case_id]["target_attributes"]
        draft_read = read_fn(draft_path)
        verdict = verifier.case_verdict(targets, draft_read, judge_fn)

        sem = semantic.get(case_id, {})
        rr = reranker.get(case_id, {}).get("scores", {}).get("draft", {})
        scores_reused = {
            "semantic_ok": sem.get("ok"),
            "reranker_text_relevance": rr.get("text_relevance"),
            "reranker_reference_relevance": rr.get("reference_relevance"),
            "dino_reserve_paths": holdout.get("dino_reserves", {}).get(case_id, []),
        }

        record = trace.build_trace_record(
            case_id, concept, cohort_of.get(case_id), labels.get(case_id),
            draft={"image_path": draft_path, "read": draft_read},
            retrieval=retrieval_records[case_id],
            mmkg_concept=mmkg_concepts[case_id],
            attribute_reads={"draft": draft_read,
                             "concept_reads": concept_reads[case_id]},
            verifier=verdict,
            scores_reused=scores_reused,
            provenance={"vlm_id": vlm_id, "sources_sha256": source_sha256},
        )
        trace_paths.append(trace.write_trace(record, out_dir))

        n_tgt = mmkg_concepts[case_id]["n_target_attributes"]
        eval_cases.append({
            "case_id": case_id, "concept": concept,
            "verdict": verdict["verdict"], "label": labels.get(case_id),
            "semantic_ok": sem.get("ok"), "cohort": cohort_of.get(case_id),
            "n_target_attributes": n_tgt, "n_target_bucket": _n_target_bucket(n_tgt),
        })

    # --- 7. offline eval -------------------------------------------------------
    conf = evaluate.confusion(eval_cases)
    gate = evaluate.gate(conf)
    crosstab = evaluate.crosstab(eval_cases)
    strata = {
        "cohort": evaluate.stratify(eval_cases, "cohort"),
        "n_target_bucket": evaluate.stratify(eval_cases, "n_target_bucket"),
    }

    result = {
        "n_cases": len(eval_cases),
        "n_concepts": len(case_ids),
        "role1_retrieval_arms": {
            "aggregate": _aggregate_role1(div_by_concept, cov_by_concept),
            "per_concept": {cid: {"divergence": div_by_concept[cid], "coverage": cov_by_concept[cid]}
                            for cid in case_ids},
        },
        "feasibility": feasibility,
        "slice_sizes": slice_sizes,
        "undersized_slices": undersized,
        "role2_verifier": {
            "confusion": conf, "gate": gate, "crosstab": crosstab, "strata": strata,
        },
        "gate": gate,
        "decision": gate["decision"],
        "n_abstain": conf["n_abstain"],
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    main()
