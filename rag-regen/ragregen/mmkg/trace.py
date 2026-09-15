from __future__ import annotations
import csv, json, os

def build_trace_record(case_id, concept, cohort, label, *, draft, retrieval,
                       mmkg_concept, attribute_reads, verifier, scores_reused, provenance):
    if not (draft.get("image_path") and "flat" in retrieval and "target_attributes" in mmkg_concept
            and "draft" in attribute_reads and verifier.get("verdict")):
        raise ValueError("trace record missing required sections/keys")
    return {"case_id": case_id, "concept": concept, "cohort": cohort, "human_label": label,
            "draft": draft, "retrieval": retrieval, "mmkg_concept": mmkg_concept,
            "attribute_reads": attribute_reads, "verifier": verifier,
            "scores_reused": scores_reused, "provenance": provenance}

def write_trace(record, out_dir):
    d = os.path.join(out_dir, "trace"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{record['case_id']}.json")
    json.dump(record, open(p, "w"), indent=2)
    return p

def load_labels(csv_path):
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict_identity") or "").strip().lower()
            if v not in {"fail", "pass"}:
                raise ValueError(f"{row.get('case_id')}: bad verdict_identity {v!r}")
            out[row["case_id"]] = v
    return out
