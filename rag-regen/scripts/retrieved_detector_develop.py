"""CPU-only retrieved-reference detector-only development study.

Joins the frozen manifest, the semantic decisions, the oracle reranker (for the
oracle_v2 baseline), and the new retrieved-reference reranker, then runs
leave-one-concept-out detector fitting on the retrieved signal. Freezes a
detector-only policy only on a numeric PASS; contamination or incomplete coverage
yields INCONCLUSIVE. No selector is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import detector_develop, eval_manifest
from ragregen import hybrid_verifier_v2 as v2


def _verify_hash(name: str, path: Path, manifest: dict) -> dict:
    src = manifest.get("sources", {}).get(name)
    if not src or "sha256" not in src:
        raise ValueError(f"manifest has no recorded hash for {name}")
    actual = eval_manifest.sha256_file(path)
    if actual != src["sha256"]:
        raise ValueError(f"{name} hash mismatch: manifest={src['sha256']} "
                         f"file={actual}")
    return json.loads(path.read_text())


def _build_rows(manifest, semantic, oracle, retrieved):
    ocases, rcases = oracle.get("cases", {}), retrieved.get("cases", {})
    rows, contaminated = [], []
    for cid, meta in manifest["cases"].items():
        if meta["identity_truth"] not in ("PASS", "FAIL"):
            continue
        for src, name in ((semantic, "semantic"), (rcases, "retrieved reranker"),
                          (ocases, "oracle reranker")):
            if cid not in src:
                raise ValueError(f"{cid}: missing {name}")
        rdraft = rcases[cid]["scores"]["draft"]
        odraft = ocases[cid]["scores"]["draft"]
        rows.append({
            "case_id": cid,
            "truth_fail": meta["identity_truth"] == "FAIL",
            "semantic_ok": bool(semantic[cid].get("ok")),
            "text_relevance": v2._number(rdraft, cid, "text_relevance"),
            "reference_relevance": v2._number(rdraft, cid, "reference_relevance"),
            "oracle_reference_relevance": v2._number(odraft, cid,
                                                     "reference_relevance"),
            "cohort": meta["cohort"],
        })
        if rcases[cid].get("contaminated"):
            contaminated.append(cid)
    return rows, contaminated


def _confusion_row(name, c):
    return (f"| {name} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} "
            f"| {c['recall']:.3f} | {c['false_positive_rate']:.3f} "
            f"| {c['mcc']:+.3f} |")


def _render(result: dict) -> str:
    det = result["detector"]
    rg = det["retrieved_guarded"]
    rw = rg["recall_wilson95"]
    lines = [
        "# Retrieved-reference detector -- development study",
        "",
        "> **Development** evaluation on the exposed 24 drafts with a "
        "*retrieved* LAION reference (not the oracle first-edit reference). "
        "Detector-only: no selector is evaluated. A PASS authorises a fresh "
        "Phase B holdout design -- not production integration or generation.",
        "",
        f"- readiness: **{result['readiness']}**",
        f"- contaminated cases: {result['contaminated'] or 'none'}",
        "",
        "## Provenance (SHA-256)",
        "",
        "| source | sha256 |",
        "|---|---|",
    ]
    for name, src in sorted(result.get("sources", {}).items()):
        lines.append(f"| {name} | `{src.get('sha256', '')}` |")

    lines += [
        "",
        "## Detector (all judgeable identities)",
        "",
        f"Denominator: n={det['denominator']['n']} "
        f"(FAIL={det['denominator']['fail']}, PASS={det['denominator']['pass']}). "
        f"retrieved recall Wilson 95% CI: [{rw[0]:.3f}, {rw[1]:.3f}].",
        "",
        "| policy | tp | fp | tn | fn | recall | fp_rate | mcc |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
        _confusion_row("retrieved_guarded", rg),
    ]
    for name, c in det["baselines"].items():
        lines.append(_confusion_row(name, c))

    lines += ["", "## Readiness gates", "",
              "| gate | passed | value | threshold | reason |",
              "|---|:--:|---|---|---|"]
    for g in result["gates"]:
        passed = {True: "PASS", False: "FAIL", None: "N/A"}[g["passed"]]
        lines.append(f"| {g['name']} | {passed} | `{g['value']}` "
                     f"| `{g['threshold']}` | {g['reason']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_folds_csv(path: Path, folds):
    fields = ["case_id", "truth", "prediction", "feasible",
              "text_threshold", "reference_threshold", "fit_case_ids"]
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for fold in folds:
            row = {k: fold.get(k) for k in fields}
            row["fit_case_ids"] = " ".join(fold.get("fit_case_ids", []))
            w.writerow(row)


def _freeze(rows, out_dir: Path):
    fit = v2.fit_detector([{**r, "reference_relevance": r["reference_relevance"]}
                           for r in rows])
    payload = {
        "schema": 1,
        "policy_version": "retrieved_detector",
        "fitted_on": "all_judgeable_rows",
        "n_rows": len(rows),
        "reference_source": "retrieved_laion",
        "policy": {"text_threshold": fit.text_threshold,
                   "reference_threshold": fit.reference_threshold},
        "detector_fit": {"feasible": fit.feasible, "recall": fit.recall,
                         "false_positives": fit.false_positives,
                         "false_positive_rate": fit.false_positive_rate,
                         "mcc": fit.mcc},
    }
    frozen = out_dir / "frozen_detector_policy.json"
    frozen.write_text(json.dumps(payload, indent=2, sort_keys=True))
    digest = eval_manifest.sha256_file(frozen)
    (out_dir / "frozen_detector_policy.sha256").write_text(
        f"{digest}  frozen_detector_policy.json\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--oracle-reranker", type=Path, required=True)
    ap.add_argument("--retrieved-reranker", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")

    manifest = json.loads(args.manifest.read_text())
    semantic = _verify_hash("semantic", args.semantic, manifest)
    oracle = _verify_hash("reranker", args.oracle_reranker, manifest)
    retrieved = json.loads(args.retrieved_reranker.read_text())

    rows, contaminated = _build_rows(manifest, semantic, oracle, retrieved)
    n_expected = sum(1 for m in manifest["cases"].values()
                     if m["identity_truth"] in ("PASS", "FAIL"))
    result = detector_develop.develop(rows, contaminated=contaminated,
                                      n_expected=n_expected)
    result["sources"] = {
        "semantic": manifest["sources"]["semantic"],
        "oracle_reranker": manifest["sources"]["reranker"],
        "retrieved_reranker": {
            "path": str(args.retrieved_reranker),
            "sha256": eval_manifest.sha256_file(args.retrieved_reranker)},
    }
    result["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    out.mkdir(parents=True, exist_ok=True)
    (out / "development.json").write_text(json.dumps(result, indent=2))
    (out / "development.md").write_text(_render(result))
    _write_folds_csv(out / "folds.csv", result["folds"])
    if result["readiness"] == "PASS":
        _freeze(rows, out)

    print(_render(result), end="")
    print(f"readiness: {result['readiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
