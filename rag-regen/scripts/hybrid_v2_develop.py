"""CPU-only hybrid-verifier-v2 development study (no GPU, seconds).

Joins the frozen manifest with the already-exposed semantic, reranker, and DINO
artifacts, verifies each input's SHA-256 against the manifest, runs the
leave-one-concept-out evaluation, and writes a complete development report. On a
numeric-PASS readiness it also refits one full-development policy on every row
and freezes it with a hash; a FAIL or INCONCLUSIVE study leaves no frozen policy.

This script performs no model inference and must never be extended to.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import eval_manifest, hybrid_v2_develop
from ragregen import hybrid_verifier_v2 as v2


def _verify_hash(name: str, path: Path, manifest: dict) -> dict:
    source = manifest.get("sources", {}).get(name)
    if not source or "sha256" not in source:
        raise ValueError(f"manifest has no recorded hash for {name}")
    actual = eval_manifest.sha256_file(path)
    if actual != source["sha256"]:
        raise ValueError(
            f"{name} hash mismatch: manifest={source['sha256']} file={actual}")
    return json.loads(path.read_text())


def _fmt_ci(ci) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _confusion_row(name: str, c: dict) -> str:
    return (f"| {name} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} "
            f"| {c['recall']:.3f} | {c['false_positive_rate']:.3f} "
            f"| {c['mcc']:+.3f} |")


def _render(result: dict) -> str:
    det = result["detector"]
    sel = result["selector"]
    v2c = det["v2_guarded"]
    rw = v2c["recall_wilson95"]
    version = result.get("policy_version", "v2")
    lines = [
        f"# Hybrid verifier {version} -- development study",
        "",
        "> This is a **development** evaluation on already-exposed artifacts. "
        "It is not a validation or test estimate. Passing every gate is a "
        "readiness gate for a *new* holdout, not evidence of generalisation.",
        "",
        f"- protocol_status: `{result['protocol_status']}`",
        f"- dataset: `{result.get('dataset')}`",
        f"- readiness: **{result['readiness']}**",
        "",
        "## Source provenance (SHA-256)",
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
        f"v2 recall Wilson 95% CI: [{rw[0]:.3f}, {rw[1]:.3f}].",
        "",
        "| policy | tp | fp | tn | fn | recall | fp_rate | mcc |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
        _confusion_row("v2_guarded", v2c),
    ]
    for name, c in det["baselines"].items():
        c = {**c, "false_positive_rate": c["false_positive_rate"]}
        lines.append(
            f"| {name} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} "
            f"| {c['recall']:.3f} | {c['false_positive_rate']:.3f} "
            f"| {c['mcc']:+.3f} |")

    excl = sel["denominator"]["excluded"]
    lines += [
        "",
        "## Selector (eligible cases only)",
        "",
        f"Denominator: n_eligible={sel['denominator']['n_eligible']}; "
        f"excluded={len(excl)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(excl.items())) or 'none'}).",
        "",
        f"- comparisons: {sel['comparisons']}",
        f"- sign_accuracy: {sel['sign_accuracy']:.4f}",
        f"- harmful (< {v2.HARM_MARGIN}): {sel['harmful']}",
        f"- mean selected DINO delta: {sel['mean_dino_delta']:+.4f}",
        f"- exact-DINO-best rate: {sel['exact_best_rate']:.4f}",
        "",
        "| cohort | n | mean DINO delta | 95% CI (bootstrap) | +/=/- |",
        "|---|--:|--:|:--:|:--:|",
    ]
    for cohort, row in sel["cohorts"].items():
        lines.append(
            f"| {cohort} | {row['n']} | {row['mean_dino_delta']:+.3f} "
            f"| {_fmt_ci(row['dino_delta_95ci'])} "
            f"| {row['improved']}/{row['unchanged']}/{row['worsened']} |")

    lines += [
        "",
        "## Readiness gates (conjunctive)",
        "",
        "| gate | passed | value | threshold | reason |",
        "|---|:--:|---|---|---|",
    ]
    for g in result["gates"]:
        passed = {True: "PASS", False: "FAIL", None: "N/A"}[g["passed"]]
        lines.append(
            f"| {g['name']} | {passed} | `{g['value']}` "
            f"| `{g['threshold']}` | {g['reason']} |")

    mr = result["manual_review"]
    lines += [
        "",
        f"**Manual gate** (`{mr['name']}`): {mr['reason']} "
        f"(resolved={mr['resolved']}). A numeric PASS still requires this "
        f"review before any Phase B work.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_folds_csv(path: Path, folds: list[dict]) -> None:
    fields = ["case_id", "truth", "detector_prediction", "selected",
              "dino_delta", "cohort", "fit_case_ids"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for fold in folds:
            row = {k: fold.get(k) for k in fields}
            row["fit_case_ids"] = " ".join(fold.get("fit_case_ids", []))
            writer.writerow(row)


def _freeze_policy(manifest, semantic, reranker, dino, out_dir: Path) -> None:
    """Refit one policy on *all* rows and freeze it with a hash (PASS only)."""
    det_rows = list(hybrid_v2_develop._detector_rows(
        manifest, semantic, reranker).values())
    sel_rows = list(hybrid_v2_develop._selector_rows(
        manifest, reranker, dino).values())
    det_fit = v2.fit_detector(det_rows)
    sel_fit = v2.fit_selector(sel_rows)
    policy = v2.Policy(det_fit.text_threshold, det_fit.reference_threshold,
                       sel_fit.text_margin, sel_fit.reference_margin)
    payload = {
        "schema": 1,
        "protocol_status": manifest.get("protocol_status"),
        "fitted_on": "all_development_rows",
        "detector_rows": len(det_rows),
        "selector_rows": len(sel_rows),
        "policy": asdict(policy),
        "detector_fit": asdict(det_fit),
        "selector_fit": asdict(sel_fit),
    }
    frozen = out_dir / "frozen_policy.json"
    # Serialise once, hash the exact bytes on disk.
    frozen.write_text(json.dumps(payload, indent=2, sort_keys=True))
    digest = eval_manifest.sha256_file(frozen)
    (out_dir / "frozen_policy.sha256").write_text(
        f"{digest}  frozen_policy.json\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {out}")

    manifest = json.loads(args.manifest.read_text())
    semantic = _verify_hash("semantic", args.semantic, manifest)
    reranker = _verify_hash("reranker", args.reranker, manifest)
    dino = _verify_hash("dino", args.dino, manifest)

    result = hybrid_v2_develop.develop(manifest, semantic, reranker, dino)
    result["sources"] = manifest.get("sources", {})
    result["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    out.mkdir(parents=True, exist_ok=True)
    (out / "development.json").write_text(json.dumps(result, indent=2))
    (out / "development.md").write_text(_render(result))
    _write_folds_csv(out / "folds.csv", result["folds"])

    if result["readiness"] == "PASS":
        _freeze_policy(manifest, semantic, reranker, dino, out)

    print(_render(result), end="")
    print(f"readiness: {result['readiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
