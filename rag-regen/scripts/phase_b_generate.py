#!/usr/bin/env python
"""Conditional generation CLI (GPU, only on a detector PASS).

Design section 6: only on a detector PASS, generate exactly one deterministic
``attempt_1`` from each of the 48 original holdout drafts (rare and control
alike -- every case gets an attempt regardless of routing, since arm2
route-all always replaces the draft). ``ragregen.phase_b_arms.assemble_arms``
then builds the three offline arms from the frozen routing decision (Task 4)
plus each attempt's outcome; a routed case whose attempt failed makes the
whole Phase B study INCONCLUSIVE rather than silently degrading the study to
a smaller n.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.phase_b_arms import PhaseBInconclusive, assemble_arms  # noqa: E402


def _default_generate(case_ids: list[str], screen_run: Path, out_dir: Path) -> dict:
    """Real generation via the frozen open-loop pipeline.

    Best-effort GPU wiring: shells out to ``scripts/run_pipeline.py
    --verifier none --open-loop-attempts 1`` (the frozen generation stack,
    hashed into the Task-1 policy manifest) and reads back its queue.json.
    The Runtime sequence (plan footer) is the operator step that actually
    exercises this against real data; every CLI test here injects a fake
    ``generate_fn`` instead.
    """
    tag = "phase_b_generation"
    run_dir = out_dir / tag
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_pipeline.py"),
          "--screen-run", str(screen_run), "--verifier", "none",
          "--open-loop-attempts", "1", "--tag", tag]
    subprocess.run(cmd, check=True, cwd=str(run_dir.parent if run_dir.parent.is_dir() else Path.cwd()))

    queue = json.loads((run_dir / "queue.json").read_text())["cases"]
    results = {}
    for cid in case_ids:
        status = queue.get(cid, {}).get("status")
        results[cid] = {
            "attempt_ok": status == "repaired",
            "attempt_path": str(run_dir / cid / "attempt_1.png"),
        }
    return results


def _render_md(arms: dict) -> str:
    lines = [f"# Phase B generation -- {arms['status']}", ""]
    if arms["status"] == "inconclusive":
        lines.append(f"reason: {arms['reason']}")
    else:
        failed = [cid for cid, v in arms["cases"].items()
                 if v["arm2"]["generation_failed"]]
        lines.append(f"cases: {len(arms['cases'])}/48")
        lines.append(f"generation_failed (arm2): {len(failed)} -> {failed}")
    return "\n".join(lines) + "\n"


def main(argv=None, *, generate_fn=_default_generate) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-manifest", type=Path, required=True)
    ap.add_argument("--detector-report", type=Path, required=True)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    detector = json.loads(args.detector_report.read_text())
    if detector.get("proceed_to_generation") is not True:
        raise PermissionError(
            "refusing to generate: --detector-report does not have "
            "proceed_to_generation=true (readiness="
            f"{detector.get('readiness')!r})")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {args.out_dir}")

    holdout = json.loads(args.holdout_manifest.read_text())
    rare_ids = holdout["cohorts"]["rare"]["cases"]
    control_ids = holdout["cohorts"]["control"]["cases"]
    all_ids = rare_ids + control_ids
    if len(rare_ids) != 24 or len(control_ids) != 24:
        raise ValueError("holdout manifest cohorts are not exactly 24 + 24")

    routing = detector["cases"]
    missing = [cid for cid in all_ids if cid not in routing]
    if missing:
        raise ValueError(f"--detector-report has no routing for: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_fn(all_ids, args.screen_run, args.out_dir)

    cases = [{
        "case_id": cid,
        "cohort": "rare" if cid in rare_ids else "control",
        "routed": bool(routing[cid]["routed"]),
        "attempt_ok": bool(generated[cid]["attempt_ok"]),
        "draft_path": str(args.screen_run / cid / "draft.png"),
        "attempt_path": generated[cid]["attempt_path"],
    } for cid in all_ids]

    try:
        arm_rows = assemble_arms(cases)
    except PhaseBInconclusive as exc:
        arms = {"status": "inconclusive", "reason": str(exc)}
        (args.out_dir / "arms.json").write_text(json.dumps(arms, indent=2, sort_keys=True))
        (args.out_dir / "generation.md").write_text(_render_md(arms))
        print(f"[phase-b-generate] INCONCLUSIVE: {exc}")
        return 0

    arms = {"status": "complete", "cases": arm_rows}
    (args.out_dir / "arms.json").write_text(json.dumps(arms, indent=2, sort_keys=True))
    (args.out_dir / "generation.md").write_text(_render_md(arms))
    print(f"[phase-b-generate] wrote {len(arm_rows)} cases -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
