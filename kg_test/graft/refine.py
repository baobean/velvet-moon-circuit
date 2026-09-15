"""M3 refine loop: re-seed generation until the verify report is ok, or the
seed budget runs out (Phase 1 uses re-seeding; targeted Kontext region-edits
of failing parts are Phase 2 -- spec section 7).

Each attempt runs in its own subprocess via worker_generate_verify.py rather
than calling generate_lever_a/verify in-process. This was forced by
repeated, differently-signatured segfaults under sustained model load/unload
churn within one long-lived process (three distinct crash signatures across
four attempts: two traced to Python's cyclic GC racing with a *different*
model's C-extension construction -- see env.py's docstring for both fixes
attempted -- and a third with no GC involvement at all, deep inside JSON
parsing during tokenizer loading). Rather than keep chasing the exact
native-library interaction, each attempt now gets a fresh process that
unconditionally releases everything (Python heap and GPU memory alike) on
exit, regardless of what internal state caused prior crashes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from typing import List, Tuple

from PIL import Image

from graft.schema import ConceptKG
from graft.verify import PartScore, VerifyReport


def pick_best(candidates: List[Tuple]):
    """candidates: (item, ok: bool, score: float). Prefer any ok candidate,
    highest score among ok ones; if none ok, the single highest score."""
    return max(candidates, key=lambda c: (c[1], c[2]))[0]


def classify_returncode(code: int) -> str:
    if code == 0:
        return "ok"
    if code == 2:
        return "error"   # usage/config bug — a real problem, surface it
    return "skip"        # 3 (OOM) or any crash/segfault — no signal, try the next seed


def _report_score(report: VerifyReport) -> float:
    sims = [p.sim for p in report.part_scores]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return 0.5 * mean_sim + 0.5 * report.attr_pass


def _run_attempt(kg_json_path: str, cfg_yaml_path: str, seed: int, workdir: str):
    """Run one worker_generate_verify.py subprocess. Returns
    ``(result, error_message, (returncode, stderr))``: `result` is
    (image, report, prompt) on success and None otherwise. `error_message` is
    None for an OOM (exit code 3) OR any other crash/segfault (code 1, -11,
    ...), signaling "no result, but not a bug -- try the next seed"; only exit
    code 2 (worker usage/config error) yields a non-None error_message that
    fails the case. The trailing (returncode, stderr) pair is always returned
    so refine() can report what it ACTUALLY observed when every seed failed,
    rather than asserting a cause it never saw."""
    image_path = os.path.join(workdir, f"attempt_{seed}.png")
    report_path = os.path.join(workdir, f"attempt_{seed}_report.json")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "graft.worker_generate_verify",
            kg_json_path,
            cfg_yaml_path,
            str(seed),
            image_path,
            report_path,
        ],
        capture_output=True,
        text=True,
    )
    status = (proc.returncode, proc.stderr or "")
    kind = classify_returncode(proc.returncode)
    if kind == "error":
        return (
            None,
            f"worker_generate_verify usage/config error (seed={seed}):\n{proc.stderr}",
            status,
        )
    if kind == "skip":
        sys.stderr.write(
            f"[refine] seed {seed} produced no signal (code={proc.returncode}); trying next\n"
        )
        return None, None, status
    # kind == "ok": load image + report as before
    image = Image.open(image_path).convert("RGB")
    image.load()  # force pixels into memory before workdir is cleaned up
    with open(report_path) as f:
        report_dict = json.load(f)
    part_scores = [PartScore(**p) for p in report_dict["part_scores"]]
    report = VerifyReport(
        part_scores=part_scores,
        attr_pass=report_dict["attr_pass"],
        failing_parts=report_dict["failing_parts"],
        ok=report_dict["ok"],
    )
    return (image, report, report_dict["prompt"]), None, status


def refine(kg: ConceptKG, models, cfg) -> Tuple[Image.Image, VerifyReport, str]:
    """Generate up to 1 + cfg.n_refine seeds (each in its own subprocess),
    verifying each, stopping early on the first ok report; otherwise
    returns the best-scoring attempt. `models` is accepted for call-site
    compatibility but unused -- each subprocess builds its own Models()."""
    del models
    with tempfile.TemporaryDirectory(prefix="graft_refine_") as workdir:
        kg_json_path = os.path.join(workdir, "kg.json")
        kg.to_json(kg_json_path)
        cfg_yaml_path = os.path.join(workdir, "cfg.yaml")
        cfg.to_yaml(cfg_yaml_path)

        candidates = []
        last_code, last_stderr = None, ""
        for seed in range(cfg.n_refine + 1):
            result, error, (last_code, last_stderr) = _run_attempt(
                kg_json_path, cfg_yaml_path, seed, workdir
            )
            if error is not None:
                raise RuntimeError(error)
            if result is None:
                continue  # no signal on this seed (OOM or crash) -- try the next one
            image, report, prompt = result
            score = _report_score(report)
            candidates.append(((image, report, prompt), report.ok, score))
            if report.ok:
                break

        if not candidates:
            # Post-Task-12 any nonzero code except 2 lands here, so name the
            # code and stderr actually observed instead of guessing "OOM".
            raise RuntimeError(
                f"refine: every attempt (0..{cfg.n_refine}) failed; "
                f"last exit code={last_code}. Last stderr:\n{last_stderr[-2000:]}"
            )
        return pick_best(candidates)
