"""End-to-end GRAFT pipeline for one concept: M1 (build/reuse MMKG) -> M2+M3
(generate with refine) -> persisted outputs.

Building the MMKG runs in its own subprocess via worker_build_kg.py, for
the same process-isolation reason refine() dispatches each attempt to
worker_generate_verify.py -- see refine.py's module docstring.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from typing import List, Optional

from graft.refine import refine
from graft.schema import ConceptKG


def _build_kg_subprocess(concept: str, build_refs: List[str], cfg, outputs_dir: str) -> None:
    with tempfile.TemporaryDirectory(prefix="graft_build_kg_") as workdir:
        cfg_yaml_path = os.path.join(workdir, "cfg.yaml")
        cfg.to_yaml(cfg_yaml_path)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "graft.worker_build_kg",
                concept,
                outputs_dir,
                cfg_yaml_path,
                *build_refs,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"worker_build_kg failed (code={proc.returncode}):\n{proc.stderr}"
            )


def run_concept(
    concept: str,
    build_refs: List[str],
    models,
    cfg,
    kg: Optional[ConceptKG] = None,
) -> dict:
    """Build the MMKG (unless `kg` is already supplied), generate with the
    M3 refine loop, and persist final.png + report.json under
    outputs/<concept>/. Returns {"image", "report", "prompt", "kg"}.
    `models` is accepted for call-site compatibility but unused -- both the
    build and refine phases run in their own subprocesses."""
    del models
    outputs_dir = os.path.join(cfg.outputs_dir, concept)
    os.makedirs(outputs_dir, exist_ok=True)

    if kg is None:
        kg_path = os.path.join(outputs_dir, "kg.json")
        if not os.path.exists(kg_path):
            _build_kg_subprocess(concept, build_refs, cfg, outputs_dir)
        kg = ConceptKG.from_json(kg_path)

    image, report, prompt = refine(kg, None, cfg)

    image.save(os.path.join(outputs_dir, "final.png"))
    report_dict = {
        "prompt": prompt,
        "attr_pass": report.attr_pass,
        "ok": report.ok,
        "failing_parts": report.failing_parts,
        "part_scores": [asdict(p) for p in report.part_scores],
    }
    with open(os.path.join(outputs_dir, "report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    return {"image": image, "report": report, "prompt": prompt, "kg": kg}
