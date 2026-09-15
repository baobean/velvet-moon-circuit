"""Per-run directories and per-case reasoning traces.

Two rules inherited from ../rag-edit, both learned the hard way:

1. Every run gets its own timestamped directory, so an ablation cannot
   overwrite the baseline it exists to be compared against.
2. Every raw VLM reply is recorded. When a RAG pipeline underperforms its own
   baseline the cause is almost never the algorithm -- it is one silently
   garbage intermediate that nothing logged.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ragregen import env

GARBAGE_SIGNATURE = "!!!!"
"""Qwen's vision tower emits this when its softmax NaNs out."""


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "diffusers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _gpu() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return [{"index": i, "name": torch.cuda.get_device_name(i),
                 "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                for i in range(torch.cuda.device_count())]
    except (RuntimeError, AssertionError):
        # A failed CUDA kernel poisons later driver calls. Provenance must
        # still be committed so the queue remains resumable.
        return []


@dataclass
class RunDir:
    path: Path
    argv: list[str]
    args: dict

    def case_dir(self, case_id: str) -> Path:
        d = self.path / case_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, name: str, obj) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def finish(self, status: str, results: dict | None = None) -> None:
        self.write_json("run.json", {
            "argv": self.argv,
            "args": self.args,
            "status": status,
            "results": results or {},
            "versions": _versions(),
            "gpu": _gpu(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        })


def open_run(tag: str, argv: list[str] | None = None, args: dict | None = None,
             root: Path | None = None) -> RunDir:
    root = root or (env.PROJECT_ROOT / "outputs")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{tag}_{stamp}"
    suffix = 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            path = root / f"{tag}_{stamp}_{suffix}"

    latest = root / f"{tag}_latest"
    if latest.is_symlink() or (latest.exists() and not latest.is_dir()):
        latest.unlink()
    elif latest.is_dir():
        raise RuntimeError(
            f"{latest} is a directory, not the expected symlink. "
            f"Remove or rename it, then re-run."
        )
    latest.symlink_to(path.name)

    return RunDir(path=path, argv=argv if argv is not None else list(sys.argv),
                  args=args or {})


@dataclass
class CaseTrace:
    case_id: str
    vlm_calls: list[dict] = field(default_factory=list)
    grounded_scores: list[dict] = field(default_factory=list)
    retrieval_hits: list[dict] = field(default_factory=list)

    def vlm(self, stage: str, raw: str) -> None:
        self.vlm_calls.append({
            "stage": stage,
            "raw": raw,
            "garbage": GARBAGE_SIGNATURE in (raw or ""),
        })

    def grounded(self, scores: dict) -> None:
        self.grounded_scores.append(
            {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
             for k, v in scores.items()}
        )

    def hits(self, hits: list) -> None:
        self.retrieval_hits.append({"hits": [str(h) for h in hits]})

    def save(self, path: Path) -> Path:
        """Atomic, same as Queue.save. A run gets SIGKILL'd (shared GPU,
        supervise.sh) far more often than it exits cleanly, and this file is
        loaded back and appended to on every subsequent round (see
        run_pipeline._trace) -- a bare write_text truncated mid-write would
        corrupt the one case it touched rather than merely losing progress."""
        payload = json.dumps({
            "case_id": self.case_id,
            "vlm": self.vlm_calls,
            "grounded": self.grounded_scores,
            "retrieval": self.retrieval_hits,
        }, indent=2, default=str)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)
        return path
