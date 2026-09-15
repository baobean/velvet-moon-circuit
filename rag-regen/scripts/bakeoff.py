#!/usr/bin/env python
"""FG-CLIP vs SigLIP, evaluated separately in each role (spec §6 Stage 1).

Two bands, because a region-text model can win one and lose the other:
  RETRIEVER   query -> whole-image ranking over the corpus (R@1, R@5, MRR)
  CROP SCORER phrase -> crop similarity, measured as breed-discrimination
              accuracy over the ground-truth references

FG-CLIP cannot run in this environment at all: its remote code targets
transformers ~4.12 and FGCLIPConfig fails to build under the 5.14.1 pinned
here (Task 7). `encoders.build_encoder` and `models.build_crop_scorer` both
refuse it by design. Its arm therefore runs out-of-process against a separate
interpreter passed as --fgclip-python, which writes its scores to disk for
this script to read back. Without that flag the arm is reported as SKIPPED --
never as a zero, which an operator would read as "FG-CLIP lost".

Usage: ./scripts/run.sh bakeoff
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, encoders, env, models, retrieve, trace  # noqa: E402

CANDIDATES = ("siglip_so400m_384", "fgclip")
OUT_OF_PROCESS = {"fgclip"}
ARM_SCRIPT = Path(__file__).resolve().parent / "fgclip_arm.py"


def recall_at_k(hits: list, correct: set, k: int) -> float:
    return 1.0 if any(h.path in correct for h in hits[:k]) else 0.0


def mrr(hits: list, correct: set) -> float:
    for h in hits:
        if h.path in correct:
            return 1.0 / (h.rank + 1)
    return 0.0


def eval_retriever(name: str, ds: config.DatasetConfig, device: str,
                   tmp_index: Path) -> dict:
    """Index every gt_ref, query with each case prompt, score the ranking."""
    enc = encoders.build_encoder(name, device=device)
    all_refs = [r for c in ds.cases for r in c.gt_refs]
    retrieve.build_index(all_refs, enc, tmp_index)
    r = retrieve.Retriever.from_index(tmp_index, enc)

    r1 = r5 = rr = 0.0
    for case in ds.cases:
        hits = r.search(case.prompt, k=5)
        correct = set(case.gt_refs)
        r1 += recall_at_k(hits, correct, 1)
        r5 += recall_at_k(hits, correct, 5)
        rr += mrr(hits, correct)
    n = len(ds.cases)
    env.reclaim_gpu()
    return {"encoder": name, "R@1": r1 / n, "R@5": r5 / n, "MRR": rr / n, "n": n}


def eval_crop_scorer(name: str, ds: config.DatasetConfig, device: str) -> dict:
    """Each gt_ref must score highest against its own concept, not another's."""
    from PIL import Image

    scorer = models.build_crop_scorer(name, device=device)
    concepts = sorted({c.concept for c in ds.cases})

    correct = total = 0
    for case in ds.cases:
        for ref in case.gt_refs:
            img = Image.open(ref).convert("RGB")
            scores = {k: scorer.score(img, k) for k in concepts}
            if max(scores, key=scores.get) == case.concept:
                correct += 1
            total += 1
    env.reclaim_gpu()
    return {"encoder": name, "accuracy": correct / total if total else 0.0,
            "n": total, "n_classes": len(concepts)}


def run_out_of_process(name: str, dataset: Path, device: str, out_dir: Path,
                       python_exe: str | None) -> tuple[dict, dict]:
    """Run a candidate that cannot be imported here, via its own interpreter.

    Returns (retriever_row, crop_scorer_row). Any failure is reported as a
    skipped arm rather than raising: one unusable candidate must not discard
    the other candidate's completed GPU work.
    """
    if not python_exe:
        reason = (f"no --fgclip-python given; {name} cannot load under this "
                  f"env's transformers (see the module docstring)")
        return ({"encoder": name, "skipped": reason},
                {"encoder": name, "skipped": reason})

    if not ARM_SCRIPT.is_file():
        reason = f"worker script missing: {ARM_SCRIPT}"
        return ({"encoder": name, "skipped": reason},
                {"encoder": name, "skipped": reason})

    result_path = out_dir / f"{name}_arm.json"
    cmd = [python_exe, str(ARM_SCRIPT), "--dataset", str(dataset),
           "--device", device, "--out", str(result_path)]
    print(f"  [out-of-process] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        data = json.loads(result_path.read_text())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return ({"encoder": name, "skipped": reason},
                {"encoder": name, "skipped": reason})

    return data["retriever"], data["crop_scorer"]


def _rows(rows: list, header: str, cols: list[str], fmt) -> list[str]:
    lines = [header, "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        if "skipped" in r:
            lines.append(f"| {r['encoder']} | SKIPPED — {r['skipped']} "
                         + "| " * (len(cols) - 2) + "|")
        else:
            lines.append(fmt(r))
    return lines


def render_summary(retriever_rows: list, scorer_rows: list) -> str:
    """The operator-facing report. Pure, so it is testable without a GPU."""
    lines = ["# Encoder bake-off", ""]
    lines += _rows(
        retriever_rows, "## Retriever (query -> whole image)",
        ["encoder", "R@1", "R@5", "MRR"],
        lambda r: (f"| {r['encoder']} | {r['R@1']:.3f} | {r['R@5']:.3f} "
                   f"| {r['MRR']:.3f} |"))
    lines += [""]
    lines += _rows(
        scorer_rows, "## Crop scorer (phrase -> crop)",
        ["encoder", "accuracy", "n", "classes"],
        lambda r: (f"| {r['encoder']} | {r['accuracy']:.3f} | {r['n']} "
                   f"| {r['n_classes']} |"))
    lines += ["", "Set the winners in `configs/pipeline.yaml` as `retriever:` "
              "and `crop_scorer:`. They need not be the same encoder."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fgclip-python", default=None,
                    help="interpreter of the separate fgclip env; without it "
                         "the FG-CLIP arm is reported as skipped")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    if not ds.cases:
        print(f"[bakeoff] no cases in {args.dataset}; nothing to compare.")
        return 2

    run = trace.open_run("bakeoff", argv=sys.argv, args=vars(args))

    retriever_rows, scorer_rows = [], []
    for name in CANDIDATES:
        if name in OUT_OF_PROCESS:
            print(f"[{name}] out-of-process arm", flush=True)
            ret_row, crop_row = run_out_of_process(
                name, args.dataset, args.device, run.path, args.fgclip_python)
            retriever_rows.append(ret_row)
            scorer_rows.append(crop_row)
            continue
        print(f"[retriever] {name}", flush=True)
        retriever_rows.append(
            eval_retriever(name, ds, args.device, run.path / f"{name}.faiss"))
        print(f"[crop scorer] {name}", flush=True)
        scorer_rows.append(eval_crop_scorer(name, ds, args.device))

    summary = render_summary(retriever_rows, scorer_rows)
    (run.path / "summary.md").write_text(summary)
    run.write_json("results.json",
                   {"retriever": retriever_rows, "crop_scorer": scorer_rows})
    run.finish("ok", {"retriever": retriever_rows, "crop_scorer": scorer_rows})
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
