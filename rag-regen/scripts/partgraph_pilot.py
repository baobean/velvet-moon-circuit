#!/usr/bin/env python
"""wk2 GO/NO-GO pilot: single_medoid vs partgraph, both arms, per species.

For every case (an mmkg-backed species with an explicit `global_id`), drafts
once, masks once, then runs BOTH `partgraph.arms.ARMS` through the same draft
and mask -- `single_medoid` (one retrieved photograph) against `partgraph`
(a composed multi-part canvas) -- and scores each with held-out DINO identity
+ preservation. The frozen decision rule (`partgraph.decide.gain_verdict`)
turns the per-species DINO deltas into one of GAIN / LOSS / NO_LARGE_EFFECT /
UNDERPOWERED (spec §8). GAIN is the only verdict that justifies scaling the
mechanism to Phase 1.

Deliberately does NOT call `regen.prepare_reference` on the partgraph
reference: `prepare_reference`'s object-detection crop/matte step is built for
a natural photograph, and the composed canvas from `compose_reference` is a
synthetic multi-part grid, not a photograph of anything -- there is no single
object in it for a detector to ground. Feeding it through untouched is not an
oversight; it is the whole point of the mechanism under test.

Usage:
  ./scripts/partgraph_pilot.py --store /path/to/mmkg_store --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from ragregen import config, draft, env, mask, metrics, models, regen, trace  # noqa: E402
from ragregen.partgraph.arms import ARMS  # noqa: E402
from ragregen.partgraph.decide import gain_verdict  # noqa: E402
from ragregen.partgraph.pilot import run_case  # noqa: E402


def print_gpu_info() -> None:
    gpu_id = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print(f"[pilot] GPU info unavailable: {exc}")
        return

    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    if not line:
        return
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3:
        idx, name, mem = parts[0], parts[1], parts[2]
        print(f"[pilot] GPU: {name} ({idx}) — {mem} MiB")
    else:
        print(f"[pilot] GPU: {line}")


# ---------------------------------------------------------------------------
# Default (real) factories. Nothing here constructs a model at import time --
# only when the factory function itself is actually called -- so `import
# scripts.partgraph_pilot` (as the CPU test does) costs nothing.
# ---------------------------------------------------------------------------

def _default_drafter_factory(pipe_cfg, device):
    return draft.Drafter(draft.load_kontext_t2i(device=device), steps=pipe_cfg.steps,
                          seed=pipe_cfg.seed, device=device)


def _default_inpainter_factory(pipe_cfg, device):
    return regen.Inpainter(regen.KontextConfig(device=device, steps=pipe_cfg.steps,
                                               seed=pipe_cfg.seed)).load()


def _default_dino_factory(pipe_cfg, device):
    return metrics.EvalEncoders.load(pipe_cfg, device).dino


def _default_store_factory(path):
    # Lazy: ragregen.mmkg_store.store imports ragregen.mmkg_store.embed_index,
    # which does `import faiss` at ITS module scope. faiss is a heavy native
    # dependency with no business loading just because this script was
    # imported (as the CPU test does) -- deferred to first actual call.
    from ragregen.mmkg_store.store import Store

    return Store.load(path)


def main(argv=None, *, drafter_factory=None, masker_factory=None,
         inpainter_factory=None, dino_factory=None, store_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--store", type=Path, required=True,
                    help="mmkg_store directory loadable by "
                         "mmkg_store.store.Store.load")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--tile", type=int, default=384,
                    help="composed-canvas cell size, passed through to "
                         "reference_for/run_case")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="trace.open_run's root; defaults to "
                         "PROJECT_ROOT/outputs")
    args = ap.parse_args(argv)
    print_gpu_info()

    drafter_factory = drafter_factory or _default_drafter_factory
    inpainter_factory = inpainter_factory or _default_inpainter_factory
    dino_factory = dino_factory or _default_dino_factory
    store_factory = store_factory or _default_store_factory

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    store = store_factory(args.store)

    cases = ds.cases[:args.limit] if args.limit else ds.cases

    # Every case must carry an explicit global_id before any GPU work starts
    # (reference-integration design §3 -- supplied on the case, no inference).
    # A missing one is a dropped case, recorded, never silently skipped and
    # never allowed to reach reference_for.
    dropped: list[dict] = []
    valid_cases = []
    for case in cases:
        if case.global_id is None:
            dropped.append({"case_id": case.id, "arm": None,
                            "reason": "missing global_id"})
        else:
            valid_cases.append(case)

    run = trace.open_run("partgraph_pilot", argv=sys.argv, args=vars(args),
                         root=args.output_root)
    print(f"[pilot] {len(valid_cases)} cases ({len(dropped)} dropped before "
          f"drafting) -> {run.path}")

    # `masker_factory`'s default closes over `pipe_cfg` (its mask_dilate_px)
    # and `case.concept`/`case.coarse` per call -- the exact factory
    # signature is `(device) -> callable(draft_image, case) -> mask | None`,
    # not `(pipe_cfg, device)`, so the default has to be defined here rather
    # than at module scope.
    if masker_factory is None:
        def _default_masker_factory(device):
            masker = mask.Masker(models.DinoDetector(device=device),
                                 sam=models.SamSegmenter(device=device))

            def _mask_fn(draft_image, case):
                result = mask.mask_draft(draft_image, case.coarse, masker,
                                         score_phrase=case.concept,
                                         dilate_px=pipe_cfg.mask_dilate_px)
                if result is None:
                    return None
                return Image.fromarray(
                    (result.mask > 0.5).astype("uint8") * 255, "L")

            return _mask_fn

        masker_factory = _default_masker_factory

    # --- Drafting stage ------------------------------------------------
    drafter = drafter_factory(pipe_cfg, args.device)
    drafts: dict[str, Image.Image] = {}
    for i, case in enumerate(valid_cases, 1):
        print(f"  [draft {i}/{len(valid_cases)}] {case.id}: {case.prompt}",
              flush=True)
        draft_img = drafter.draft(case.prompt)
        draft_img.save(run.case_dir(case.id) / "draft.png")
        drafts[case.id] = draft_img
    drafter = None
    env.reclaim_gpu()

    # --- Masking stage ---------------------------------------------------
    mask_fn = masker_factory(args.device)
    masks: dict[str, Image.Image] = {}
    masked_cases = []
    for case in valid_cases:
        mask_img = mask_fn(drafts[case.id], case)
        if mask_img is None:
            dropped.append({"case_id": case.id, "arm": None,
                            "reason": "not grounded"})
            continue
        mask_img.save(run.case_dir(case.id) / "mask.png")
        masks[case.id] = mask_img
        masked_cases.append(case)
    mask_fn = None
    env.reclaim_gpu()

    # --- Regen + score stage ---------------------------------------------
    inpainter = inpainter_factory(pipe_cfg, args.device)
    dino_encoder = dino_factory(pipe_cfg, args.device)

    per_case_rows: list[dict] = []
    rows_path = run.path / "rows.jsonl"
    with rows_path.open("w") as rows_fh:
        for case in masked_cases:
            draft_img = drafts[case.id]
            mask_img = masks[case.id]
            for arm in ARMS:
                # `sink` also stashes the composed/medoid reference and the
                # RegenResult it produced -- run_case only returns scores, and
                # the brief requires per-case traces of both the reference
                # image and the arm's output, so this is the one place either
                # is ever visible without a second (wasted) FLUX call.
                captured: dict = {}

                def sink(d, m, r, _case=case, _inp=inpainter, _cap=captured):
                    result = _inp.regen(d, m, r, _case.prompt)
                    _cap["reference"] = r
                    _cap["result"] = result
                    return result

                try:
                    row = run_case(arm=arm, store=store, global_id=case.global_id,
                                   draft=draft_img, mask=mask_img,
                                   gt_refs=case.gt_refs, encoder=dino_encoder,
                                   sink=sink, heldout=pipe_cfg.eval_heldout_refs,
                                   tile=args.tile)
                except LookupError as exc:
                    dropped.append({"case_id": case.id, "arm": arm,
                                    "reason": str(exc)})
                    continue

                row["case_id"] = case.id
                per_case_rows.append(row)
                # Written immediately, not buffered to the end: a crash
                # mid-run must not discard already-finished GPU work
                # (mirrors screen_premise.py's fh.flush()).
                rows_fh.write(json.dumps(row) + "\n")
                rows_fh.flush()

                case_dir = run.case_dir(case.id)
                captured["reference"].save(case_dir / f"reference_{arm}.png")
                captured["result"].image.save(case_dir / f"output_{arm}.png")

    inpainter.free()
    dino_encoder.free()
    inpainter = dino_encoder = None
    env.reclaim_gpu()

    # --- Decide ------------------------------------------------------------
    by_case: dict[str, dict[str, dict]] = {}
    for row in per_case_rows:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row

    deltas = []
    for case in masked_cases:
        arms_seen = by_case.get(case.id, {})
        if "partgraph" in arms_seen and "single_medoid" in arms_seen:
            deltas.append(arms_seen["partgraph"]["dino"]
                          - arms_seen["single_medoid"]["dino"])

    decision = gain_verdict(deltas)

    run.write_json("result.json", {
        "per_case": per_case_rows,
        "deltas": deltas,
        "decision": decision,
        "dropped": dropped,
    })

    run.finish("ok", {"n_cases": len(cases), "n_dropped": len(dropped),
                      "decision": decision["verdict"]})

    print(f"[pilot] decision: {decision['verdict']} "
          f"(mean={decision['mean']:.4f}, n={decision['n']})")
    if dropped:
        print(f"[pilot] {len(dropped)} case/arm drops -- see {run.path / 'result.json'}")

    return 0 if not dropped else 1


if __name__ == "__main__":
    # Not SystemExit: unloading torch+faiss+PIL together segfaults in this
    # env, which scores a finished run as a failure. See env.exit_now.
    env.exit_now(main())
