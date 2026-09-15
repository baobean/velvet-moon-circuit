#!/usr/bin/env python
"""Re-verify a finished run's images into a NEW run directory.

Regenerates nothing. The source run is evidence and is never written to --
operator standing rule, 2026-08-10: when a run's provenance is compromised,
re-run it cleanly rather than hand-edit its metadata.

    ./scripts/run.sh reverify --run outputs/doc5_20260810_183939 \
        --dataset configs/dataset_common.yaml [--proto-arm retrieved]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, schedule, trace                    # noqa: E402
import run_pipeline                                             # noqa: E402


def plan_targets(src: Path) -> dict[str, list[int]]:
    """{case_id: [0, 1, 2...]} -- attempt 0 is the draft. Read-only."""
    out: dict[str, list[int]] = {}
    for case_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        attempts = sorted(int(p.stem.split("_")[1])
                          for p in case_dir.glob("attempt_*.png"))
        out[case_dir.name] = [0] + attempts
    return out


def link_pixels(src: Path, run_dir, case_ids: list[str]) -> None:
    """Symlink, never copy. These pixels are the one thing this run did not
    produce, and claiming authorship of them is the error being corrected."""
    for cid in case_ids:
        d = run_dir.case_dir(cid)
        for pattern in ("mask.png", "attempt_*.png", "cutout_*.png"):
            for f in sorted((src / cid).glob(pattern)):
                target = d / f.name
                if not target.exists():
                    target.symlink_to(f.resolve())


def reselect(scored: dict) -> str:
    """Re-run schedule.select_winner over a corrected scores.json.

    A genuinely passing draft wins outright here rather than being eligible
    for displacement by a later attempt that also now passes -- see
    select_winner's docstring for why. report._best_label and
    run_pipeline._finalise_one call the same function, so the three cannot
    disagree about which candidate a case's metrics describe.
    """
    draft_ok, draft_score = scored.get("draft", [False, None])
    attempts = [(label, bool(v[0]), v[1])
                for label, v in sorted(scored.items()) if label != "draft"]
    return schedule.select_winner(bool(draft_ok), draft_score, attempts)


def _open(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="a finished pipeline run directory")
    ap.add_argument("--dataset", type=Path,
                    default=Path("configs/dataset_common.yaml"))
    ap.add_argument("--pipeline", type=Path,
                    default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--screen", type=Path, default=Path("outputs/screen_latest"))
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="reverify")
    ap.add_argument("--resume", type=Path, default=None,
                    help="an existing outputs/reverify_<TS>/ to continue")
    return ap.parse_args(argv)


def _skip_out_of_round(queue, targets: dict[str, list[int]],
                       attempt: int) -> None:
    """Keep a case with no attempt_<attempt>.png out of THIS round only.

    plan_targets already knows which rounds each case has -- a case with one
    attempt, in a run where another case has three, must not be asked for a
    round it has no image for. Without this, `image_of` would open a
    non-existent attempt_N.png and run_stage would catch the
    FileNotFoundError and mark the case `failed`, stranding a stray
    traceback in queue.json even though the case's scores.json (from
    earlier rounds) is fine.

    Unlike run_pipeline._retire_exhausted, this cannot permanently retire
    the case: `targets[cid]` can have a gap (attempt_1.png missing,
    attempt_2.png present), so a case skipped at round N may still have
    work at round N+1. Pre-marking the round's stage keys "skipped" is the
    same mechanism Queue.pending already understands to skip a case -- no
    new code path -- and if it leaves NO case pending for a round,
    stage_with_model's own guard means DINO/SigLIP/Qwen never load for it.
    """
    for cid, rounds in targets.items():
        if attempt in rounds:
            continue
        state = queue.cases.get(cid)
        if state is None or state.status != "pending":
            continue
        for stage in ("grounded", "semantic"):
            state.stages[schedule.stage_key(stage, attempt)] = "skipped"
    queue.save()


def _run(queue, run_dir, by_id: dict, targets: dict[str, list[int]],
         pipe_cfg, ds, args) -> None:
    src = args.run.resolve()
    bank, embedder = run_pipeline._build_bank(args.proto_arm, ds, pipe_cfg,
                                              args.device)

    max_attempt = max(max(v) for v in targets.values())
    for attempt in range(0, max_attempt + 1):
        _skip_out_of_round(queue, targets, attempt)

        def image_of(cid, _a=attempt):
            if _a == 0:
                #: The draft lives in the SCREEN run, not `src` -- confirmed
                #: against outputs/doc5_20260810_183939, whose case dirs hold
                #: no draft.png; only outputs/screen_latest/<case>/draft.png
                #: does. It is read directly rather than symlinked into
                #: run_dir: link_pixels only carries pixels `src` itself
                #: produced, and the draft was never one of them.
                return _open(args.screen / cid / "draft.png")
            return _open(src / cid / f"attempt_{_a}.png")

        run_pipeline._verify_round(queue, run_dir, by_id, pipe_cfg,
                                   attempt=attempt, device=args.device,
                                   image_of=image_of, bank=bank,
                                   embedder=embedder)

    for cid in by_id:
        path = run_dir.case_dir(cid) / "scores.json"
        if not path.is_file():
            continue
        scored = json.loads(path.read_text())
        best = reselect(scored)
        draft_ok = bool(scored.get("draft", [False, None])[0])
        queue.cases[cid].best = best
        queue.cases[cid].status = run_pipeline._status_for(best, draft_ok)
    queue.save()


def main(argv=None) -> int:
    args = _parse_args(argv)

    src = args.run.resolve()
    targets = plan_targets(src)
    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    by_id = {c.id: c for c in ds.cases if c.id in targets}
    if not by_id:
        raise ValueError(
            f"no case in {args.dataset} matches any directory under {src}. "
            f"Wrong --dataset, or {src} is not a finished pipeline run?")
    #: Only the cases actually being reverified -- a stray leftover
    #: directory under `src` that no longer matches the dataset must not
    #: inflate the round count or the retry budget below.
    targets = {cid: targets[cid] for cid in by_id}

    if args.resume:
        #: Construct OVER the existing directory rather than minting a fresh
        #: one -- trace.open_run always timestamps a new one, so without
        #: this a killed reverify could never resume, only start over.
        #: schedule.Queue.open below then finds the existing queue.json and
        #: does the actual resuming; this just has to hand it the same path.
        run_dir = trace.RunDir(path=Path(args.resume), argv=sys.argv,
                               args=vars(args))
    else:
        run_dir = trace.open_run(args.tag, argv=sys.argv, args=vars(args))
    link_pixels(src, run_dir, list(by_id))

    #: `- 1`: `v` is [0, 1, ..., N] -- the draft plus every attempt -- so its
    #: length over-counts the retry budget (the number of REPAIR rounds) by
    #: exactly the draft. Cosmetic while nothing here read Queue.retry_budget
    #: back; load-bearing now that --resume carries a persisted queue.json's
    #: retry_budget forward into needs_round() on the resumed run.
    queue = schedule.Queue.open(run_dir.path / "queue.json", list(by_id),
                                retry_budget=max(len(v) for v in targets.values()) - 1,
                                screen_run=str(args.screen))
    queue.save()

    try:
        _run(queue, run_dir, by_id, targets, pipe_cfg, ds, args)
    except schedule.StageAborted as exc:
        print(f"[reverify] {exc}")
        #: `run_dir.finish` rewrites run.json wholesale (trace.py:64-72),
        #: which would silently erase a `source_run` written before it -- so
        #: it travels inside `finish`'s own `results` dict instead, same as
        #: the "ok" path below.
        run_dir.finish("aborted", {"source_run": str(src),
                                   "unresolved": queue.unresolved()})
        print(f"[reverify] resume with: ./scripts/run.sh reverify "
              f"--resume {run_dir.path}")
        return 2

    run_dir.finish("ok", {"cases": len(by_id), "source_run": str(src),
                          "proto_arm": args.proto_arm})
    print(f"[reverify] {len(by_id)} cases <- {src}  -> {run_dir.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
