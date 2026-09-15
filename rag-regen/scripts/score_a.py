#!/usr/bin/env python
"""Stream A over every labelled draft (spec §3).

Writes raw similarities for both the concept and its coarse superordinate,
never thresholded verdicts: tau and delta are recovered offline by c1.state_at,
so calibration is arithmetic instead of thousands of GPU passes.

Loads GroundingDINO + the crop scorer and nothing else. Qwen must not be in
this process -- see scripts/score_b.py.

Usage: ./scripts/run.sh score-a --labels outputs/screen_<ts>/labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, concepts, config, env, models, trace  # noqa: E402
from ragregen.verify import record  # noqa: E402
from ragregen.verify.grounded import GroundedVerifier  # noqa: E402

#: Any tau in (0, 1) works, and delta is left at the verifier's default: only
#: the raw similarities are persisted, and `c1.state_at` recovers the state at
#: whatever (tau, delta) the report asks for.
SCORING_TAU = 0.5


def score_case(verifier, image, prompt: str, concept: str,
               coarse: str | None) -> dict:
    """Serialise one case's similarities. Deliberately no `state` field.

    `ConceptScore.state` is a verdict at this process's tau and delta, both of
    which are arbitrary here -- SCORING_TAU is 0.5 only because GroundedVerifier
    demands some tau, and delta is never set at all. Persisting it would put a
    thresholded verdict in a file this module's docstring promises holds none,
    and every real consumer recomputes via `c1.state_at` regardless.
    """
    parsed = concepts.parse(prompt, target=concept, coarse=coarse)
    scores = verifier.score(image, parsed)
    return {phrase: {k: v for k, v in record.to_dict(s, with_state=False).items()
                     if k != "phrase"}
            for phrase, s in scores.items()}


def _write(path: Path, data: dict) -> None:
    """Write-then-rename, so a crash cannot truncate completed work."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True,
                    help="labels.csv from a screen run")
    ap.add_argument("--out", type=Path, default=None,
                    help="stream_a.json (default: alongside labels.csv)")
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH,
                    help="dataset.yaml supplying each case's coarse term")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none",
                    help="prototype source. 'ceiling' uses gt_refs and is a "
                         "diagnostic upper bound, never a verifier result; "
                         "'retrieved' is the shipping configuration.")
    ap.add_argument("--force", action="store_true",
                    help="rescore cases already present in the output")
    args = ap.parse_args()

    rows = c1.load_labels(args.labels)
    out_path = args.out or args.labels.parent / "stream_a.json"
    existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
    if args.force:
        existing = {}

    todo = [r for r in rows if r.case_id not in existing]
    print(f"[score-a] {len(todo)} to score, {len(existing)} cached -> {out_path}")
    if not todo:
        return 0

    draft_paths = {r.case_id: c1.resolve_draft(r, args.labels) for r in todo}
    for r in todo:
        if not draft_paths[r.case_id].is_file():
            print(f"[error] case '{r.case_id}': draft not found: "
                  f"{draft_paths[r.case_id]}")
            return 2

    ds = config.load_dataset(args.dataset)
    coarse_by_id = {c.id: c.coarse for c in ds.cases}
    absent = sorted({r.case_id for r in todo} - set(coarse_by_id))
    if absent:
        print(f"[error] no coarse term in {args.dataset} for: {', '.join(absent)}. "
              f"Every labelled case needs one -- see docs/RUNBOOK.md.")
        return 2

    pipe_cfg = config.load_pipeline(args.pipeline)
    run = trace.open_run("score_a", argv=sys.argv, args=vars(args))

    from PIL import Image
    detector = models.DinoDetector(device=args.device)
    scorer = models.build_crop_scorer(pipe_cfg.crop_scorer, device=args.device)
    bank = embedder = None
    if args.proto_arm != "none":
        from ragregen import encoders
        from ragregen.verify import prototype

        # The default scorer and retriever are the same SigLIP checkpoint.
        # Reuse its resident image/text towers instead of loading ~3.5 GB of
        # duplicate weights beside GroundingDINO on the shared card.
        if (pipe_cfg.crop_scorer == pipe_cfg.retriever
                and hasattr(scorer, "encode_pil")
                and hasattr(scorer, "encode_text")):
            embedder = scorer
        else:
            embedder = encoders.build_encoder(pipe_cfg.retriever,
                                               device=args.device)
        if args.proto_arm == "ceiling":
            bank = prototype.bank_from_gt_refs(ds, embedder)
        else:
            from ragregen import retrieve
            db = config.load_retrieval_db()
            retriever = retrieve.Retriever.from_index(db.index_path, embedder)
            bank = prototype.bank_from_retrieval(ds, retriever, embedder)
        print(f"[score-a] prototype arm '{args.proto_arm}': "
              f"{len(bank.phrases)} prototypes", flush=True)

    verifier = GroundedVerifier(detector, scorer, tau=SCORING_TAU,
                                prototypes=bank, embedder=embedder)

    for i, r in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {r.case_id}", flush=True)
        image = Image.open(draft_paths[r.case_id]).convert("RGB")
        existing[r.case_id] = score_case(verifier, image, r.prompt, r.concept,
                                         coarse_by_id[r.case_id])
        _write(out_path, existing)

    run.finish("ok", {"n_scored": len(todo), "out": str(out_path)})
    env.reclaim_gpu()
    print(f"[score-a] wrote {len(existing)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
