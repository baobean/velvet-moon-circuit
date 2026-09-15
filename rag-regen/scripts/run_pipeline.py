#!/usr/bin/env python
"""The stage-batched scheduler driver.

Owns model residency and stage order. Every scheduling RULE lives in
ragregen/schedule.py, which imports no torch -- this file is the only place
that knows FLUX exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import numpy as np  # noqa: E402

from ragregen import (composite, config, env, mask, metrics,  # noqa: E402
                      models, regen, retrieve, schedule, trace)
from ragregen.verify import fusion, grounded, record, semantic  # noqa: E402

#: Peak VRAM per stage, from spec §4's measured figures: Qwen-7B ~16 GB,
#: FLUX-nf4 ~12 GB, GroundingDINO+SAM ~6 GB, SigLIP ~4 GB. Headroom included,
#: because a stage that fits exactly is a stage that OOMs on fragmentation.
#: Keyed on schedule.STAGES exactly -- "semantic" is the Qwen-7B verifier
#: (inherits the 17.0 a dead "verify" key used to hold) and "grounded" is the
#: GroundingDINO-family detector, sharing "mask"'s ~6 GB. Unlike the other
#: figures, grounded's 6.0 is inferred from the detector family rather than
#: separately measured -- not a spec §4 number.
STAGE_VRAM_GB = {
    "semantic": 14.0,  # TEMP: lowered from 17.0 to fit the 16 GB A4000 (GPU 0); AWQ weights ~6.5 GB. Revert after reverify.
    "regen": 13.0,
    "grounded": 6.0,
    "mask": 6.0,
    "retrieve": 4.0,
}

#: --proto-arm's embedder (IMPORTANT 4, final review). _build_bank runs once
#: at the top of _run, OUTSIDE stage_with_model's load/free-in-`finally`
#: discipline, and GroundedVerifier.embedder is genuinely consulted on EVERY
#: grounded-stage call, every round (verify/grounded.py:115-120: it scores
#: each candidate crop against the prototype bank) -- so it cannot simply be
#: freed once the bank is built, and it stays GPU-resident through every
#: stage for the rest of the run once an arm is requested. None of the
#: STAGE_VRAM_GB figures above budget for it, so the preflight under-counts
#: what the card actually needs to hold whenever proto_arm != "none". Value
#: matches STAGE_VRAM_GB["retrieve"] -- the "retrieved" arm's bank literally
#: reuses that same encoder (_build_bank), and "ceiling" builds an equivalent
#: one via encoders.build_encoder(pipe_cfg.retriever, ...).
PROTO_EMBEDDER_VRAM_GB = 4.0


def resolve_screen_run(path: Path) -> Path:
    """The drafts to repair. Never generated here.

    `screen` is the gate (RUNBOOK §4) and its drafts are the ones that were
    hand-labelled. Drafting fresh ones at a different seed would mean the
    labels no longer describe the images being repaired -- so a missing screen
    run is an error, not a prompt to be helpful.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"no screen run at {path}. Run `./scripts/run.sh screen` first -- "
            f"it is the gate everything downstream reads.")
    if not list(path.glob("*/draft.png")):
        raise FileNotFoundError(
            f"{path} has no drafts (expected <case>/draft.png).")
    return path


def rounds_for(retry_budget: int, n_edit_refs: int) -> int:
    """How many repair rounds this case can actually afford.

    Attempt N edits with refs[N - 1] and _regen_one raises when the cutout is
    absent, so a budget larger than the edit pool does not produce more
    attempts -- it produces spurious `failed` cases.
    """
    return min(int(retry_budget), int(n_edit_refs))


def stage_with_model(queue, stage: str, load, work, *,
                     attempt: int | None = None,
                     need_gb: float | None = None) -> None:
    """Load a model only if some case needs it, run the stage, always free it.

    The guard is the difference between a resumed run that takes seconds and
    one that spends 5m21s loading FLUX to discover every case is done.

    The VRAM preflight yields the card deliberately rather than waiting to be
    evicted mid-allocation: an OOM partway through loading 16 GB of Qwen
    leaves the allocator in a worse state than a clean checkpoint, and
    supervise.sh will re-acquire and resume (doc 5 §9).
    """
    if not queue.pending(stage, attempt=attempt):
        return

    want = STAGE_VRAM_GB.get(stage) if need_gb is None else need_gb
    if want is not None:
        free = env.free_vram_gb()
        #: None means unknown -- a CPU-only machine, or nvidia-smi unreadable.
        #: Unknown is permission to proceed; blocking would abort every CPU run.
        if free is not None and free < want:
            queue.save()
            raise schedule.StageAborted(
                f"{stage} needs {want:.0f} GB VRAM, {free:.1f} GB free. "
                f"The card is held by another job -- `nvidia-smi` shows who. "
                f"Re-run the identical command to resume.")

    model = load()
    stage_failed = False
    try:
        schedule.run_stage(queue, stage, lambda cid: work(model, cid),
                           attempt=attempt)
    except BaseException:
        stage_failed = True
        raise
    finally:
        free_fn = getattr(model, "free", None)
        if callable(free_fn):
            try:
                free_fn()
            except BaseException as cleanup_exc:
                if not stage_failed:
                    raise
                # Preserve the stage exception. In particular, a device-side
                # CUDA failure makes empty_cache fail too; masking the first
                # error used to prevent main() from writing resumable state.
                print(f"[pipeline] cleanup after failed {stage} also failed: "
                      f"{cleanup_exc}", file=sys.stderr, flush=True)


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pipeline")
    ap.add_argument("--resume", type=Path, default=None,
                    help="an existing outputs/pipeline_<TS>/ to continue")
    ap.add_argument("--arm", choices=("oracle", "full"), default="oracle")
    ap.add_argument("--mechanism", choices=regen.MECHANISMS, default="inpaint")
    ap.add_argument("--verifier", choices=("fused", "none"), default="fused",
                    help="none is a one-shot open-loop ablation: edit every "
                         "case once, perform no verifier-based routing or "
                         "selection, then evaluate external metrics")
    ap.add_argument("--open-loop-attempts", type=int, default=1,
                    help="with --verifier none, cache this many reference "
                         "attempts while retaining attempt 1 as the declared "
                         "no-verifier output (default: 1)")
    ap.add_argument("--ref-prep", choices=("none", "crop", "crop_matte"),
                    default=None,
                    help="override pipeline.yaml reference preparation; "
                         "none reproduces the old full-photo conditioning")
    ap.add_argument("--screen-run", type=Path,
                    default=Path("outputs/screen_latest"))
    ap.add_argument("--limit", type=int, default=0, help="0 = every case")
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path,
                    default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none",
                    help="build a prototype bank and record image-side scores "
                         "in streams.json. It affects live verdicts only when "
                         "pipeline.yaml sets prototype_delta.")
    ap.add_argument("--mask-prototypes", action="store_true",
                    help="ALSO use that bank to pick the mask box. This "
                         "changes mask geometry, so it changes results. "
                         "Requires --proto-arm != none.")
    ns = ap.parse_args(argv)
    if ns.mask_prototypes and ns.proto_arm == "none":
        #: Silent no-op here would be worse than a crash: in a repo where run
        #: provenance is load-bearing evidence, an operator who believes they
        #: ran a prototype-masked arm but did not would mis-attribute every
        #: result that followed. Fail before any model loads, not with a
        #: warning that scrolls past in a multi-hour run.
        ap.error("--mask-prototypes requires --proto-arm != none")
    if ns.open_loop_attempts < 1:
        ap.error("--open-loop-attempts must be a positive integer")
    if ns.open_loop_attempts != 1 and ns.verifier != "none":
        ap.error("--open-loop-attempts > 1 requires --verifier none")
    return ns


def _resume_command(run_dir_path: Path) -> str:
    """The full original flags plus `--resume`, not just `--resume PATH`.

    `args = _parse_args()` re-parses fresh on every invocation -- nothing
    from a prior run is restored automatically -- so a bare `--resume PATH`
    silently reverts every other flag (`--dataset`, `--verifier`, ...) to its
    argparse default. That is exactly how a `--verifier none` open-loop run
    resumed as `--verifier fused`, ran the round-0 grounded/semantic verify
    it was never meant to, and marked every case's queue status "failed" on
    a KeyError from `by_id` no longer containing the run's real dataset.
    Printing the complete original argv here is the fix -- copy-paste it
    unmodified rather than hand-appending `--resume` to a bare invocation.
    """
    original: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--resume":
            skip_next = True  # drop this run's own --resume value too
            continue
        if arg.startswith("--resume="):
            continue
        original.append(arg)
    parts = [shlex.quote(a) for a in original]
    parts.append(f"--resume {shlex.quote(str(run_dir_path))}")
    return " ".join(parts)


def main() -> int:
    args = _parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    if args.verifier == "none" and pipe_cfg.retry_budget < 1:
        raise ValueError("the no-verifier ablation requires retry_budget >= 1")
    if pipe_cfg.prototype_delta is not None and args.proto_arm == "none":
        raise ValueError(
            "pipeline.yaml enables prototype_delta but --proto-arm is none; "
            "a prototype verdict requires image prototypes")
    args.ref_prep = args.ref_prep or pipe_cfg.ref_prep
    screen_run = resolve_screen_run(args.screen_run)

    cases = list(ds.cases)[:args.limit] if args.limit else list(ds.cases)
    by_id = {c.id: c for c in cases}
    #: Trimmed to the cases actually running -- a --limit run should not pay
    #: to build prototypes for concepts it never scores.
    ds = dataclasses.replace(ds, cases=cases)

    if args.resume:
        run_dir = trace.RunDir(path=Path(args.resume), argv=sys.argv,
                               args=vars(args))
    else:
        run_dir = trace.open_run(args.tag, argv=sys.argv, args=vars(args))

    queue = schedule.Queue.open(
        run_dir.path / "queue.json", [c.id for c in cases],
        retry_budget=pipe_cfg.retry_budget, screen_run=str(screen_run))

    print(f"[pipeline] {len(queue.cases)} cases -> {run_dir.path}  "
          f"arm={args.arm} mechanism={args.mechanism} "
          f"verifier={args.verifier}", flush=True)

    try:
        _run(queue, run_dir, by_id, pipe_cfg, screen_run, args, ds)
    except schedule.StageAborted as exc:
        print(f"[pipeline] {exc}")
        run_dir.finish("aborted", {"unresolved": queue.unresolved()})
        print(f"[pipeline] resume with: ./scripts/run.sh pipeline "
              f"{_resume_command(run_dir.path)}")
        return 2

    run_dir.finish("ok", _finish_results(queue))
    print(f"[pipeline] done -> {run_dir.path}")
    return 0


def _finish_results(queue) -> dict:
    """Case IDs by final status, for run.json's `results`.

    A human-readable summary only -- report.py recomputes its buckets from
    scores.json, never from this. Every terminal status CaseState can hold
    gets its own key; a status left out here would go uncounted rather than
    merely miscounted, which the `healthy`/`unrepaired` split makes an easy
    mistake to reintroduce.
    """
    return {status: [cid for cid, s in queue.cases.items()
                     if s.status == status]
           for status in ("healthy", "repaired", "unrepaired", "failed")}


def _run(queue, run_dir, by_id, pipe_cfg, screen_run, args, ds) -> None:
    from PIL import Image

    def draft_of(cid):
        return Image.open(screen_run / cid / "draft.png").convert("RGB")

    #: None, None unless --proto-arm asked for one -- see _build_bank.
    bank, embedder = _build_bank(args.proto_arm, ds, pipe_cfg, args.device)
    # Masker.select_box consumes crops, while PrototypeBank consumes vectors.
    # Passing the bank directly (the old wiring) only failed once a reference
    # produced multiple detector boxes, because that is the first time
    # select_box calls ``score(crop, phrase)``.  Adapt the bank at the model
    # boundary so live masking uses the same embedder that built it.
    mask_classifier = None
    if args.mask_prototypes:
        from ragregen.verify.prototype import CropClassifier

        mask_classifier = CropClassifier(bank, embedder)
    #: See PROTO_EMBEDDER_VRAM_GB -- the embedder outlives _build_bank and
    #: stays resident through every stage below, so each one's preflight
    #: budget has to include it explicitly.
    proto_extra = PROTO_EMBEDDER_VRAM_GB if embedder is not None else 0.0

    verifier_mode = getattr(args, "verifier", "fused")

    # --- ROUND 0: verify the draft -------------------------------------
    # The open-loop ablation deliberately routes every draft to repair.
    if verifier_mode == "fused":
        _verify_round(queue, run_dir, by_id, pipe_cfg, attempt=0,
                      device=args.device, image_of=draft_of,
                      bank=bank, embedder=embedder)

    # --- ONCE: retrieve, then mask (design §4) -------------------------
    if args.arm == "oracle":
        for cid in list(queue.pending("retrieve")):
            edit, held = metrics.split_refs(by_id[cid].gt_refs,
                                            pipe_cfg.eval_heldout_refs)
            d = run_dir.case_dir(cid)
            (d / "refs.json").write_text(json.dumps([str(p) for p in edit]))
            #: The scoring target, written once so the report never has to
            #: re-derive the split and risk disagreeing with the run.
            (d / "heldout_refs.json").write_text(
                json.dumps([str(p) for p in held]))
            queue.mark(cid, "retrieve", "done")
    else:
        stage_with_model(
            queue, "retrieve",
            lambda: _load_retriever(pipe_cfg, args.device),
            lambda r, cid: _retrieve_one(r, run_dir, by_id[cid],
                                         pipe_cfg.retry_budget),
            need_gb=STAGE_VRAM_GB["retrieve"] + proto_extra)

    stage_with_model(queue, "mask",
                     lambda: _load_masker(args.device),
                     lambda m, cid: _mask_one(m, run_dir, by_id[cid],
                                              draft_of(cid), pipe_cfg,
                                              prototypes=mask_classifier,
                                              ref_prep=getattr(
                                                  args, "ref_prep",
                                                  getattr(pipe_cfg, "ref_prep",
                                                          "crop"))),
                     need_gb=STAGE_VRAM_GB["mask"] + proto_extra)

    # --- ROUNDS 1..N ---------------------------------------------------
    available = max((len(_refs_of(run_dir, c)) for c in queue.cases),
                    default=0)
    # No verifier still declares attempt 1 as its output. Extra open-loop
    # rounds only cache a candidate bank for an external, frozen selector;
    # they never select on the held-out evaluation metric.
    open_loop_attempts = int(getattr(args, "open_loop_attempts", 1))
    n_rounds = (min(open_loop_attempts, pipe_cfg.retry_budget, available)
                if verifier_mode == "none" else
                rounds_for(pipe_cfg.retry_budget, available))
    if verifier_mode == "none" and n_rounds == 0:
        _fail_open_loop_without_refs(queue)

    for attempt in range(1, n_rounds + 1):
        if verifier_mode == "fused":
            _retire_exhausted(queue, run_dir, attempt)
        elif open_loop_attempts > 1:
            _retire_open_loop_exhausted(queue, run_dir, attempt)
        if not queue.needs_round(attempt):
            break
        stage_with_model(
            queue, "regen",
            lambda: _load_regen(args, pipe_cfg),
            lambda eng, cid: _regen_one(eng, run_dir, by_id[cid],
                                        draft_of(cid), attempt, args),
            attempt=attempt, need_gb=STAGE_VRAM_GB["regen"] + proto_extra)
        if verifier_mode == "none":
            if open_loop_attempts == 1:
                _resolve_open_loop(queue, run_dir, attempt)
        else:
            _verify_round(
                queue, run_dir, by_id, pipe_cfg, attempt=attempt,
                device=args.device,
                image_of=lambda cid: Image.open(
                    run_dir.case_dir(cid) / f"attempt_{attempt}.png"
                ).convert("RGB"),
                bank=bank, embedder=embedder)

    if verifier_mode == "fused":
        _finalise(queue, run_dir)
    elif open_loop_attempts > 1:
        _resolve_candidate_bank(queue, run_dir)
    else:
        queue.save()


def _load_masker(device: str):
    return mask.Masker(models.DinoDetector(device=device),
                       sam=models.SamSegmenter(device=device))


def _mask_one(masker, run_dir, case, draft, pipe_cfg, *,
              prototypes=None, ref_prep: str | None = None) -> None:
    """The draft's mask, and a cutout per reference. Once per case, not per round.

    Every reference a run could need is already known -- retrieval depth is the
    retry budget -- so masking them all in one DINO+SAM residency turns N model
    loads into one (design §4).

    `prototypes` is None unless --mask-prototypes was passed -- see
    _build_bank. Forwarded, never built here: this function has no opinion
    about which arm produced the bank.
    """
    from PIL import Image

    d = run_dir.case_dir(case.id)
    drafted = mask.mask_draft(draft, case.coarse, masker,
                              prototypes=prototypes,
                              score_phrase=case.concept,
                              dilate_px=pipe_cfg.mask_dilate_px)
    if drafted is None:
        raise ValueError(f"'{case.coarse}' did not ground in the draft")

    #: How the box was chosen, not just which box -- doc 1 §6: without this a
    #: masking failure and a generation failure are indistinguishable.
    (d / "mask.json").write_text(json.dumps({
        "selected_by": drafted.selected_by,
        "n_candidates": drafted.n_candidates,
        "box": list(drafted.box),
        "score": drafted.score,
    }, indent=2))

    Image.fromarray((drafted.mask > 0.5).astype("uint8") * 255, "L").save(
        d / "mask.png")

    prep_mode = ref_prep or getattr(pipe_cfg, "ref_prep", "crop")
    refs = [Path(p) for p in json.loads((d / "refs.json").read_text())]
    prep_records = {}
    rejected = []
    accepted_sources = []
    for ref_path in refs:
        reference = Image.open(ref_path).convert("RGB")
        ref_mask = mask.mask_reference(reference, case.concept, masker,
                                       prototypes=prototypes)
        # ``crop`` must mean crop. Falling back to a full, ungrounded photo is
        # the exact silent path that copied a woman/flower scene and crochet
        # axolotls into the draft. Keep the old behaviour available only via
        # the explicit ref_prep=none ablation.
        if ref_mask is None and prep_mode != "none":
            rejected.append({
                "source": str(ref_path),
                "reason": f"{case.concept!r} not grounded in reference",
            })
            continue

        i = len(accepted_sources) + 1
        cut = composite.cutout_from(reference, ref_mask)
        if cut is not None:
            cut.save(d / f"cutout_{i}.png")
        prepared, prep_info = regen.prepare_reference(
            reference, case.concept, mode=prep_mode,
            masker=lambda _image, _phrase, found=ref_mask: found)
        prepared.save(d / f"reference_{i}.png")
        prep_records[str(i)] = {
            "source": str(ref_path), "cutout_available": cut is not None,
            **prep_info,
        }
        accepted_sources.append(str(ref_path))
    prep_records["_rejected"] = rejected
    (d / "refs.json").write_text(json.dumps(accepted_sources))
    (d / "references.json").write_text(json.dumps(prep_records, indent=2))
    if refs and not accepted_sources:
        raise ValueError(
            f"no reference grounded the fine concept {case.concept!r}")


def _load_retriever(pipe_cfg, device: str):
    from ragregen import encoders, retrieve

    db = config.load_retrieval_db()
    return retrieve.Retriever.from_index(
        db.index_path, encoders.build_encoder(pipe_cfg.retriever, device=device))


def _build_bank(arm: str, ds, pipe_cfg, device: str):
    """None unless an arm was asked for. Two arms, two sources, never mixed.

    Nothing is imported or constructed for arm == "none" -- the default -- so
    a resumed doc-5 run with no --proto-arm never builds an embedder at all.

    "retrieved" reuses _load_retriever rather than repeating
    config.load_retrieval_db()/Retriever.from_index here: it is the one place
    that already knows how to open the retrieval index, and Retriever keeps
    the encoder it was built with on `.encoder`, so the retrieval half and the
    embedder handed to the bank are guaranteed to be the same object.
    """
    if arm == "none":
        return None, None
    from ragregen import encoders
    from ragregen.verify import prototype

    if arm == "ceiling":
        embedder = encoders.build_encoder(pipe_cfg.retriever, device=device)
        return prototype.bank_from_gt_refs(ds, embedder), embedder

    retriever = _load_retriever(pipe_cfg, device)
    return (prototype.bank_from_retrieval(ds, retriever, retriever.encoder),
           retriever.encoder)


def _retrieve_one(retriever, run_dir, case, k: int) -> None:
    query = retrieve.reference_query(case.concept, case.coarse)
    hits = retriever.search(query, k)
    d = run_dir.case_dir(case.id)
    (d / "refs.json").write_text(
        json.dumps([str(h.path) for h in hits]))
    (d / "retrieval.json").write_text(json.dumps({
        "query": query,
        "hits": [{"path": str(h.path), "score": h.score, "rank": h.rank}
                 for h in hits],
    }, indent=2))

    t = _trace(run_dir, case.id)
    t.hits([h.path for h in hits])
    t.save(run_dir.case_dir(case.id) / "trace.json")


class _Stitcher:
    """Gives the diffusion-free mechanism the same load/free shape as Inpainter."""

    def free(self) -> None:
        env.reclaim_gpu()


def _load_regen(args, pipe_cfg):
    if args.mechanism == "stitch":
        return _Stitcher()                      # no weights, no card
    cfg = regen.KontextConfig(device=args.device, steps=pipe_cfg.steps,
                              seed=pipe_cfg.seed)
    return regen.Inpainter(cfg).load()


def _regen_one(engine, run_dir, case, draft, attempt: int, args) -> None:
    """Attempt `attempt`, always from the ORIGINAL draft.

    `draft` is re-opened by the caller every round. Chaining edits -- feeding
    attempt N-1 into attempt N -- would degrade the whole image, which is the
    guarantee RUNBOOK §1 makes to anyone reading the results.
    """
    from PIL import Image

    d = run_dir.case_dir(case.id)
    mask_pil = Image.open(d / "mask.png").convert("L")
    if args.mechanism == "stitch":
        cutout_path = d / f"cutout_{attempt}.png"
        if not cutout_path.is_file():
            raise ValueError(f"no reference cutout for attempt {attempt}")
        result = regen.stitch(draft, mask_pil,
                              Image.open(cutout_path).convert("RGBA"),
                              feather_px=regen.KontextConfig().feather_px)
    else:
        prepared_path = d / f"reference_{attempt}.png"
        if not prepared_path.is_file():
            raise ValueError(
                f"no prepared reference for attempt {attempt}; rerun the "
                f"mask stage rather than conditioning on the full photo")
        reference = Image.open(prepared_path).convert("RGB")
        prompt = regen.edit_prompt(case.prompt, case.coarse, case.concept)
        result = engine.regen(draft, mask_pil, reference, prompt)

    result.image.save(d / f"attempt_{attempt}.png")
    if result.raw is not None:
        result.raw.save(d / f"raw_attempt_{attempt}.png")
    Image.fromarray(np.rint(result.alpha * 255).astype("uint8"), "L").save(
        d / f"alpha_{attempt}.png")
    (d / f"regen_{attempt}.json").write_text(json.dumps({
        "mechanism": result.mechanism,
        "reference": (str(d / f"reference_{attempt}.png")
                      if args.mechanism == "inpaint"
                      else str(d / f"cutout_{attempt}.png")),
        "meta": result.meta,
    }, indent=2, default=str))


def _stash(run_dir, case_id: str, attempt: int, stream: str, payload) -> None:
    """Append one stream's result for one attempt to the case's streams file."""
    path = run_dir.case_dir(case_id) / "streams.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault(str(attempt), {})[stream] = payload
    path.write_text(json.dumps(data, indent=2, default=str))


def _stash_grounded(run_dir, case_id: str, attempt: int, scores: dict) -> None:
    """Persist Stream A in full -- every similarity, every candidate.

    The old shape was {phrase: state}, which cannot be re-graded at any other
    tau. See docs/superpowers/specs/2026-08-12-verifier-persistence-design.md.
    """
    _stash(run_dir, case_id, attempt, "grounded",
           {phrase: record.to_dict(s, with_state=True)
            for phrase, s in scores.items()})


def _trace(run_dir, case_id: str) -> trace.CaseTrace:
    """Load-or-start this case's trace. Appended to, never overwritten."""
    path = run_dir.case_dir(case_id) / "trace.json"
    t = trace.CaseTrace(case_id=case_id)
    if path.is_file():
        old = json.loads(path.read_text())
        t.vlm_calls = old.get("vlm", [])
        t.grounded_scores = old.get("grounded", [])
        t.retrieval_hits = old.get("retrieval", [])
    return t


def _trace_vlm(run_dir, case_id: str, attempt: int, raw: str) -> None:
    """The raw reply, with garbage flagged.

    Parent spec §11: the cause is almost never the algorithm, it is one
    silently-garbage intermediate that nothing logged.
    """
    t = _trace(run_dir, case_id)
    t.vlm(f"semantic@{attempt}", raw)
    t.save(run_dir.case_dir(case_id) / "trace.json")


def _stash_semantic(run_dir, case_id: str, attempt: int, verdict) -> None:
    _stash(run_dir, case_id, attempt, "semantic",
           {"ok": verdict.ok, "degenerate": verdict.degenerate,
            "raw": verdict.raw,
            "issues": [{"concept": i.concept, "problem": i.problem}
                       for i in verdict.issues]})


def _status_for(label: str, ok: bool) -> str:
    """The queue status for a chosen candidate.

    `draft` is the one label two opposite outcomes can produce: it PASSED and
    never needed repair (`healthy`), or every repair attempt also failed and
    it was kept as the least-bad fallback (`unrepaired`). Any other label
    reaching here means schedule.select_best preferred that candidate over
    the draft -- `_resolve` calls this only for a candidate whose verdict
    just passed, but `_finalise_one` feeds it whatever select_best returned,
    and select_best's fallback (no attempt passed) picks the
    highest-SCORING attempt even if it also failed. So a non-draft label is
    usually a pass, but is not guaranteed to be one, and this function
    reports it "repaired" either way. That gap is pre-existing (the code
    this replaced had it too) and out of this task's scope to close --
    tests/test_schedule.py::test_a_higher_scoring_failing_attempt_still_wins_when_it_has_evidence
    pins the select_best behaviour that produces it.

    One function, called from both `_resolve` (stops a case the moment it
    passes) and `_finalise_one` (recomputes every case at the end of the
    run), because two call sites independently expressing this rule is
    exactly how `unrepaired` ended up meaning both things in the first place
    (findings/2026-07-29-orchestration-result.md §2). scripts/report.py's
    `bucket`/`_best_label` docstrings require the report and queue.json to
    never be able to disagree; sharing this rule is what keeps that true.
    """
    if label != "draft":
        return "repaired"
    return "healthy" if ok else "unrepaired"


def _resolve(queue, case_id: str, label: str) -> None:
    """A passing verdict ends this case's run immediately.

    Without this the round loop never stops early: needs_round() asks for
    cases still `pending`, and nothing else moves a case off `pending` until
    _finalise, which runs AFTER the last round. Measured cost of getting this
    wrong: three full rounds, 15:45 to 18:01, for a case that passed on
    attempt 1. Design R1 -- the loop is only affordable if early-stop works.
    """
    queue.cases[case_id].best = label
    #: Only ever called with a label whose verdict just passed -- see
    #: _fuse_pending and _status_for.
    queue.cases[case_id].status = _status_for(label, ok=True)
    queue.save()


def _record_score(run_dir, case_id: str, label: str, both: dict) -> bool:
    """Fuse Stream A and Stream B into {label: [ok, score]} in scores.json.
    `score` is null when Stream A saw no evidence for this candidate.

    Returns the verdict, so the caller can stop the case the moment it passes.
    """
    scores = {phrase: record.from_dict(payload)
              for phrase, payload in both["grounded"].items()}
    verdict = fusion.fuse(scores, semantic.SemanticVerdict(
        ok=both["semantic"]["ok"], raw="",
        degenerate=both["semantic"]["degenerate"]))

    path = run_dir.case_dir(case_id) / "scores.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[label] = [bool(verdict.ok),
                   None if verdict.score is None else float(verdict.score)]
    path.write_text(json.dumps(data, indent=2, default=str))
    return bool(verdict.ok)


def _verify_round(queue, run_dir, by_id, pipe_cfg, *, attempt: int,
                  device: str, image_of, bank=None, embedder=None) -> None:
    """Stream A then Stream B, as TWO residencies.

    Never one call: DINO+SigLIP and Qwen3-VL are separate loads, and a run
    killed between them must resume at the second rather than redo both.

    `bank`/`embedder` are None unless --proto-arm asked for one. Passed
    straight through to GroundedVerifier, which only records their scores --
    grounded.py's live state logic is unchanged either way (grounded.py:91-93).
    """
    from ragregen import concepts

    #: The bank's embedder (if any) stays GPU-resident through both stages
    #: below -- see PROTO_EMBEDDER_VRAM_GB. Padding the preflight here rather
    #: than in STAGE_VRAM_GB itself keeps the base figures true for the
    #: default (embedder is None) case, which every existing run stays on.
    extra = PROTO_EMBEDDER_VRAM_GB if embedder is not None else 0.0

    def load_grounded():
        return grounded.GroundedVerifier(
            models.DinoDetector(device=device),
            models.build_crop_scorer(pipe_cfg.crop_scorer, device=device),
            tau=pipe_cfg.tau, prototypes=bank, embedder=embedder,
            prototype_delta=getattr(pipe_cfg, "prototype_delta", None))

    def score_grounded(verifier, cid):
        case = by_id[cid]
        #: parse takes the target and coarse term explicitly -- the same call
        #: score_a.py:41 makes. Passing only the prompt silently yields no
        #: fine-grained contrast.
        parsed = concepts.parse(case.prompt, target=case.concept,
                                coarse=case.coarse)
        scores = verifier.score(image_of(cid), parsed)
        _stash_grounded(run_dir, cid, attempt, scores)

        t = _trace(run_dir, cid)
        t.grounded({k: v.state for k, v in scores.items()})
        t.save(run_dir.case_dir(cid) / "trace.json")

    stage_with_model(queue, "grounded", load_grounded, score_grounded,
                     attempt=attempt, need_gb=STAGE_VRAM_GB["grounded"] + extra)

    def load_semantic():
        from ragregen import vlm

        #: QwenVLM, not Qwen3VL -- the class in ragregen/vlm.py:30, wired the
        #: same way score_b.py:71 wires it.
        return semantic.SemanticVerifier(vlm.QwenVLM(device=device))

    def judge(verifier, cid):
        case = by_id[cid]
        verdict = verifier.judge(image_of(cid), case.prompt)
        _stash_semantic(run_dir, cid, attempt, verdict)
        _trace_vlm(run_dir, cid, attempt, verdict.raw)

    stage_with_model(queue, "semantic", load_semantic, judge, attempt=attempt,
                     need_gb=STAGE_VRAM_GB["semantic"] + extra)

    _fuse_pending(queue, run_dir, by_id, attempt=attempt)


#: Statuses meaning the case has left the loop. Re-fusing one rewrites
#: scores.json for finished work on every resume
#: (findings/2026-07-29-orchestration-result.md §3).
RESOLVED = ("healthy", "repaired", "unrepaired", "failed")


def _fuse_pending(queue, run_dir, by_id, *, attempt: int) -> None:
    for cid in list(by_id):
        st = queue.cases.get(cid)
        if st is not None and st.status in RESOLVED:
            continue
        streams = run_dir.case_dir(cid) / "streams.json"
        if not streams.is_file():
            continue
        both = json.loads(streams.read_text()).get(str(attempt), {})
        if "grounded" not in both or "semantic" not in both:
            continue                      # one stream failed; not fusable
        label = "draft" if attempt == 0 else f"attempt_{attempt}"
        if _record_score(run_dir, cid, label, both):
            _resolve(queue, cid, label)


def _finalise_one(queue, run_dir, case_id: str) -> None:
    """Pick the candidate to report for one case.

    Reads scores.json -- {label: [ok, score]} -- which _record_score wrote.
    score may be null when Stream A saw no evidence for that candidate.
    """
    #: A case that failed at `mask` or `regen` can still carry a round-0
    #: scores.json (the draft's own verdict) -- without this guard, such a
    #: case gets relabelled unrepaired/repaired below, stranding its `error`
    #: string and undercounting run.json's `failed` list (_finish_results'
    #: docstring promises every terminal status is counted there).
    if queue.cases[case_id].status == "failed":
        return
    path = run_dir.case_dir(case_id) / "scores.json"
    if not path.is_file():
        return
    scored = json.loads(path.read_text())
    draft_ok, draft_score = scored.get("draft", [False, 0.0])
    #: score may be None -- Stream A saw no evidence for that candidate --
    #: and select_best already treats None as "cannot win". Coercing it
    #: through float() here is what used to crash, not the selection rule.
    attempts = [(label, bool(v[0]), None if v[1] is None else float(v[1]))
                for label, v in sorted(scored.items()) if label != "draft"]
    draft_score = None if draft_score is None else float(draft_score)
    #: select_winner, not select_best directly -- a no-op here (a live run's
    #: own invariant is draft_ok=True implies no attempts exist, since
    #: _resolve already stopped the round loop), but the same function
    #: reverify.reselect and report._best_label call, so all three agree by
    #: construction rather than by three independently-maintained rules.
    best = schedule.select_winner(bool(draft_ok), draft_score, attempts)
    queue.cases[case_id].best = best
    #: draft_ok is only consulted by _status_for when best == "draft" -- see
    #: its docstring for why a non-draft best is always "repaired".
    queue.cases[case_id].status = _status_for(best, bool(draft_ok))


def _finalise(queue, run_dir) -> None:
    for case_id in queue.cases:
        _finalise_one(queue, run_dir, case_id)
    queue.save()


def _refs_of(run_dir, case_id) -> list:
    p = run_dir.case_dir(case_id) / "refs.json"
    return json.loads(p.read_text()) if p.is_file() else []


def _retire_exhausted(queue, run_dir, attempt: int) -> None:
    """End the run for cases with no reference left for this attempt.

    Attempt N edits with refs[N - 1]. A case whose edit pool is shorter than
    `attempt` has nothing to try, and _regen_one would raise on the missing
    cutout -- which run_stage marks `failed`, inventing a failure out of a
    dataset that simply had fewer references.
    """
    for cid in list(queue.pending("regen", attempt=attempt)):
        if attempt > len(_refs_of(run_dir, cid)):
            _finalise_one(queue, run_dir, cid)
    queue.save()


def _fail_open_loop_without_refs(queue) -> None:
    """Fail pending no-verifier cases explicitly when no edit is possible."""
    for cid, state in queue.cases.items():
        if state.status == "pending":
            state.status = "failed"
            state.error = "open-loop ablation has no editable reference"
    queue.save()


def _resolve_open_loop(queue, run_dir, attempt: int) -> None:
    """Select the sole generated attempt without inventing verifier scores."""
    key = schedule.stage_key("regen", attempt)
    label = f"attempt_{attempt}"
    for cid, state in queue.cases.items():
        if state.status != "pending" or state.stages.get(key) != "done":
            continue
        state.best = label
        state.status = "repaired"
        (run_dir.case_dir(cid) / "selection.json").write_text(json.dumps({
            "best": label,
            "mode": "open_loop",
            "verifier_used": False,
        }, indent=2))
    queue.save()


def _candidate_labels(state) -> list[str]:
    attempts = []
    for key, outcome in state.stages.items():
        if outcome != "done" or not key.startswith("regen@"):
            continue
        try:
            attempts.append(int(key.split("@", 1)[1]))
        except ValueError:
            continue
    return [f"attempt_{attempt}" for attempt in sorted(attempts)]


def _resolve_candidate_bank(queue, run_dir, case_ids=None) -> None:
    """Finish cached open-loop cases while keeping attempt 1 as the arm.

    Later candidates exist solely for the frozen offline selector. Calling
    them the no-verifier winner would silently turn candidate generation into
    a best-of-N method with no valid selection rule.
    """
    wanted = set(case_ids) if case_ids is not None else set(queue.cases)
    for cid, state in queue.cases.items():
        if cid not in wanted or state.status != "pending":
            continue
        labels = _candidate_labels(state)
        if "attempt_1" not in labels:
            continue
        state.best = "attempt_1"
        state.status = "repaired"
        (run_dir.case_dir(cid) / "selection.json").write_text(json.dumps({
            "best": "attempt_1",
            "mode": "open_loop",
            "verifier_used": False,
            "candidate_attempts": labels,
        }, indent=2))
    queue.save()


def _retire_open_loop_exhausted(queue, run_dir, attempt: int) -> None:
    """Finish shorter-reference cases before a candidate-bank round."""
    exhausted = [
        cid for cid in queue.pending("regen", attempt=attempt)
        if attempt > len(_refs_of(run_dir, cid))
    ]
    _resolve_candidate_bank(queue, run_dir, exhausted)


if __name__ == "__main__":
    # Not SystemExit: unloading torch+faiss+PIL together segfaults in this
    # env. Here it would also be self-perpetuating -- a resumed run finds
    # every case already done and reaches teardown, the only thing left to
    # crash in. See env.exit_now.
    env.exit_now(main())
