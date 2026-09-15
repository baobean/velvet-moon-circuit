#!/usr/bin/env python
"""Turn a finished pipeline run into the results table. Doc 4.

Buckets come from scores.json's DRAFT VERDICT, never from queue.json's
`status`, which is written by two paths meaning opposite things
(findings/2026-07-29-orchestration-result.md §2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ragregen import metrics, schedule  # noqa: E402


def _best_label(scores: dict) -> str:
    """The candidate the run chose, re-derived with the run's own rule.

    Deliberately schedule.select_winner rather than an argmax over scores: a
    PASSING attempt wins outright, even against a failed draft that scored as
    high. `axolotl` in outputs/doc4_20260731_095641 is exactly that shape --
    draft [False, 1.0], attempt_1 [True, 1.0] -- and an argmax calls a repaired
    case unrepairable. select_winner also short-circuits a genuinely PASSING
    draft against a later attempt that also passes -- the population
    `reverify` re-grades, where an attempt can survive a draft that a
    corrected scoring now passes for the first time. Same expression as
    reverify.reselect and run_pipeline._finalise_one, so the three cannot
    disagree about which image a case's metrics describe.
    """
    draft_ok, draft_raw = scores.get("draft", [False, 0.0])
    #: score may be None -- Stream A saw no evidence for that candidate --
    #: and select_best already treats None as "cannot win"; float() is what
    #: used to crash, not the selection rule. Same coercion as
    #: run_pipeline._finalise_one, so the two cannot disagree.
    draft_score = None if draft_raw is None else float(draft_raw)
    attempts = [(label, bool(v[0]), None if v[1] is None else float(v[1]))
                for label, v in sorted(scores.items()) if label != "draft"]
    return schedule.select_winner(bool(draft_ok), draft_score, attempts)


def bucket(scores: dict) -> str:
    """Which of the three groups this case belongs to.

    healthy       the draft passed; there was nothing to repair
    repaired      the draft failed and the run chose an attempt over it
    unrepairable  the draft failed and nothing beat it

    "Beat it" is spec §4's definition -- `best != "draft"` -- not a score
    comparison. See _best_label.
    """
    if bool(scores.get("draft", [False, 0.0])[0]):
        return "healthy"
    return "repaired" if _best_label(scores) != "draft" else "unrepairable"


def repair_rate(buckets) -> float | None:
    """repaired / (repaired + unrepairable), or None if nothing needed repair.

    Healthy cases are excluded from the denominator. Including them is the
    mistake `status` invites, and it inflates the rate.
    """
    needed = [b for b in buckets if b in ("repaired", "unrepairable")]
    if not needed:
        return None
    return sum(1 for b in needed if b == "repaired") / len(needed)


def _selected_image(case_dir: Path, scores: dict, draft: Image.Image, *,
                    selected_override: str | None = None):
    """The candidate the pipeline chose. The draft when nothing beat it.

    Scored image and reported bucket come from the same rule, so the metrics
    always describe the image queue.json calls `best`.
    """
    label = selected_override or _best_label(scores)
    if label == "draft":
        return draft, "draft"
    path = case_dir / f"{label}.png"
    if not path.is_file():
        return draft, "draft"
    return Image.open(path).convert("RGB"), label


def score_case(case, case_dir: Path, draft: Image.Image, encoders, *,
               heldout_paths, mask_failed: bool = False,
               selected_override: str | None = None) -> dict:
    """Every metric for one case, or an explanation of why not.

    Healthy cases never ran `mask`, so they have no bounding box. They are
    scored whole-image in their own column rather than being averaged into a
    cropped mean -- comparing a tight object crop against a whole scene and
    reading the difference as identity (spec §4).

    `mask_failed` says the run TRIED to mask this case and the detector came
    back empty ("'ship' did not ground in the draft"). Such a case has a
    failed draft and no mask.png -- indistinguishable, from the directory
    alone, from a run that was interrupted before `mask` ran. Only queue.json
    can tell those apart, so main() reads it and passes the answer down.
    """
    scores_path = case_dir / "scores.json"
    if scores_path.is_file():
        scores = json.loads(scores_path.read_text())
        group = bucket(scores)
    elif selected_override is not None:
        scores = {}
        group = "repaired" if selected_override != "draft" else "unrepairable"
    else:
        raise FileNotFoundError(f"{case.id}: no scores.json")
    output, label = _selected_image(
        case_dir, scores, draft, selected_override=selected_override)

    if output.size != draft.size:
        raise ValueError(
            f"{case.id}: output size {output.size} != draft {draft.size}")

    mask_path = case_dir / "mask.png"
    mask = (Image.open(mask_path).convert("L") if mask_path.is_file()
            else None)
    if mask is not None and mask.size != draft.size:
        #: The Global Constraint is enforced on BOTH inputs or on neither.
        #: crop_to_mask derives the bbox from the mask's grid and applies it
        #: to the draft's, clamped -- a mask at another resolution yields a
        #: wrong crop box on both sides of the pair, self-consistently, so
        #: the delta still prints a plausible number.
        raise ValueError(
            f"{case.id}: mask size {mask.size} != draft {draft.size}")
    refs = [Image.open(p).convert("RGB") for p in heldout_paths]

    row = {
        "case_id": case.id,
        "kind": case.kind,
        "cohort": getattr(case, "cohort", "bridge"),
        "bucket": group,
        "selected": label,
        "mask_failed": bool(mask_failed),
        "dino_cropped": None,
        "dino_cropped_draft": None,
        "dino_delta": None,
        "dino_whole": None,
        "preservation": None,
        "clip": metrics.prompt_alignment(output, case.prompt, encoders.clip),
        "siglip": metrics.prompt_alignment(output, case.prompt,
                                           encoders.siglip),
    }

    if mask is None:
        if group != "healthy" and not mask_failed:
            raise FileNotFoundError(
                f"{case.id}: no mask.png, but its draft failed -- this case "
                f"should have run the `mask` stage. The run is damaged.")
        row["dino_whole"] = metrics.dino_identity(output, refs, encoders.dino)
        if group == "healthy":
            #: `best` is the draft here, so the pair is the same image twice.
            #: Zero by construction, and summarise excludes it from the mean.
            row["dino_delta"] = 0.0
        #: A mask-failed case leaves dino_delta as None instead. Its `best` is
        #: also the draft, but for the opposite reason: not "repair was
        #: unnecessary" but "repair was never attempted". 0.0 would enter the
        #: `unchanged` count as a repair that moved the metric nowhere, which
        #: reads as a null result from the mechanism rather than a case the
        #: mechanism never saw.
        return row

    alpha = (np.array(mask).astype(np.float32) / 255.0 > 0.5).astype(
        np.float32)
    #: The SAME mask and the SAME held-out refs on both sides. Any divergence
    #: here makes the headline meaningless while still printing a plausible
    #: number (doc 5 §7), which is why it is pinned by a test. The predicate
    #: is draft.size, the one resolution every metric computes at -- mask.size
    #: is guaranteed equal to it above, and comparing against it instead just
    #: made the mask the reference frame by accident.
    ref_crops = [metrics.crop_to_mask(r, mask) if r.size == draft.size else r
                 for r in refs]
    row["dino_cropped"] = metrics.dino_identity(
        metrics.crop_to_mask(output, mask), ref_crops, encoders.dino)
    row["dino_cropped_draft"] = metrics.dino_identity(
        metrics.crop_to_mask(draft, mask), ref_crops, encoders.dino)
    row["dino_delta"] = row["dino_cropped"] - row["dino_cropped_draft"]
    row["preservation"] = metrics.preservation(draft, output, alpha)["score"]
    return row


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarise(rows) -> dict:
    """Per-stratum means, keyed "<kind>/<cohort>".

    Cropped and whole-image DINO stay separate (doc 4 §4), and cohorts stay
    separate too: `bridge` cases carry full-resolution references, `common`
    cases 256px ones, and one mean across both would read a resolution
    difference as identity (doc 5 §5).
    """
    out = {}
    keys = sorted({(r["kind"], r.get("cohort", "bridge")) for r in rows})
    for kind, cohort in keys:
        group = [r for r in rows
                 if r["kind"] == kind and r.get("cohort", "bridge") == cohort]
        #: Healthy cases never entered repair; their delta is zero by
        #: construction and averaging it in would dilute a real effect.
        repaired = [r for r in group if r["bucket"] != "healthy"]
        deltas = [r["dino_delta"] for r in repaired
                  if r["dino_delta"] is not None]
        ci = metrics.paired_bootstrap_ci(deltas, seed=0) if deltas else None
        out[f"{kind}/{cohort}"] = {
            "n": len(group),
            #: The CI's and dino_delta's actual denominator. `n` is the whole
            #: stratum, and a row reading "n 70 | healthy 68 | [+0.01,
            #: +0.01]" invites attributing the interval to 70 cases when it
            #: came from 2.
            "n_delta": len(deltas),
            "healthy": sum(1 for r in group if r["bucket"] == "healthy"),
            "repaired": sum(1 for r in group if r["bucket"] == "repaired"),
            "unrepairable": sum(1 for r in group
                                if r["bucket"] == "unrepairable"),
            #: A subset of `unrepairable`, broken out because it is a
            #: different claim. These cases never reached the mechanism at
            #: all -- the detector could not find the object in the draft, so
            #: there was no region to repair. Folded silently into
            #: `unrepairable` they read as "the mechanism tried and lost" and
            #: understate its repair rate.
            "mask_failed": sum(1 for r in group if r.get("mask_failed")),
            "repair_rate": repair_rate([r["bucket"] for r in group]),
            "dino_cropped": _mean(r["dino_cropped"] for r in group),
            "dino_cropped_draft": _mean(r["dino_cropped_draft"]
                                        for r in group),
            "dino_delta": _mean(deltas),
            "delta_ci": ci,
            "delta_verdict": metrics.delta_verdict(*ci) if ci else None,
            #: `unchanged` is `best == "draft"` -- the pipeline declined to
            #: act -- OR a real edit whose dino_delta landed at exactly
            #: 0.0. Exact equality with zero is not a tunable threshold, so
            #: this stays independent of MARGIN; it just means an edit that
            #: moved the metric not at all honestly reads "=" rather than
            #: falling through all three counts. Rows with an unscored
            #: (None) delta are deliberately excluded from all three -- `or
            #: 0` would silently fold them into "unchanged".
            "unchanged": sum(1 for r in repaired
                            if r["selected"] == "draft"
                            or r["dino_delta"] == 0.0),
            "improved": sum(1 for r in repaired
                            if r["selected"] != "draft"
                            and r["dino_delta"] is not None
                            and r["dino_delta"] > 0),
            "worsened": sum(1 for r in repaired
                            if r["selected"] != "draft"
                            and r["dino_delta"] is not None
                            and r["dino_delta"] < 0),
            "dino_whole": _mean(r["dino_whole"] for r in group),
            "preservation": _mean(r["preservation"] for r in group),
            "clip": _mean(r["clip"] for r in group),
            "siglip": _mean(r["siglip"] for r in group),
        }
    return out


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def render(summary: dict, meta: dict) -> str:
    """The human-readable table, with its own caveats attached."""
    lines = [
        f"# Results — {meta['run']}",
        "",
        f"Arm: **{meta['arm']}**  ·  mechanism: **{meta['mechanism']}**  ·  "
        f"verifier: **{meta.get('verifier', 'fused')}**  ·  "
        f"held-out refs: **{meta['heldout_refs']}**  ·  "
        f"non-inferiority margin: **{metrics.MARGIN}** DINO",
        "",
        "| stratum | n | healthy | repaired | unrepairable | repair rate | "
        "DINO draft | DINO best | **delta** | 95% CI | verdict | "
        "+/=/− | preservation | CLIP | SigLIP |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        rate = "—" if s["repair_rate"] is None else f"{s['repair_rate']:.0%}"
        #: The interval carries its own denominator. Without it the reader
        #: attributes it to the row's `n`, which counts healthy cases the CI
        #: never saw.
        ci = ("—" if s["delta_ci"] is None
              else f"[{s['delta_ci'][0]:+.3f}, {s['delta_ci'][1]:+.3f}]")
        ci = f"{ci} (n={s['n_delta']})"
        delta = "—" if s["dino_delta"] is None else f"{s['dino_delta']:+.3f}"
        #: Inline rather than as its own column: these cases are already
        #: counted in `unrepairable`, and a separate column would invite
        #: adding the two together.
        unrep = str(s["unrepairable"])
        if s.get("mask_failed"):
            unrep += f" ({s['mask_failed']} unmasked)"
        lines.append(
            f"| {name} | {s['n']} | {s['healthy']} | {s['repaired']} | "
            f"{unrep} | {rate} | {_fmt(s['dino_cropped_draft'])} "
            f"| {_fmt(s['dino_cropped'])} | **{delta}** | {ci} | "
            f"{s['delta_verdict'] or '—'} | "
            f"{s['improved']}/{s['unchanged']}/{s['worsened']} | "
            f"{_fmt(s['preservation'])} | {_fmt(s['clip'])} | "
            f"{_fmt(s['siglip'])} |")

    lines += [
        "",
        "## How to read this",
        "",
        f"- **The delta is the claim.** `DINO(best) − DINO(draft)`, paired "
        f"within each case: same prompt, same mask, same held-out references "
        f"on both sides. `no_harm` means the CI's lower bound clears "
        f"{metrics.MARGIN}; `harm` means the upper bound is below zero.",
        "- **`no_harm` is a non-inferiority verdict; it does not mean zero "
        "change.** An interval that lies entirely below zero but stays "
        "above the margin -- e.g. (-0.015, -0.005) -- still reads `no_harm`; "
        "it means the drop is too small to call harm, not that there was "
        "no drop.",
        "- **The margin was fixed before any data existed** (doc 5 §2). It is "
        "not tuned to the result.",
        f"- **The CI covers the repair cases only.** Its `(n=…)` is the "
        f"number of scored repair cases behind the interval — never the "
        f"row's `n`, which counts healthy cases the delta never saw. Below "
        f"{metrics.MIN_CI_N} of them there is no interval at all: every "
        f"resample of a one-case stratum is that same case, and a zero-width "
        f"interval reads as precision it does not have.",
        "- **Healthy cases are excluded from the delta.** Their delta is zero "
        "by construction — `best` is the draft — and averaging them in would "
        "dilute a real effect toward zero.",
        "- **`+/=/−` counts repair cases only.** `=` means the pipeline chose "
        "the draft; it is decided by `best`, not by a threshold on the delta.",
        "- **Strata never merge.** `bridge` cases carry full-resolution "
        "references and `common` cases 256px ones; one mean across both would "
        "read a resolution difference as identity.",
        "- **`(n unmasked)` cases never reached the mechanism.** The detector "
        "did not find the concept in the draft, so `mask` failed and there "
        "was no region to repair. They count as `unrepairable` — the system "
        "did fail to repair them — but they are NOT evidence about the "
        "mechanism, which never ran on them. Subtract them from the "
        "denominator to read the mechanism's own repair rate, and note that "
        "they carry no delta and no crop, only whole-image DINO.",
        "- **Read preservation next to the delta.** Identity bought by "
        "repainting the canvas is not a repair.",
        "- **DINO (whole) is not comparable to DINO (crop).** It covers "
        "healthy cases, which never ran `mask` and so have no box to crop to.",
        "- **Repair rate is not comparable across cohorts.** Rounds clamp to "
        "the number of editable references, so `common` cases get 2 and "
        "`bridge` cases 2-3. More rounds is more chances to repair; read each "
        "cohort's rate against itself, never against the other's.",
        "- **Verifier pass-rate is absent on purpose.** The pipeline "
        "optimises against the verifier, so it is a development signal only.",
    ]
    if meta.get("verifier") == "none":
        lines += [
            "- **This is the open-loop ablation.** Every maskable case was "
            "edited exactly once with its first reference. `repaired` means "
            "the generated attempt was evaluated, not that a verifier "
            "declared it correct; there was no verifier routing, early stop, "
            "or best-of-N selection.",
        ]
    return "\n".join(lines) + "\n"


def attach_density(rows, manifest_path, cases):
    """Per-concept corpus counts, recorded beside each case's metrics.

    None means unknown -- no manifest -- which is not the same claim as zero
    ("searched, and the corpus had nothing"). Doc 5 §6 keeps this per-case so
    a weak row can be attributed to corpus sparsity rather than guessed at.

    `per_concept`'s keys are lowercase (fetch_corpus.py normalises them,
    because configs/dataset.yaml's "Boston bull" and
    configs/imagenet_coarse.yaml's "boston bull" would otherwise collide into
    two keys). `case.concept` stays mixed-case, so the lookup lowercases it
    first -- same fix as validate.py's `c.concept.lower()`.
    """
    counts = {}
    manifest_path = Path(manifest_path)
    if manifest_path.is_file():
        counts = json.loads(manifest_path.read_text()).get("per_concept", {})
    by_id = {c.id: c.concept for c in cases}
    for row in rows:
        concept = by_id.get(row["case_id"], "")
        row["corpus_density"] = counts.get(concept.lower())
    return rows


def main() -> int:
    from ragregen import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="an outputs/pipeline_<TS>/ to report on")
    ap.add_argument("--screen-run", type=Path,
                    default=Path("outputs/screen_latest"))
    ap.add_argument("--dataset", type=Path,
                    default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path,
                    default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    run_info = json.loads((args.run / "run.json").read_text())
    verifier_mode = run_info.get("args", {}).get("verifier", "fused")

    #: Which cases the run tried to mask and could not. A case dir alone
    #: cannot distinguish "the detector found nothing" from "the run died
    #: before `mask`"; both are a failed draft with no mask.png. queue.json
    #: is the only record that separates them, so read it here rather than
    #: guessing per case. Absent queue.json, the set is empty and score_case
    #: keeps raising on every such case -- the safe direction.
    mask_failed = set()
    queue = {}
    queue_path = args.run / "queue.json"
    if queue_path.is_file():
        queue = json.loads(queue_path.read_text()).get("cases", {})
        mask_failed = {cid for cid, st in queue.items()
                       if st.get("stages", {}).get("mask") == "failed"}

    encoders = metrics.EvalEncoders.load(pipe_cfg, args.device)
    rows, skipped = [], []
    try:
        for case in ds.cases:
            d = args.run / case.id
            state = queue.get(case.id, {})
            if verifier_mode == "none":
                #: A mask-grounding failure never gets a `best` field (the
                #: open-loop finalisers that set it never run for it), but
                #: it is still the documented "retain the draft" fallback --
                #: force it to `draft` rather than falling through to the
                #: generic "no output" skip, which would silently drop it
                #: from every stratum's denominator instead of counting it
                #: `unrepairable (N unmasked)` per the table's own footnote.
                selected_override = ("draft" if case.id in mask_failed
                                     else state.get("best"))
            else:
                selected_override = None
            if not (d / "scores.json").is_file() and selected_override is None:
                skipped.append({"case_id": case.id, "why": "no output"})
                continue
            held_path = d / "heldout_refs.json"
            if held_path.is_file():
                held = [Path(p) for p in json.loads(held_path.read_text())]
            else:
                _, held = metrics.split_refs(case.gt_refs,
                                             pipe_cfg.eval_heldout_refs)
            draft = Image.open(
                args.screen_run / case.id / "draft.png").convert("RGB")
            rows.append(score_case(case, d, draft, encoders,
                                   heldout_paths=held,
                                   mask_failed=case.id in mask_failed,
                                   selected_override=selected_override))
    finally:
        encoders.free()

    db = config.load_retrieval_db()
    attach_density(rows, db.index_path.parent / "corpus_manifest.json",
                   ds.cases)

    summary = summarise(rows)
    meta = {"run": args.run.name,
            "arm": run_info.get("args", {}).get("arm", "?"),
            "mechanism": run_info.get("args", {}).get("mechanism", "?"),
            "verifier": verifier_mode,
            "heldout_refs": pipe_cfg.eval_heldout_refs}

    (args.run / "report.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "rows": rows, "skipped": skipped},
        indent=2, default=str))
    (args.run / "report.md").write_text(render(summary, meta))
    print(f"[report] {len(rows)} cases, {len(skipped)} skipped -> "
          f"{args.run / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
