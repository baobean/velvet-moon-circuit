"""Preflight over operator-supplied data. Runs before any GPU work.

Every message names the offending file or case and says what to do. The reader
is an operator who does not edit Python (spec §7).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ragregen.config import DatasetConfig, PipelineConfig, RetrievalDBConfig

# Embedding widths, used to catch an index built with a different encoder.
ENCODER_DIMS: dict[str, int] = {
    "siglip_so400m_384": 1152,
    "siglip_base_224": 768,
    "fgclip": 512,
    "openclip_l14": 768,
    "clip_b32": 512,
}


@dataclass(frozen=True)
class Problem:
    severity: str   # "error" | "warning"
    code: str
    message: str


def _err(code: str, msg: str) -> Problem:
    return Problem("error", code, msg)


def _warn(code: str, msg: str) -> Problem:
    return Problem("warning", code, msg)


def validate_dataset(ds: DatasetConfig) -> list[Problem]:
    problems: list[Problem] = []
    for case in ds.cases:
        for ref in case.gt_refs:
            if not ref.is_file():
                problems.append(_err(
                    "gt_ref_missing",
                    f"case '{case.id}': gt_ref not found: {ref}. "
                    f"Check images_root and the paths under gt_refs.",
                ))
                continue
            try:
                with Image.open(ref) as im:
                    im.verify()
            except Exception as exc:
                problems.append(_err(
                    "gt_ref_unreadable",
                    f"case '{case.id}': cannot read {ref} ({type(exc).__name__}). "
                    f"Re-export it as JPEG or PNG.",
                ))
    return problems


def validate_leakage(ds: DatasetConfig, db: RetrievalDBConfig) -> list[Problem]:
    """Ground-truth references must not be retrievable (spec §7 / RUNBOOK §3.1)."""
    problems: list[Problem] = []
    try:
        corpus = db.images_root.resolve()
    except OSError as exc:
        problems.append(_warn(
            "leakage_check_skipped",
            f"cannot resolve {db.images_root} ({type(exc).__name__}): "
            f"gt_ref leakage check skipped for this path.",
        ))
        return problems
    for case in ds.cases:
        for ref in case.gt_refs:
            try:
                resolved = ref.resolve()
            except OSError as exc:
                problems.append(_warn(
                    "leakage_check_skipped",
                    f"cannot resolve {ref} ({type(exc).__name__}): "
                    f"gt_ref leakage check skipped for this path.",
                ))
                continue
            if corpus == resolved or corpus in resolved.parents:
                problems.append(_err(
                    "gt_ref_in_corpus",
                    f"case '{case.id}': gt_ref {ref} lives inside the retrieval "
                    f"corpus ({db.images_root}). The 'full' arm would retrieve its "
                    f"own answer key. Move it outside the corpus.",
                ))
    return problems


def validate_db(db: RetrievalDBConfig, expected_dim: int | None) -> list[Problem]:
    problems: list[Problem] = []
    if db.encoder not in ENCODER_DIMS:
        problems.append(_err(
            "unknown_encoder",
            f"retrieval_db.yaml: unknown encoder '{db.encoder}'. "
            f"Known: {', '.join(sorted(ENCODER_DIMS))}.",
        ))
        return problems

    want = ENCODER_DIMS[db.encoder]
    if expected_dim is not None and expected_dim != want:
        problems.append(_err(
            "index_dim_mismatch",
            f"index at {db.index_path} has dimension {expected_dim}, but encoder "
            f"'{db.encoder}' produces {want}. The index was built with a different "
            f"encoder — re-run `./scripts/run.sh build-index`.",
        ))
    if not db.index_path.is_file():
        problems.append(_warn(
            "index_absent",
            f"no index at {db.index_path} yet. Run `./scripts/run.sh build-index`.",
        ))
    return problems


def validate_disk(min_free_gb: float = 20.0,
                  path: Path = Path("/mnt/mmlab2024nas")) -> list[Problem]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        return [_err(
            "disk_low",
            f"only {free_gb:.0f} GB free on {path} (need >= {min_free_gb:.0f} GB). "
            f"This volume is shared — see RUNBOOK §6. Re-encode the corpus at 384px "
            f"before uploading.",
        )]
    if free_gb < min_free_gb * 5:
        return [_warn(
            "disk_tight",
            f"{free_gb:.0f} GB free on {path}. Shared volume — see RUNBOOK §6.",
        )]
    return []


def validate_all(ds: DatasetConfig, db: RetrievalDBConfig,
                 pipe: PipelineConfig) -> list[Problem]:
    problems = validate_dataset(ds)
    problems += validate_leakage(ds, db)
    problems += validate_db(db, expected_dim=None)
    problems += validate_disk()
    #: The manifest lives beside the index, so retrieval_db.yaml's one path
    #: key locates both. A missing manifest warns rather than raising -- the
    #: oracle arm needs no corpus at all.
    problems += validate_corpus_density(
        db.index_path.parent / "corpus_manifest.json", ds, pipe.retry_budget)
    if pipe.retry_budget < 1:
        problems.append(_err(
            "bad_retry_budget",
            f"pipeline.yaml: retry_budget must be >= 1, got {pipe.retry_budget}.",
        ))
    if not 0.0 < pipe.tau < 1.0:
        problems.append(_err(
            "bad_tau",
            f"pipeline.yaml: tau must be strictly between 0 and 1, got {pipe.tau}.",
        ))
    return problems


#: Below this many references a prototype is dominated by individual
#: photographs rather than the category. A warning, not an error: the operator
#: may be running the phrase mechanism only.
MIN_COARSE_REFS = 3


def check_heldout_refs(case_ref_counts, heldout_refs: int) -> list[str]:
    """Cases that cannot afford the hold-out.

    Takes (case_id, n_refs) pairs rather than a dataset so it stays testable
    without touching disk.
    """
    problems = []
    for case_id, n in case_ref_counts:
        if n - heldout_refs < 1:
            problems.append(
                f"case {case_id!r} has {n} gt_refs; holding out "
                f"{heldout_refs} leaves nothing to edit with. Add a "
                f"reference or lower eval.heldout_refs.")
    return problems


def check_coarse_refs(dataset) -> list[str]:
    """Warn about thin coarse reference sets.

    Spread cannot be checked mechanically -- three photographs of the same
    parrot satisfy any count -- so this catches only the countable half and
    the RUNBOOK carries the rest.
    """
    out = []
    for term, refs in sorted(dataset.coarse_refs.items()):
        if len(refs) < MIN_COARSE_REFS:
            out.append(
                f"coarse term '{term}' has only {len(refs)} reference "
                f"image(s); {MIN_COARSE_REFS}+ recommended, spread across the "
                f"category rather than several photographs of one member.")
    return out


def validate_corpus_density(manifest_path: Path, ds: DatasetConfig,
                            k: int) -> list[Problem]:
    """Which concepts the corpus cannot serve `k` references for.

    Warnings, never errors. A thin concept is a result the report publishes
    beside its case (doc 5 §4) -- aborting here would trade a known-weak row
    for no row at all, hours after a GPU block was booked.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return [_warn(
            "corpus_manifest_missing",
            f"no corpus manifest at {manifest_path}. Retrieval density is "
            f"unknown; run `./scripts/run.sh fetch-corpus manifest`. The "
            f"oracle arm does not need it.",
        )]

    counts = json.loads(manifest_path.read_text()).get("per_concept", {})
    thin = [c.concept for c in ds.cases if int(counts.get(c.concept.lower(), 0)) < k]
    if not thin:
        return []
    return [_warn(
        "corpus_thin",
        f"{len(thin)} concept(s) have fewer than k={k} caption matches in the "
        f"corpus: {', '.join(sorted(thin)[:10])}"
        f"{' ...' if len(thin) > 10 else ''}. Retrieval will return "
        f"off-concept references for these; the report records the count. "
        f"To deepen the corpus, re-run `./scripts/run.sh fetch-corpus select "
        f"--n-per-concept 400` and download again; to accept them, proceed "
        f"and read these rows against their corpus_density column.",
    )]
