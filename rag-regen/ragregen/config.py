"""Typed loading of the three operator-editable YAML files.

These are the only files an operator edits (spec §7). Every failure here must
name the offending case or key, because the person reading the message is not
going to open the Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragregen import env

DEFAULT_DATASET_PATH = env.PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_DB_PATH = env.PROJECT_ROOT / "configs" / "retrieval_db.yaml"
DEFAULT_PIPELINE_PATH = env.PROJECT_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    concept: str
    coarse: str
    #: "target" = a fine-grained concept the method aims at; "control" = one
    #: FLUX should already render. If the controls fail too, the drafts are
    #: broken rather than the concepts rare -- which is what makes the fail
    #: rate interpretable (configs/dataset.yaml header).
    kind: str = "target"
    #: "bridge" = one of the 22 legacy cases with full-resolution gt_refs;
    #: "common" = an ImageNet-derived case with 256px refs. The report never
    #: averages DINO across the two -- different reference resolutions
    #: (doc 5 §5).
    cohort: str = "bridge"
    gt_refs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images_root: Path
    cases: list[Case]
    #: Reference images per COARSE term, for the prototype contrast. Keyed by
    #: term rather than by case so the terms several cases share ("dog",
    #: "tree", "bird", "bear") are authored once.
    coarse_refs: dict[str, list[Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalDBConfig:
    name: str
    images_root: Path
    index_path: Path
    encoder: str
    captions: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    retry_budget: int
    tau: float
    steps: int
    seed: int
    crop_scorer: str
    retriever: str
    #: Slack around the draft mask, in pixels. A pixel-tight mask leaves a halo
    #: of the ORIGINAL object's edge which the inpainter rebuilds, reintroducing
    #: what we asked it to replace. Owned here, never applied twice: regen's
    #: KontextConfig.dilate_px is 0.
    mask_dilate_px: int = 12
    #: How a retrieved/oracle photograph is presented to Kontext.  Passing a
    #: full photograph lets unrelated people, text, scenery and style leak
    #: into the edit region.  ``crop`` is the production default; ``none`` is
    #: retained as the explicit ablation that reproduces the old behaviour.
    ref_prep: str = "crop"
    #: Optional image-prototype margin used by the live grounded verifier.
    #: None preserves the measured text-only baseline. A value must be chosen
    #: on a development split, never on the final evaluation cases.
    prototype_delta: float | None = None
    #: References reserved for scoring and never edited with (doc 4 §2).
    eval_heldout_refs: int = 1
    eval_dino: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    eval_clip: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    eval_siglip: str = "google/siglip-base-patch16-384"


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _require(data: dict, key: str, path: Path):
    if key not in data:
        raise ValueError(f"{path}: missing required key '{key}'")
    return data[key]


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> DatasetConfig:
    data = _read(path)
    root = Path(_require(data, "images_root", path))
    raw_cases = _require(data, "cases", path)
    if not raw_cases:
        raise ValueError(f"{path}: 'cases' is empty")

    KINDS = ("target", "control")
    COHORTS = ("bridge", "common")

    seen: set[str] = set()
    cases: list[Case] = []
    for raw in raw_cases:
        cid = raw.get("id")
        if not cid:
            raise ValueError(f"{path}: a case is missing 'id'")
        if cid in seen:
            raise ValueError(f"{path}: duplicate case id: {cid}")
        seen.add(cid)
        for key in ("prompt", "concept", "coarse"):
            if not raw.get(key):
                raise ValueError(f"{path}: case '{cid}' has no {key}")
        if raw["coarse"].strip().lower() == raw["concept"].strip().lower():
            raise ValueError(
                f"{path}: case '{cid}' has coarse == concept "
                f"('{raw['concept']}'). The coarse term must be the concept's "
                f"superordinate category -- concept 'African grey parrot' -> "
                f"coarse 'parrot'. Identical terms make the fine-grained "
                f"margin identically zero, silently disabling the test."
            )
        refs = raw.get("gt_refs") or []
        if not refs:
            raise ValueError(
                f"{path}: case '{cid}' has no gt_refs. They are required: the "
                f"DINO identity metric and the 'oracle' arm both need them."
            )
        kind = str(raw.get("kind", "target"))
        if kind not in KINDS:
            raise ValueError(
                f"case {cid!r} has kind {kind!r}; expected one of {KINDS}")
        cohort = str(raw.get("cohort", "bridge"))
        if cohort not in COHORTS:
            raise ValueError(
                f"case {cid!r} has cohort {cohort!r}; expected one of "
                f"{COHORTS}")
        cases.append(Case(
            id=cid,
            prompt=raw["prompt"],
            concept=raw["concept"],
            coarse=raw["coarse"],
            kind=kind,
            cohort=cohort,
            gt_refs=[root / r for r in refs],
        ))
    known = {c.coarse for c in cases}
    coarse_refs: dict[str, list[Path]] = {}
    for term, refs in (data.get("coarse_refs") or {}).items():
        if term not in known:
            raise ValueError(
                f"{path}: coarse_refs has an entry for '{term}', which is not "
                f"the coarse term of any case. Known terms: "
                f"{', '.join(sorted(known))}.")
        coarse_refs[term] = [root / r for r in refs]

    return DatasetConfig(name=_require(data, "name", path), images_root=root,
                         cases=cases, coarse_refs=coarse_refs)


def load_retrieval_db(path: Path = DEFAULT_DB_PATH) -> RetrievalDBConfig:
    data = _read(path)
    captions = data.get("captions")
    #: index_path is commonly authored relative (configs/retrieval_db.yaml
    #: ships `data/rag_db/index.faiss`). Resolve it here, once, against the
    #: repo root -- every entry point (validate_all, report.main, build-index,
    #: run_pipeline, score_a) reads db.index_path and must agree regardless
    #: of the process's working directory (ledger MINOR 13). An already
    #: absolute path passes through untouched.
    index_path = Path(_require(data, "index_path", path))
    if not index_path.is_absolute():
        index_path = env.PROJECT_ROOT / index_path
    return RetrievalDBConfig(
        name=_require(data, "name", path),
        images_root=Path(_require(data, "images_root", path)),
        index_path=index_path,
        encoder=_require(data, "encoder", path),
        captions=Path(captions) if captions else None,
    )


def load_pipeline(path: Path = DEFAULT_PIPELINE_PATH) -> PipelineConfig:
    data = _read(path)
    crop_scorer = data.get("crop_scorer", "siglip_so400m_384")
    retriever = data.get("retriever", "siglip_so400m_384")
    ref_prep = str(data.get("ref_prep", "crop"))
    if ref_prep not in ("none", "crop", "crop_matte"):
        raise ValueError(
            f"ref_prep is {ref_prep!r}; expected one of none, crop, "
            f"crop_matte")
    raw_proto_delta = data.get("prototype_delta")
    prototype_delta = (None if raw_proto_delta is None
                       else float(raw_proto_delta))
    if prototype_delta is not None and not -2.0 <= prototype_delta <= 2.0:
        raise ValueError("prototype_delta must be in [-2, 2] or null")

    ev = data.get("eval") or {}
    heldout = int(ev.get("heldout_refs", 1))
    if heldout < 1:
        raise ValueError(
            f"eval.heldout_refs is {heldout}; it must be >= 1, or DINO scores "
            f"the oracle arm against the very images it edited with.")

    eval_dino = str(ev.get("dino", PipelineConfig.eval_dino))
    eval_clip = str(ev.get("clip", PipelineConfig.eval_clip))
    eval_siglip = str(ev.get("siglip", PipelineConfig.eval_siglip))

    #: The paper retrieves with one checkpoint and evaluates with another on
    #: purpose (design §5). Scoring an output with the model that chose its
    #: reference is self-marking.
    in_loop = {str(retriever), str(crop_scorer)}
    for label, value in (("dino", eval_dino), ("clip", eval_clip),
                         ("siglip", eval_siglip)):
        if value in in_loop:
            raise ValueError(
                f"eval.{label} is {value!r}, which is also the retriever or "
                f"crop_scorer. That is self-marking -- pick a different "
                f"checkpoint for evaluation.")

    return PipelineConfig(
        retry_budget=int(data.get("retry_budget", 3)),
        tau=float(data.get("tau", 0.25)),
        steps=int(data.get("steps", 28)),
        seed=int(data.get("seed", 0)),
        crop_scorer=crop_scorer,
        retriever=retriever,
        mask_dilate_px=int(data.get("mask_dilate_px", 12)),
        ref_prep=ref_prep,
        prototype_delta=prototype_delta,
        eval_heldout_refs=heldout,
        eval_dino=eval_dino,
        eval_clip=eval_clip,
        eval_siglip=eval_siglip,
    )
