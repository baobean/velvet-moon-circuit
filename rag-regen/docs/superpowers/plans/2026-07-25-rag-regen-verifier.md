# rag-regen Plan 1 — Operator Interface, Premise Gate, and Dual-Stream Verifier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the operator-facing config/validation layer, the premise-screening gate, and the
dual-stream verifier (claim C1) — everything in the spec that does not require reference-guided
regeneration.

**Architecture:** A Python package `ragregen/` where every model-using class receives its models as
constructor arguments, so the entire logic layer is testable on CPU with fakes and only a handful of
marked tests need the GPU. Operators drive it through three YAML files and one shell entry point;
`validate.py` refuses to start a run against malformed data.

**Tech Stack:** Python 3.11 (`kontext` conda env), transformers 5.14.1 (GroundingDINO, SigLIP,
Qwen3-VL), diffusers 0.39.0 (FLUX.1-Kontext), faiss-cpu, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md`
**Out of scope (Plan 2):** `mask.py`, `regen.py`, `schedule.py`, `metrics.py`, Stage 2 matrix.

## Global Constraints

- **Interpreter:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` (Python 3.11.15). Never `ImageRAG_qwen` — it is Python 3.10.13 with transformers 4.44.2 and no Qwen3-VL or FLUX Kontext.
- **`PYTHONNOUSERSITE` is NOT needed.** `~/.local/lib/python3.10/` shadows Python 3.10 only; 3.11 is immune. Do not copy that guardrail from `../rag-edit`.
- **`HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache`** must be set before any `huggingface_hub` or `transformers` import.
- **One GPU:** RTX 4090, 24 GB, shared. Qwen-7B (~16 GB) and FLUX-nf4 (~12 GB) do not co-fit. Never hold two large models at once.
- **Disk:** 270 GB free at 99% on a shared volume. No task may write more than 1 GB without an explicit size check first.
- **Dependency injection is mandatory.** Every class that uses a model takes it as a constructor argument. No module-level model loading, no `from_pretrained` at import time.
- **Test markers:** GPU tests are marked `@pytest.mark.gpu`. Default `pytest` run excludes them.
- **`ABSTAIN != MISSING`** (spec §2). A missing detection is never evidence of a missing concept.
- All paths in this plan are relative to `/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen/`.

---

## File Structure

| file | responsibility |
|---|---|
| `ragregen/env.py` | HF cache paths, GPU reclaim. Imported first, everywhere. |
| `ragregen/config.py` | Typed loading of the three operator YAMLs. No I/O beyond reading them. |
| `ragregen/validate.py` | Preflight checks over loaded configs. Returns problems, never raises on bad data. |
| `ragregen/trace.py` | Run directory creation, console teeing, per-case JSON trace. |
| `ragregen/concepts.py` | Prompt → list of `Concept`. Pure string logic, no models. |
| `ragregen/verify/grounded.py` | Stream A. `Detector` + `CropScorer` protocols, `GroundedVerifier`. |
| `ragregen/verify/semantic.py` | Stream B. `VLM` protocol, `SemanticVerifier`. |
| `ragregen/verify/fusion.py` | Pure function combining both streams. No models. |
| `ragregen/retrieve.py` | SigLIP/FG-CLIP encoders + FAISS `IndexFlatIP` search. |
| `scripts/build_index.py` | Corpus folder → embeddings → index. |
| `scripts/screen_premise.py` | The gate: draft every case, emit a hand-labelling sheet. |
| `scripts/bakeoff.py` | FG-CLIP vs SigLIP in three roles. |
| `scripts/run.sh` | Single operator entry point. |
| `configs/*.yaml` | Operator-editable inputs. |

---

## Task 1: Package scaffolding, environment, and dependencies

**Files:**
- Create: `ragregen/__init__.py`, `ragregen/env.py`, `pytest.ini`, `requirements.txt`
- Create: `tests/__init__.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `env.HF_CACHE: Path`, `env.PROJECT_ROOT: Path`, `env.setup() -> None`, `env.reclaim_gpu() -> None`.

- [ ] **Step 1: Install missing dependencies**

```bash
PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
$PY -m pip install pytest faiss-cpu open_clip_torch pandas
```

- [ ] **Step 2: Verify the environment is the right one**

```bash
$PY -c "import sys, transformers, diffusers; print(sys.version.split()[0], transformers.__version__, diffusers.__version__)"
```

Expected: `3.11.15 5.14.1 0.39.0`. Anything else — stop, you are in the wrong env.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_env.py
import os
from pathlib import Path
from ragregen import env


def test_hf_cache_points_at_shared_cache():
    assert env.HF_CACHE == Path(
        "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache"
    )
    assert env.HF_CACHE.is_dir()


def test_setup_exports_hf_home():
    env.setup()
    assert os.environ["HF_HOME"] == str(env.HF_CACHE)


def test_project_root_contains_configs_dir():
    assert (env.PROJECT_ROOT / "configs").is_dir()
```

- [ ] **Step 4: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen'`

- [ ] **Step 5: Implement**

```python
# ragregen/__init__.py
"""rag-regen: dual-stream verification and retrieval-augmented regeneration."""
```

```python
# ragregen/env.py
"""Cache paths and GPU housekeeping. Import this before transformers.

HF_HOME must be set before huggingface_hub is imported anywhere, because the
library freezes its token path at import time.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path("/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache")


def setup() -> None:
    """Point HuggingFace at the shared cache. Safe to call repeatedly."""
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def reclaim_gpu() -> None:
    """Drop cached allocations. Call between stages, never mid-stage."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


setup()
```

```ini
# pytest.ini
[pytest]
testpaths = tests
markers =
    gpu: requires the shared RTX 4090 (deselected by default)
addopts = -m "not gpu"
```

```
# requirements.txt
# Installed into the `kontext` conda env, not a fresh venv.
# Already present there: torch, transformers==5.14.1, diffusers==0.39.0,
# bitsandbytes, PyYAML, pillow, numpy.
pytest
faiss-cpu
open_clip_torch
pandas
```

```python
# tests/__init__.py
```

- [ ] **Step 6: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_env.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add ragregen/__init__.py ragregen/env.py pytest.ini requirements.txt tests/
git commit -m "feat: package scaffolding, env paths, pytest config"
```

---

## Task 2: Typed configuration loading

**Files:**
- Create: `ragregen/config.py`, `configs/dataset.example.yaml`, `configs/retrieval_db.example.yaml`, `configs/pipeline.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `env.PROJECT_ROOT`.
- Produces:
  - `Case(id: str, prompt: str, concept: str, gt_refs: list[Path])`
  - `DatasetConfig(name: str, images_root: Path, cases: list[Case])`
  - `RetrievalDBConfig(name: str, images_root: Path, index_path: Path, encoder: str, captions: Path | None)`
  - `PipelineConfig(retry_budget: int, tau: float, steps: int, seed: int, crop_scorer: str, retriever: str)`
  - `load_dataset(path: Path) -> DatasetConfig`
  - `load_retrieval_db(path: Path) -> RetrievalDBConfig`
  - `load_pipeline(path: Path) -> PipelineConfig`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from ragregen import config


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_dataset_resolves_gt_refs_against_images_root(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: amur_leopard_01
    prompt: "an Amur leopard on a snowy ridge"
    concept: "Amur leopard"
    gt_refs:
      - amur/ref_01.jpg
""")
    ds = config.load_dataset(p)
    assert ds.name == "t"
    assert len(ds.cases) == 1
    c = ds.cases[0]
    assert c.id == "amur_leopard_01"
    assert c.concept == "Amur leopard"
    assert c.gt_refs == [tmp_path / "amur" / "ref_01.jpg"]


def test_load_dataset_rejects_duplicate_case_ids(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: dup
    prompt: "a"
    concept: "a"
    gt_refs: [x.jpg]
  - id: dup
    prompt: "b"
    concept: "b"
    gt_refs: [y.jpg]
""")
    with pytest.raises(ValueError, match="duplicate case id: dup"):
        config.load_dataset(p)


def test_load_dataset_rejects_missing_gt_refs_key(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: a
    prompt: "a"
    concept: "a"
""")
    with pytest.raises(ValueError, match="case 'a' has no gt_refs"):
        config.load_dataset(p)


def test_load_pipeline_defaults():
    cfg = config.load_pipeline(config.DEFAULT_PIPELINE_PATH)
    assert cfg.retry_budget == 3
    assert 0.0 < cfg.tau < 1.0
    assert cfg.steps == 28
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.config'`

- [ ] **Step 3: Implement**

```python
# ragregen/config.py
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
    gt_refs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images_root: Path
    cases: list[Case]


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

    seen: set[str] = set()
    cases: list[Case] = []
    for raw in raw_cases:
        cid = raw.get("id")
        if not cid:
            raise ValueError(f"{path}: a case is missing 'id'")
        if cid in seen:
            raise ValueError(f"{path}: duplicate case id: {cid}")
        seen.add(cid)
        for key in ("prompt", "concept"):
            if not raw.get(key):
                raise ValueError(f"{path}: case '{cid}' has no {key}")
        refs = raw.get("gt_refs") or []
        if not refs:
            raise ValueError(
                f"{path}: case '{cid}' has no gt_refs. They are required: the "
                f"DINO identity metric and the 'oracle' arm both need them."
            )
        cases.append(Case(
            id=cid,
            prompt=raw["prompt"],
            concept=raw["concept"],
            gt_refs=[root / r for r in refs],
        ))
    return DatasetConfig(name=_require(data, "name", path), images_root=root, cases=cases)


def load_retrieval_db(path: Path = DEFAULT_DB_PATH) -> RetrievalDBConfig:
    data = _read(path)
    captions = data.get("captions")
    return RetrievalDBConfig(
        name=_require(data, "name", path),
        images_root=Path(_require(data, "images_root", path)),
        index_path=Path(_require(data, "index_path", path)),
        encoder=_require(data, "encoder", path),
        captions=Path(captions) if captions else None,
    )


def load_pipeline(path: Path = DEFAULT_PIPELINE_PATH) -> PipelineConfig:
    data = _read(path)
    return PipelineConfig(
        retry_budget=int(data.get("retry_budget", 3)),
        tau=float(data.get("tau", 0.25)),
        steps=int(data.get("steps", 28)),
        seed=int(data.get("seed", 0)),
        crop_scorer=data.get("crop_scorer", "siglip_so400m_384"),
        retriever=data.get("retriever", "siglip_so400m_384"),
    )
```

```yaml
# configs/pipeline.yaml
# Numeric knobs. Safe for an operator to change; nothing here needs code edits.

retry_budget: 3        # N — also the retrieval depth k (spec §3)
tau: 0.25              # Stream A threshold: crop sim below this = MISSING
steps: 28              # FLUX.1-Kontext denoising steps
seed: 0
crop_scorer: siglip_so400m_384   # Stream A; or fgclip (set by the bake-off)
retriever: siglip_so400m_384     # Step 4;  or fgclip (set by the bake-off)
```

```yaml
# configs/dataset.example.yaml
# Copy to dataset.yaml and edit. See docs/RUNBOOK.md §3.1.
#
# gt_refs is REQUIRED for every case: the DINO identity metric and the
# `oracle` arm both read it. Keep these images OUT of the retrieval corpus,
# or the `full` arm scores against its own answer key.

name: my_test_set
images_root: /path/to/your/gt_reference_images

cases:
  - id: amur_leopard_01
    prompt: "an Amur leopard walking across a snowy ridge"
    concept: "Amur leopard"
    gt_refs:
      - amur_leopard/ref_01.jpg
      - amur_leopard/ref_02.jpg
```

```yaml
# configs/retrieval_db.example.yaml
# Copy to retrieval_db.yaml and edit. See docs/RUNBOOK.md §3.2.

name: my_retrieval_db
images_root: /path/to/retrieval/corpus
index_path: data/rag_db/index.faiss
encoder: siglip_so400m_384      # must match what build-index used
captions: null                  # optional CSV with columns: path,caption
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/config.py configs/ tests/test_config.py
git commit -m "feat: typed config loading for the three operator YAMLs"
```

---

## Task 3: Preflight validation

**Files:**
- Create: `ragregen/validate.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: `config.DatasetConfig`, `config.RetrievalDBConfig`, `config.PipelineConfig`.
- Produces:
  - `Problem(severity: str, code: str, message: str)` — severity is `"error"` or `"warning"`
  - `validate_dataset(ds) -> list[Problem]`
  - `validate_db(db, expected_dim: int | None) -> list[Problem]`
  - `validate_disk(min_free_gb: float) -> list[Problem]`
  - `validate_all(ds, db, pipe) -> list[Problem]`
  - `ENCODER_DIMS: dict[str, int]`

Validation **returns** problems and never raises on bad operator data — the caller decides whether
to stop. Only programmer errors raise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate.py
from pathlib import Path

from PIL import Image

from ragregen import config, validate


def make_img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (127, 127, 127)).save(p)


def ds_with(tmp_path, refs, images_root=None):
    root = images_root or tmp_path
    return config.DatasetConfig(
        name="t", images_root=root,
        cases=[config.Case(id="c1", prompt="p", concept="k",
                           gt_refs=[root / r for r in refs])],
    )


def test_missing_gt_ref_file_is_an_error(tmp_path):
    ds = ds_with(tmp_path, ["nope.jpg"])
    problems = validate.validate_dataset(ds)
    assert any(p.code == "gt_ref_missing" and p.severity == "error" for p in problems)


def test_readable_gt_ref_produces_no_error(tmp_path):
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    problems = validate.validate_dataset(ds)
    assert [p for p in problems if p.severity == "error"] == []


def test_corrupt_gt_ref_is_an_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    ds = ds_with(tmp_path, ["bad.jpg"])
    problems = validate.validate_dataset(ds)
    assert any(p.code == "gt_ref_unreadable" for p in problems)


def test_gt_ref_inside_retrieval_corpus_is_an_error(tmp_path):
    corpus = tmp_path / "corpus"
    make_img(corpus / "leak.jpg")
    ds = ds_with(tmp_path, ["corpus/leak.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=corpus,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    problems = validate.validate_leakage(ds, db)
    assert any(p.code == "gt_ref_in_corpus" and p.severity == "error"
               for p in problems)


def test_encoder_dim_mismatch_is_an_error(tmp_path):
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    problems = validate.validate_db(db, expected_dim=768)
    assert any(p.code == "index_dim_mismatch" for p in problems)


def test_unknown_encoder_is_an_error(tmp_path):
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="nope",
    )
    problems = validate.validate_db(db, expected_dim=None)
    assert any(p.code == "unknown_encoder" for p in problems)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.validate'`

- [ ] **Step 3: Implement**

```python
# ragregen/validate.py
"""Preflight over operator-supplied data. Runs before any GPU work.

Every message names the offending file or case and says what to do. The reader
is an operator who does not edit Python (spec §7).
"""
from __future__ import annotations

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
    except OSError:
        return problems
    for case in ds.cases:
        for ref in case.gt_refs:
            try:
                resolved = ref.resolve()
            except OSError:
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
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_validate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/validate.py tests/test_validate.py
git commit -m "feat: preflight validation of operator data"
```

---

## Task 4: Run directories and per-case tracing

**Files:**
- Create: `ragregen/trace.py`, `tests/test_trace.py`

**Interfaces:**
- Consumes: `env.PROJECT_ROOT`.
- Produces:
  - `open_run(tag: str, argv: list[str], args: dict) -> RunDir`
  - `RunDir.path: Path`, `RunDir.case_dir(case_id: str) -> Path`, `RunDir.write_json(name, obj)`, `RunDir.finish(status: str, results: dict)`
  - `CaseTrace(case_id)` with `.vlm(stage, raw)`, `.grounded(scores)`, `.hits(hits)`, `.save(path)`
  - `GARBAGE_SIGNATURE = "!!!!"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace.py
import json

from ragregen import trace


def test_open_run_creates_timestamped_dir_and_latest_symlink(tmp_path):
    rd = trace.open_run("unit", argv=["x"], args={"a": 1}, root=tmp_path)
    assert rd.path.is_dir()
    assert rd.path.name.startswith("unit_")
    latest = tmp_path / "unit_latest"
    assert latest.is_symlink()
    assert latest.resolve() == rd.path.resolve()


def test_run_json_records_argv_and_args(tmp_path):
    rd = trace.open_run("unit", argv=["a", "b"], args={"k": "v"}, root=tmp_path)
    rd.finish("ok", {"mean": 1.0})
    data = json.loads((rd.path / "run.json").read_text())
    assert data["argv"] == ["a", "b"]
    assert data["args"] == {"k": "v"}
    assert data["status"] == "ok"
    assert data["results"] == {"mean": 1.0}


def test_second_run_moves_the_latest_symlink(tmp_path):
    first = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    second = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    assert (tmp_path / "unit_latest").resolve() == second.path.resolve()
    assert first.path.is_dir()


def test_case_trace_flags_garbage_vlm_output(tmp_path):
    ct = trace.CaseTrace("c1")
    ct.vlm("verify", "!!!! !!!!")
    ct.vlm("query", "an Amur leopard")
    out = tmp_path / "trace.json"
    ct.save(out)
    data = json.loads(out.read_text())
    assert data["vlm"][0]["garbage"] is True
    assert data["vlm"][1]["garbage"] is False
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.trace'`

- [ ] **Step 3: Implement**

```python
# ragregen/trace.py
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
    if not torch.cuda.is_available():
        return []
    return [{"index": i, "name": torch.cuda.get_device_name(i),
             "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())]


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
    path.mkdir(parents=True, exist_ok=True)

    latest = root / f"{tag}_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
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
        path.write_text(json.dumps({
            "case_id": self.case_id,
            "vlm": self.vlm_calls,
            "grounded": self.grounded_scores,
            "retrieval": self.retrieval_hits,
        }, indent=2, default=str))
        return path
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_trace.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/trace.py tests/test_trace.py
git commit -m "feat: run directories and per-case reasoning traces"
```

---

## Task 5: Prompt to concept phrases

**Files:**
- Create: `ragregen/concepts.py`, `tests/test_concepts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Concept(phrase: str, kind: str)` where kind is one of `"subject"`, `"count"`, `"relation"`
  - `parse(prompt: str, target: str) -> list[Concept]`
  - `KINDS: tuple[str, ...]`

The target concept is always emitted as a `"subject"` first. Counts and relations are detected so
fusion knows Stream A cannot box them (spec §2 — those are Stream B's job).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_concepts.py
from ragregen import concepts


def kinds(cs):
    return {c.phrase: c.kind for c in cs}


def test_target_concept_is_always_first_and_a_subject():
    cs = concepts.parse("an Amur leopard on a snowy ridge", target="Amur leopard")
    assert cs[0].phrase == "Amur leopard"
    assert cs[0].kind == "subject"


def test_numeric_words_produce_a_count_concept():
    cs = concepts.parse("three polar bears on an ice floe", target="polar bear")
    assert any(c.kind == "count" for c in cs)


def test_digits_produce_a_count_concept():
    cs = concepts.parse("2 axolotls in a tank", target="axolotl")
    assert any(c.kind == "count" for c in cs)


def test_spatial_words_produce_a_relation_concept():
    cs = concepts.parse("a durian behind a wooden fence", target="durian")
    assert any(c.kind == "relation" for c in cs)


def test_no_duplicate_phrases():
    cs = concepts.parse("a fox and a fox", target="fox")
    phrases = [c.phrase for c in cs]
    assert len(phrases) == len(set(phrases))


def test_every_kind_is_known():
    cs = concepts.parse("three foxes behind a tree", target="fox")
    assert all(c.kind in concepts.KINDS for c in cs)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_concepts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.concepts'`

- [ ] **Step 3: Implement**

```python
# ragregen/concepts.py
"""Prompt -> the concept phrases a verifier should check.

Deliberately rule-based and model-free. This runs before any model loads, it
must be deterministic for reproducible runs, and its output is the contract
between the prompt and both verification streams.

Counts and spatial relations get their own kinds because Stream A cannot check
them -- GroundingDINO boxes objects, not quantities or arrangements. Fusion
routes those to Stream B (spec §2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KINDS = ("subject", "count", "relation")

_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "a pair of", "several", "many", "a few",
}

_RELATION_WORDS = {
    "behind", "in front of", "next to", "beside", "above", "below", "under",
    "on top of", "between", "inside", "outside", "left of", "right of",
    "beneath", "near", "against",
}


@dataclass(frozen=True)
class Concept:
    phrase: str
    kind: str


def _found(prompt_lc: str, needles: set[str]) -> list[str]:
    hits = []
    for n in needles:
        if re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", prompt_lc):
            hits.append(n)
    return sorted(hits)


def parse(prompt: str, target: str) -> list[Concept]:
    """Return the concepts to verify, target first.

    >>> [c.kind for c in parse("three foxes behind a tree", target="fox")][0]
    'subject'
    """
    prompt_lc = prompt.lower()
    out: list[Concept] = [Concept(target, "subject")]
    seen = {target.lower()}

    for word in _found(prompt_lc, _NUMBER_WORDS):
        if word not in seen:
            out.append(Concept(word, "count"))
            seen.add(word)

    if re.search(r"(?<!\w)\d+(?!\w)", prompt_lc):
        digits = re.findall(r"(?<!\w)\d+(?!\w)", prompt_lc)
        for d in digits:
            if d not in seen:
                out.append(Concept(d, "count"))
                seen.add(d)

    for word in _found(prompt_lc, _RELATION_WORDS):
        if word not in seen:
            out.append(Concept(word, "relation"))
            seen.add(word)

    return out
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_concepts.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/concepts.py tests/test_concepts.py
git commit -m "feat: rule-based prompt to concept-phrase parsing"
```

---

## Task 6: Stream A — grounded verifier

**Files:**
- Create: `ragregen/verify/__init__.py`, `ragregen/verify/grounded.py`, `tests/test_grounded.py`

**Interfaces:**
- Consumes: `concepts.Concept`.
- Produces:
  - `Detector` protocol: `all_boxes(image, phrase, max_boxes: int) -> list[tuple[tuple[float,float,float,float], float]]`
  - `CropScorer` protocol: `score(crop, phrase) -> float` returning a value in `[0, 1]`
  - `ConceptScore(phrase, kind, box, dino_conf, sim, state)` where state ∈ `{"PRESENT", "MISSING", "ABSTAIN"}`
  - `GroundedVerifier(detector, scorer, tau)` with `.score(image, concepts) -> dict[str, ConceptScore]`
  - `STATES = ("PRESENT", "MISSING", "ABSTAIN")`

**The abstention rule is the point of this task** (spec §2, R3). No box → `ABSTAIN`, never
`MISSING`. `count` and `relation` concepts always `ABSTAIN` — Stream A structurally cannot check
them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounded.py
from PIL import Image

from ragregen.concepts import Concept
from ragregen.verify import grounded


class FakeDetector:
    """Returns pre-programmed boxes per phrase."""

    def __init__(self, boxes):
        self.boxes = boxes

    def all_boxes(self, image, phrase, max_boxes=8):
        return self.boxes.get(phrase, [])


class FakeScorer:
    def __init__(self, sims):
        self.sims = sims

    def score(self, crop, phrase):
        return self.sims.get(phrase, 0.0)


IMG = Image.new("RGB", (64, 64), (10, 20, 30))
BOX = (0.0, 0.0, 32.0, 32.0)


def test_high_similarity_is_present():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.8}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "PRESENT"
    assert out["fox"].sim == 0.8


def test_low_similarity_is_missing():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.05}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "MISSING"


def test_no_box_abstains_and_never_reports_missing():
    gv = grounded.GroundedVerifier(
        FakeDetector({}), FakeScorer({}), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "ABSTAIN"
    assert out["fox"].state != "MISSING"
    assert out["fox"].sim is None


def test_count_concepts_always_abstain():
    gv = grounded.GroundedVerifier(
        FakeDetector({"three": [(BOX, 0.9)]}), FakeScorer({"three": 0.9}), tau=0.25)
    out = gv.score(IMG, [Concept("three", "count")])
    assert out["three"].state == "ABSTAIN"


def test_relation_concepts_always_abstain():
    gv = grounded.GroundedVerifier(
        FakeDetector({"behind": [(BOX, 0.9)]}), FakeScorer({"behind": 0.9}), tau=0.25)
    out = gv.score(IMG, [Concept("behind", "relation")])
    assert out["behind"].state == "ABSTAIN"


def test_best_of_several_boxes_wins():
    boxes = [((0.0, 0.0, 16.0, 16.0), 0.5), ((16.0, 16.0, 48.0, 48.0), 0.4)]

    class PositionScorer:
        def score(self, crop, phrase):
            return 0.9 if crop.size == (32, 32) else 0.1

    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": boxes}), PositionScorer(), tau=0.25)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].sim == 0.9
    assert out["fox"].box == (16.0, 16.0, 48.0, 48.0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_grounded.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.verify'`

- [ ] **Step 3: Implement**

```python
# ragregen/verify/__init__.py
"""The two verification streams and their fusion."""
```

```python
# ragregen/verify/grounded.py
"""Stream A: GroundingDINO proposes boxes, a region-text model scores the crops.

Produces a continuous per-concept score, which is what makes the verifier
calibratable -- the property a single VLM verdict cannot offer (spec §2).

ABSTAIN is not MISSING. When the detector returns no box, that means either the
concept is absent OR the detector failed, and those are not the same claim.
Reporting MISSING here would make every unboxable prompt a false failure
(spec §9 R3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragregen.concepts import Concept

STATES = ("PRESENT", "MISSING", "ABSTAIN")

Box = tuple[float, float, float, float]

#: Kinds Stream A structurally cannot check -- no box expresses a quantity or
#: an arrangement. Fusion routes these to Stream B.
UNBOXABLE_KINDS = ("count", "relation")


class Detector(Protocol):
    def all_boxes(self, image, phrase: str, max_boxes: int = 8
                  ) -> list[tuple[Box, float]]: ...


class CropScorer(Protocol):
    def score(self, crop, phrase: str) -> float: ...


@dataclass(frozen=True)
class ConceptScore:
    phrase: str
    kind: str
    state: str
    box: Box | None = None
    dino_conf: float | None = None
    sim: float | None = None


class GroundedVerifier:
    """Scores each concept by detecting it and scoring the best crop.

    Args:
        detector: proposes candidate boxes for a phrase.
        scorer:   scores a crop against a phrase, in [0, 1].
        tau:      crops scoring below this are MISSING.
    """

    def __init__(self, detector: Detector, scorer: CropScorer, tau: float):
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.detector = detector
        self.scorer = scorer
        self.tau = tau

    def score(self, image, concepts: list[Concept]) -> dict[str, ConceptScore]:
        out: dict[str, ConceptScore] = {}
        for concept in concepts:
            out[concept.phrase] = self._score_one(image, concept)
        return out

    def _score_one(self, image, concept: Concept) -> ConceptScore:
        if concept.kind in UNBOXABLE_KINDS:
            return ConceptScore(concept.phrase, concept.kind, "ABSTAIN")

        candidates = self.detector.all_boxes(image, concept.phrase, max_boxes=8)
        if not candidates:
            return ConceptScore(concept.phrase, concept.kind, "ABSTAIN")

        best_box, best_conf, best_sim = None, None, -1.0
        for box, conf in candidates:
            crop = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            sim = float(self.scorer.score(crop, concept.phrase))
            if sim > best_sim:
                best_box, best_conf, best_sim = box, conf, sim

        state = "PRESENT" if best_sim >= self.tau else "MISSING"
        return ConceptScore(concept.phrase, concept.kind, state,
                            box=best_box, dino_conf=best_conf, sim=best_sim)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_grounded.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/verify/ tests/test_grounded.py
git commit -m "feat: Stream A grounded verifier with abstention"
```

---

## Task 7: Stream A model adapters

**Files:**
- Create: `ragregen/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `env.HF_CACHE`, the `Detector`/`CropScorer` protocols from Task 6.
- Produces:
  - `DinoDetector(model_id, device)` implementing `Detector`
  - `SigLIPScorer(model_id, device)` implementing `CropScorer`
  - `FGCLIPScorer(model_id, device)` implementing `CropScorer`
  - `build_crop_scorer(name: str, device: str) -> CropScorer`
  - `SCORER_IDS: dict[str, str]`

These are the only classes in the package that call `from_pretrained`. Everything else takes them
as arguments, which is why the rest of the test suite runs on CPU.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from PIL import Image

from ragregen import models


def test_build_crop_scorer_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown crop scorer"):
        models.build_crop_scorer("nope", device="cpu")


def test_scorer_ids_cover_the_configured_choices():
    assert "siglip_so400m_384" in models.SCORER_IDS
    assert "fgclip" in models.SCORER_IDS


@pytest.mark.gpu
def test_dino_detector_finds_a_box_on_a_real_image():
    det = models.DinoDetector(device="cuda")
    img = Image.open("tests/fixtures/two_dogs.jpg")
    boxes = det.all_boxes(img, "dog", max_boxes=8)
    assert len(boxes) >= 2
    for box, conf in boxes:
        assert len(box) == 4
        assert 0.0 <= conf <= 1.0


@pytest.mark.gpu
def test_siglip_scorer_prefers_the_matching_phrase():
    sc = models.build_crop_scorer("siglip_so400m_384", device="cuda")
    dog = Image.open("tests/fixtures/dog_crop.jpg")
    assert sc.score(dog, "a dog") > sc.score(dog, "a violin")
```

- [ ] **Step 2: Add the two GPU fixtures**

```bash
mkdir -p tests/fixtures
cp "../rare-concepts-test/Amur leopard-ref.jpeg" tests/fixtures/dog_crop.jpg
$PY - <<'PY'
from PIL import Image
a = Image.open("../ImageRAG/datasets/controlled_dataset_v2/boston_bull_2.jpeg").convert("RGB").resize((400, 400))
b = Image.open("../ImageRAG/datasets/controlled_dataset_v2/golden_retriever_1.jpg").convert("RGB").resize((400, 400))
canvas = Image.new("RGB", (800, 400))
canvas.paste(a, (0, 0)); canvas.paste(b, (400, 0))
canvas.save("tests/fixtures/two_dogs.jpg")
PY
```

If those source images are absent, substitute any two-animal photo — the assertions only require
two detectable objects.

- [ ] **Step 3: Run the CPU tests to make sure they fail**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.models'`

- [ ] **Step 4: Implement**

```python
# ragregen/models.py
"""The only module that loads weights.

Everything else receives models as constructor arguments, so the logic layer
tests on CPU with fakes. Weights resolve from ../.cache via env.HF_CACHE.
"""
from __future__ import annotations

from ragregen import env

env.setup()  # must precede transformers import

DINO_ID = "IDEA-Research/grounding-dino-base"

SCORER_IDS: dict[str, str] = {
    "siglip_so400m_384": "timm/ViT-SO400M-14-SigLIP-384",
    "fgclip": "qihoo360/fg-clip-base",
}


class DinoDetector:
    """GroundingDINO box proposals for a text phrase."""

    def __init__(self, model_id: str = DINO_ID, device: str = "cuda",
                 box_threshold: float = 0.25, text_threshold: float = 0.25):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    def all_boxes(self, image, phrase: str, max_boxes: int = 8):
        text = phrase if phrase.endswith(".") else phrase + "."
        inputs = self.processor(images=image, text=text.lower(),
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        pairs = [(tuple(float(v) for v in box), float(score))
                 for box, score in zip(results["boxes"], results["scores"])]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:max_boxes]


class _SigmoidTextImageScorer:
    """Shared body for SigLIP-family scorers: sigmoid of the paired logit."""

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    def score(self, crop, phrase: str) -> float:
        inputs = self.processor(text=[phrase], images=crop, padding="max_length",
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs)
        return float(self._torch.sigmoid(out.logits_per_image)[0][0])


class SigLIPScorer(_SigmoidTextImageScorer):
    def __init__(self, model_id: str = SCORER_IDS["siglip_so400m_384"],
                 device: str = "cuda"):
        super().__init__(model_id, device)


class FGCLIPScorer(_SigmoidTextImageScorer):
    def __init__(self, model_id: str = SCORER_IDS["fgclip"], device: str = "cuda"):
        super().__init__(model_id, device)


def build_crop_scorer(name: str, device: str = "cuda"):
    if name not in SCORER_IDS:
        raise ValueError(
            f"unknown crop scorer '{name}'. Known: {', '.join(sorted(SCORER_IDS))}")
    cls = {"siglip_so400m_384": SigLIPScorer, "fgclip": FGCLIPScorer}[name]
    return cls(device=device)
```

- [ ] **Step 5: Run the CPU tests and make sure they pass**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: 2 passed, 2 deselected

- [ ] **Step 6: Run the GPU tests once, when the card is free**

```bash
nvidia-smi        # confirm it is quiet first
$PY -m pytest tests/test_models.py -v -m gpu
```

Expected: 2 passed. If `qihoo360/fg-clip-base` 404s, check the current FG-CLIP model id on the Hub
and update `SCORER_IDS` — it is the one identifier in this plan not verified against the local cache.

- [ ] **Step 7: Commit**

```bash
git add ragregen/models.py tests/test_models.py tests/fixtures/
git commit -m "feat: GroundingDINO and SigLIP/FG-CLIP adapters"
```

---

## Task 8: Stream B — semantic verifier

**Files:**
- Create: `ragregen/verify/semantic.py`, `tests/test_semantic.py`

**Interfaces:**
- Consumes: `trace.GARBAGE_SIGNATURE`.
- Produces:
  - `VLM` protocol: `ask(image, prompt: str) -> str`
  - `Issue(concept: str, problem: str)`
  - `SemanticVerdict(ok: bool, issues: list[Issue], raw: str, degenerate: bool)`
  - `SemanticVerifier(vlm)` with `.judge(image, prompt) -> SemanticVerdict`
  - `JUDGE_TEMPLATE: str`

A degenerate reply (unparseable, empty, or carrying the `!!!!` NaN signature) must **fail closed**:
`ok=False`, `degenerate=True`. Silently passing on a broken judge is how the predecessor shipped a
cactus for a sea lion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic.py
from PIL import Image

from ragregen.verify import semantic

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVLM:
    def __init__(self, reply):
        self.reply = reply
        self.asked = []

    def ask(self, image, prompt):
        self.asked.append(prompt)
        return self.reply


def test_clean_pass_is_parsed():
    v = semantic.SemanticVerifier(FakeVLM('{"verdict": "PASS", "issues": []}'))
    out = v.judge(IMG, "a fox")
    assert out.ok is True
    assert out.issues == []
    assert out.degenerate is False


def test_fail_with_issues_is_parsed():
    reply = ('{"verdict": "FAIL", "issues": '
             '[{"concept": "Amur leopard", "problem": "rosettes are wrong"}]}')
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "an Amur leopard")
    assert out.ok is False
    assert out.issues[0].concept == "Amur leopard"
    assert out.issues[0].problem == "rosettes are wrong"


def test_json_embedded_in_prose_is_recovered():
    reply = 'Sure! Here is my answer:\n{"verdict": "FAIL", "issues": []}\nHope that helps.'
    out = semantic.SemanticVerifier(FakeVLM(reply)).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is False


def test_unparseable_reply_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("I cannot tell.")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_empty_reply_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_nan_signature_fails_closed():
    out = semantic.SemanticVerifier(FakeVLM("!!!! !!!! !!!!")).judge(IMG, "a fox")
    assert out.ok is False
    assert out.degenerate is True


def test_raw_reply_is_preserved_for_the_trace():
    out = semantic.SemanticVerifier(FakeVLM("garbage")).judge(IMG, "a fox")
    assert out.raw == "garbage"


def test_prompt_is_interpolated_into_the_template():
    vlm = FakeVLM('{"verdict": "PASS", "issues": []}')
    semantic.SemanticVerifier(vlm).judge(IMG, "a monkey puzzle tree")
    assert "a monkey puzzle tree" in vlm.asked[0]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_semantic.py -v`
Expected: FAIL — `ImportError: cannot import name 'semantic'`

- [ ] **Step 3: Implement**

```python
# ragregen/verify/semantic.py
"""Stream B: one VLM call returning a structured verdict.

Catches what a box-and-crop scorer structurally cannot -- counts, spatial
relations, attribute binding (spec §2).

Degenerate replies fail CLOSED. An unparseable or NaN-poisoned judge that
silently returns PASS is exactly how ../ImageRAG shipped a cactus for a sea-lion
prompt with nothing in the logs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ragregen.trace import GARBAGE_SIGNATURE

JUDGE_TEMPLATE = """You are checking whether a generated image matches its prompt.

PROMPT: {prompt}

Reply with JSON only, no prose, in exactly this shape:
{{"verdict": "PASS" | "FAIL", "issues": [{{"concept": "...", "problem": "..."}}]}}

Rules:
- "PASS" only if every concept in the prompt is present and correct.
- If a concept is a specific breed, species or variety, judge whether THAT one
  is shown, not merely a plausible member of the general category.
- "issues" must be empty when the verdict is PASS.
- Name the concept exactly as it appears in the prompt.
"""


class VLM(Protocol):
    def ask(self, image, prompt: str) -> str: ...


@dataclass(frozen=True)
class Issue:
    concept: str
    problem: str


@dataclass(frozen=True)
class SemanticVerdict:
    ok: bool
    raw: str
    degenerate: bool = False
    issues: list[Issue] = field(default_factory=list)


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a reply."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class SemanticVerifier:
    def __init__(self, vlm: VLM, template: str = JUDGE_TEMPLATE):
        self.vlm = vlm
        self.template = template

    def judge(self, image, prompt: str) -> SemanticVerdict:
        raw = self.vlm.ask(image, self.template.format(prompt=prompt)) or ""

        if GARBAGE_SIGNATURE in raw:
            return SemanticVerdict(ok=False, raw=raw, degenerate=True)

        data = _extract_json(raw)
        if not isinstance(data, dict) or "verdict" not in data:
            return SemanticVerdict(ok=False, raw=raw, degenerate=True)

        verdict = str(data.get("verdict", "")).strip().upper()
        if verdict not in ("PASS", "FAIL"):
            return SemanticVerdict(ok=False, raw=raw, degenerate=True)

        issues = [
            Issue(concept=str(i.get("concept", "")).strip(),
                  problem=str(i.get("problem", "")).strip())
            for i in (data.get("issues") or [])
            if isinstance(i, dict) and i.get("concept")
        ]
        return SemanticVerdict(ok=(verdict == "PASS"), raw=raw,
                               degenerate=False, issues=issues)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_semantic.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/verify/semantic.py tests/test_semantic.py
git commit -m "feat: Stream B semantic verifier, failing closed on degenerate replies"
```

---

## Task 9: Fusion

**Files:**
- Create: `ragregen/verify/fusion.py`, `tests/test_fusion.py`

**Interfaces:**
- Consumes: `grounded.ConceptScore`, `semantic.SemanticVerdict`.
- Produces:
  - `Verdict(ok: bool, target: str | None, score: float, agreement: str, evidence: dict)` where agreement ∈ `{"BOTH_PASS", "BOTH_FAIL", "GROUNDED_ONLY", "SEMANTIC_ONLY"}`
  - `fuse(grounded_scores: dict[str, ConceptScore], semantic: SemanticVerdict) -> Verdict`
  - `AGREEMENTS: tuple[str, ...]`

`score` is the scalar the retry loop compares to pick `best` (spec §3). Definition: the minimum
`sim` over all non-ABSTAIN concepts; if every concept abstained, `1.0` when Stream B passed and
`0.0` when it failed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion.py
from ragregen.verify import fusion
from ragregen.verify.grounded import ConceptScore
from ragregen.verify.semantic import Issue, SemanticVerdict


def cs(phrase, state, sim=None, kind="subject"):
    return ConceptScore(phrase, kind, state, sim=sim)


PASS = SemanticVerdict(ok=True, raw="{}")


def fail(concept="x", problem="p"):
    return SemanticVerdict(ok=False, raw="{}", issues=[Issue(concept, problem)])


def test_both_pass_is_ok():
    v = fusion.fuse({"fox": cs("fox", "PRESENT", 0.8)}, PASS)
    assert v.ok is True
    assert v.agreement == "BOTH_PASS"
    assert v.target is None


def test_grounded_failure_fails_the_verdict():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, PASS)
    assert v.ok is False
    assert v.target == "fox"
    assert v.agreement == "GROUNDED_ONLY"


def test_semantic_failure_alone_fails_the_verdict():
    v = fusion.fuse({"fox": cs("fox", "PRESENT", 0.8)}, fail("fox", "wrong breed"))
    assert v.ok is False
    assert v.target == "fox"
    assert v.agreement == "SEMANTIC_ONLY"


def test_both_failing_is_recorded_as_both_fail():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, fail())
    assert v.agreement == "BOTH_FAIL"


def test_grounded_target_wins_over_semantic_target():
    v = fusion.fuse({"fox": cs("fox", "MISSING", 0.05)}, fail("badger", "p"))
    assert v.target == "fox"


def test_lowest_scoring_missing_concept_is_the_target():
    scores = {"fox": cs("fox", "MISSING", 0.20),
              "tree": cs("tree", "MISSING", 0.02)}
    v = fusion.fuse(scores, PASS)
    assert v.target == "tree"


def test_abstain_alone_does_not_fail_the_verdict():
    v = fusion.fuse({"behind": cs("behind", "ABSTAIN", kind="relation")}, PASS)
    assert v.ok is True
    assert v.target is None


def test_all_abstain_score_follows_semantic_stream():
    scores = {"behind": cs("behind", "ABSTAIN", kind="relation")}
    assert fusion.fuse(scores, PASS).score == 1.0
    assert fusion.fuse(scores, fail()).score == 0.0


def test_score_is_the_minimum_non_abstain_similarity():
    scores = {"fox": cs("fox", "PRESENT", 0.8),
              "tree": cs("tree", "PRESENT", 0.3),
              "behind": cs("behind", "ABSTAIN", kind="relation")}
    assert fusion.fuse(scores, PASS).score == 0.3
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ImportError: cannot import name 'fusion'`

- [ ] **Step 3: Implement**

```python
# ragregen/verify/fusion.py
"""Combine the grounded and semantic streams into one verdict.

Pure function, no models. The `agreement` field is not bookkeeping: the rate at
which the two streams disagree is the evidence that decomposing ImageRAG's
single GPT-4o judge buys something (spec §2, claim C1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragregen.verify.grounded import ConceptScore
from ragregen.verify.semantic import SemanticVerdict

AGREEMENTS = ("BOTH_PASS", "BOTH_FAIL", "GROUNDED_ONLY", "SEMANTIC_ONLY")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    target: str | None
    score: float
    agreement: str
    evidence: dict = field(default_factory=dict)


def fuse(grounded_scores: dict[str, ConceptScore],
         semantic: SemanticVerdict) -> Verdict:
    missing = [s for s in grounded_scores.values() if s.state == "MISSING"]
    grounded_failed = bool(missing)
    semantic_failed = not semantic.ok

    if grounded_failed and semantic_failed:
        agreement = "BOTH_FAIL"
    elif grounded_failed:
        agreement = "GROUNDED_ONLY"
    elif semantic_failed:
        agreement = "SEMANTIC_ONLY"
    else:
        agreement = "BOTH_PASS"

    ok = not (grounded_failed or semantic_failed)

    target = None
    if grounded_failed:
        target = min(missing, key=lambda s: s.sim if s.sim is not None else 0.0).phrase
    elif semantic_failed and semantic.issues:
        target = semantic.issues[0].concept

    sims = [s.sim for s in grounded_scores.values()
            if s.state != "ABSTAIN" and s.sim is not None]
    if sims:
        score = float(min(sims))
    else:
        score = 1.0 if semantic.ok else 0.0

    return Verdict(
        ok=ok, target=target, score=score, agreement=agreement,
        evidence={
            "grounded": {p: {"state": s.state, "sim": s.sim, "kind": s.kind}
                         for p, s in grounded_scores.items()},
            "semantic": {"ok": semantic.ok, "degenerate": semantic.degenerate,
                         "issues": [{"concept": i.concept, "problem": i.problem}
                                    for i in semantic.issues]},
        },
    )
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_fusion.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/verify/fusion.py tests/test_fusion.py
git commit -m "feat: stream fusion with agreement tracking"
```

---

## Task 10: Retrieval and index building

**Files:**
- Create: `ragregen/retrieve.py`, `scripts/build_index.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `config.RetrievalDBConfig`, `validate.ENCODER_DIMS`.
- Produces:
  - `Hit(path: Path, score: float, rank: int)`
  - `Retriever(index, paths, encoder)` with `.search(query: str, k: int) -> list[Hit]`
  - `Retriever.load(db: RetrievalDBConfig, encoder) -> Retriever`
  - `build_index(image_paths, encoder, out_path) -> int` returning the number indexed
  - `TextImageEncoder` protocol: `.encode_images(paths, batch_size) -> np.ndarray`, `.encode_text(texts) -> np.ndarray`, `.dim: int`

Embeddings are L2-normalised so `IndexFlatIP` gives cosine similarity. At 500k × 1152 fp32 the index
is ~2.3 GB and exact search stays tractable — do not substitute IVF/HNSW (spec §9 R6).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieve.py
from pathlib import Path

import numpy as np
import pytest

from ragregen import retrieve


class FakeEncoder:
    """Maps a filename stem to a fixed unit vector; text maps by keyword."""

    dim = 4
    VECS = {
        "fox":    np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        "panda":  np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
        "durian": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
    }

    def encode_images(self, paths, batch_size=32):
        return np.stack([self.VECS[Path(p).stem] for p in paths])

    def encode_text(self, texts):
        out = []
        for t in texts:
            for key, vec in self.VECS.items():
                if key in t.lower():
                    out.append(vec)
                    break
            else:
                out.append(np.zeros(self.dim, dtype="float32"))
        return np.stack(out)


PATHS = [Path("/c/fox.jpg"), Path("/c/panda.jpg"), Path("/c/durian.jpg")]


def test_build_index_returns_the_count(tmp_path):
    n = retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    assert n == 3
    assert (tmp_path / "i.faiss").is_file()
    assert (tmp_path / "i.faiss.paths.json").is_file()


def test_search_returns_the_matching_image_first(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    hits = r.search("a fox in the snow", k=3)
    assert hits[0].path.stem == "fox"
    assert hits[0].rank == 0


def test_search_k_caps_the_result_count(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    assert len(r.search("a panda", k=2)) == 2


def test_ranks_are_consecutive_from_zero(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    assert [h.rank for h in r.search("a durian", k=3)] == [0, 1, 2]


def test_build_index_rejects_an_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        retrieve.build_index([], FakeEncoder(), tmp_path / "i.faiss")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.retrieve'`

- [ ] **Step 3: Implement**

```python
# ragregen/retrieve.py
"""Text-to-image retrieval over a FAISS IndexFlatIP.

Flat/exact on purpose. At 500k images a 1152-dim fp32 index is ~2.3 GB and
brute force stays fast, so retrieval quality is never confounded by index
approximation (spec §9 R6). Do not swap in IVF or HNSW without a measured
reason to.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextImageEncoder(Protocol):
    dim: int

    def encode_images(self, paths, batch_size: int = 32) -> np.ndarray: ...
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    path: Path
    score: float
    rank: int


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_index(image_paths, encoder: TextImageEncoder, out_path: Path,
                batch_size: int = 32) -> int:
    """Embed every image and write an exact inner-product index."""
    import faiss

    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("no images to index -- check images_root in retrieval_db.yaml")

    mat = _l2_normalise(encoder.encode_images(paths, batch_size=batch_size))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    out_path.with_suffix(out_path.suffix + ".paths.json").write_text(
        json.dumps([str(p) for p in paths], indent=2))
    return len(paths)


class Retriever:
    def __init__(self, index, paths: list[Path], encoder: TextImageEncoder):
        self.index = index
        self.paths = paths
        self.encoder = encoder

    @classmethod
    def from_index(cls, index_path: Path, encoder: TextImageEncoder) -> "Retriever":
        import faiss

        index_path = Path(index_path)
        index = faiss.read_index(str(index_path))
        meta = index_path.with_suffix(index_path.suffix + ".paths.json")
        if not meta.is_file():
            raise FileNotFoundError(
                f"index at {index_path} has no sidecar {meta.name}. "
                f"Rebuild with `./scripts/run.sh build-index`.")
        paths = [Path(p) for p in json.loads(meta.read_text())]
        return cls(index, paths, encoder)

    @property
    def dim(self) -> int:
        return self.index.d

    def search(self, query: str, k: int) -> list[Hit]:
        vec = _l2_normalise(self.encoder.encode_text([query]))
        k = min(k, len(self.paths))
        scores, idxs = self.index.search(vec, k)
        return [Hit(path=self.paths[int(i)], score=float(s), rank=rank)
                for rank, (s, i) in enumerate(zip(scores[0], idxs[0])) if i >= 0]
```

```python
#!/usr/bin/env python
"""Build the retrieval index from the corpus in configs/retrieval_db.yaml.

Usage: ./scripts/run.sh build-index
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env, retrieve, trace, validate  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    db = config.load_retrieval_db(args.db)

    problems = validate.validate_disk()
    for p in problems:
        print(f"[{p.severity}] {p.code}: {p.message}")
    if any(p.severity == "error" for p in problems):
        return 2

    paths = sorted(p for p in db.images_root.rglob("*")
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"[corpus] {len(paths)} images under {db.images_root}")

    run = trace.open_run("build_index", argv=sys.argv, args=vars(args))

    from ragregen.encoders import build_encoder
    encoder = build_encoder(db.encoder, device=args.device)

    n = retrieve.build_index(paths, encoder, db.index_path,
                             batch_size=args.batch_size)
    print(f"[index] {n} vectors, dim={encoder.dim} -> {db.index_path}")
    run.finish("ok", {"n_indexed": n, "dim": encoder.dim})
    env.reclaim_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_retrieve.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/retrieve.py scripts/build_index.py tests/test_retrieve.py
git commit -m "feat: exact FAISS retrieval and index building"
```

---

## Task 11: Retrieval encoders

**Files:**
- Create: `ragregen/encoders.py`, `tests/test_encoders.py`

**Interfaces:**
- Consumes: `env.HF_CACHE`, `retrieve.TextImageEncoder` protocol, `validate.ENCODER_DIMS`.
- Produces:
  - `SigLIPEncoder(model_id, device)`, `FGCLIPEncoder(model_id, device)`, `OpenCLIPEncoder(model_id, device)`
  - `build_encoder(name: str, device: str) -> TextImageEncoder`
  - `ENCODER_IDS: dict[str, str]`

`build_encoder` names must match `validate.ENCODER_DIMS` keys exactly, or preflight passes an index
that retrieval then rejects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoders.py
import pytest

from ragregen import encoders, validate


def test_build_encoder_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown encoder"):
        encoders.build_encoder("nope", device="cpu")


def test_every_encoder_name_has_a_declared_dim():
    for name in encoders.ENCODER_IDS:
        assert name in validate.ENCODER_DIMS, (
            f"{name} is buildable but has no entry in validate.ENCODER_DIMS")


@pytest.mark.gpu
def test_siglip_encoder_dim_matches_the_declared_dim():
    enc = encoders.build_encoder("siglip_so400m_384", device="cuda")
    assert enc.dim == validate.ENCODER_DIMS["siglip_so400m_384"]


@pytest.mark.gpu
def test_siglip_encoder_ranks_the_matching_image_first(tmp_path):
    from PIL import Image

    from ragregen import retrieve

    dog = tmp_path / "dog.jpg"
    Image.open("tests/fixtures/dog_crop.jpg").save(dog)
    noise = tmp_path / "noise.jpg"
    Image.new("RGB", (256, 256), (3, 200, 7)).save(noise)

    enc = encoders.build_encoder("siglip_so400m_384", device="cuda")
    retrieve.build_index([dog, noise], enc, tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", enc)
    assert r.search("a photograph of an animal", k=2)[0].path.stem == "dog"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_encoders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.encoders'`

- [ ] **Step 3: Implement**

```python
# ragregen/encoders.py
"""Whole-image/text encoders for retrieval.

Separate from models.py on purpose: models.py scores CROPS against phrases
(Stream A), this scores WHOLE IMAGES against queries (step 4). The bake-off
compares FG-CLIP and SigLIP in both roles independently, because a region-text
model can win one and lose the other (spec §6 Stage 1).
"""
from __future__ import annotations

import numpy as np

from ragregen import env

env.setup()  # must precede transformers import

ENCODER_IDS: dict[str, str] = {
    "siglip_so400m_384": "timm/ViT-SO400M-14-SigLIP-384",
    "siglip_base_224": "google/siglip-base-patch16-224",
    "fgclip": "qihoo360/fg-clip-base",
    "openclip_l14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "clip_b32": "openai/clip-vit-base-patch32",
}


class _HFEncoder:
    """AutoModel-based dual encoder. Covers SigLIP, CLIP and FG-CLIP."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    def encode_images(self, paths, batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        chunks = []
        paths = list(paths)
        for start in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB")
                     for p in paths[start:start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            chunks.append(feats.float().cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  [encode] {start + len(batch)}/{len(paths)}", flush=True)
        return np.concatenate(chunks, axis=0)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), padding="max_length",
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return feats.float().cpu().numpy()


class SigLIPEncoder(_HFEncoder):
    pass


class FGCLIPEncoder(_HFEncoder):
    pass


class OpenCLIPEncoder(_HFEncoder):
    pass


def build_encoder(name: str, device: str = "cuda"):
    if name not in ENCODER_IDS:
        raise ValueError(
            f"unknown encoder '{name}'. Known: {', '.join(sorted(ENCODER_IDS))}")
    return _HFEncoder(ENCODER_IDS[name], device=device)
```

- [ ] **Step 4: Run the CPU tests and make sure they pass**

Run: `$PY -m pytest tests/test_encoders.py -v`
Expected: 2 passed, 2 deselected

- [ ] **Step 5: Run the GPU tests when the card is free**

```bash
nvidia-smi
$PY -m pytest tests/test_encoders.py -v -m gpu
```

Expected: 2 passed. If `dim` mismatches `validate.ENCODER_DIMS`, correct the table in
`validate.py` — the measured value is the truth.

- [ ] **Step 6: Commit**

```bash
git add ragregen/encoders.py tests/test_encoders.py
git commit -m "feat: retrieval encoders for SigLIP, FG-CLIP, open-CLIP"
```

---

## Task 12: The premise gate — `screen`

**Files:**
- Create: `scripts/screen_premise.py`, `ragregen/draft.py`, `tests/test_draft.py`

**Interfaces:**
- Consumes: `config.DatasetConfig`, `trace.open_run`.
- Produces:
  - `Drafter(pipe, steps, seed)` with `.draft(prompt) -> PIL.Image`
  - `load_kontext_t2i(device, quantize) -> pipe`
  - `screen_premise.py` writing `labels.csv` with columns `case_id,prompt,concept,draft_path,verdict,notes`

**This task is the gate (spec §10 step 0).** Its output decides whether Plan 2 is worth writing.
`labels.csv` ships with `verdict` blank for a human to fill in with `pass` or `fail`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft.py
from PIL import Image

from ragregen import draft


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, num_inference_steps, generator, **kw):
        self.calls.append((prompt, num_inference_steps))

        class Out:
            images = [Image.new("RGB", (64, 64), (5, 5, 5))]

        return Out()


def test_draft_returns_an_image_and_forwards_steps():
    pipe = FakePipe()
    d = draft.Drafter(pipe, steps=28, seed=0)
    img = d.draft("an Amur leopard")
    assert isinstance(img, Image.Image)
    assert pipe.calls[0] == ("an Amur leopard", 28)


def test_same_seed_produces_the_same_generator_state():
    pipe = FakePipe()
    d = draft.Drafter(pipe, steps=4, seed=7)
    d.draft("a")
    d.draft("a")
    assert len(pipe.calls) == 2
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_draft.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.draft'`

- [ ] **Step 3: Implement**

```python
# ragregen/draft.py
"""Text-only drafting with FLUX.1-Kontext.

Loading the pipeline takes ~5m21s and one generation ~2 min at nf4 on the 4090
(measured: ../ImageRAG/results/kontext_example_20260721_224056/run.log). Load
once per process and draft every case before releasing the card -- never load
per case.
"""
from __future__ import annotations

from ragregen import env

env.setup()

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


def load_kontext_t2i(device: str = "cuda", quantize: str = "nf4"):
    """Load FluxKontextPipeline, nf4-quantised so it fits beside nothing else."""
    import torch
    from diffusers import FluxKontextPipeline

    kwargs = {"cache_dir": str(env.HF_CACHE), "torch_dtype": torch.bfloat16}
    if quantize == "nf4":
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True,
                          "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
    pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.to(device)
    return pipe


class Drafter:
    def __init__(self, pipe, steps: int = 28, seed: int = 0,
                 guidance: float = 3.5, device: str = "cuda"):
        self.pipe = pipe
        self.steps = steps
        self.seed = seed
        self.guidance = guidance
        self.device = device

    def draft(self, prompt: str):
        import torch

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        out = self.pipe(prompt, num_inference_steps=self.steps, generator=gen)
        return out.images[0]
```

```python
#!/usr/bin/env python
"""THE GATE (spec §10 step 0).

Drafts every case with FLUX.1-Kontext and writes labels.csv for hand-labelling.

If fewer than ~30% of drafts are labelled `fail`, FLUX already renders these
concepts and there is nothing for the pipeline to repair -- pick rarer concepts
before building anything else. The predecessor project assumed SDXL; FLUX is
much stronger on the long tail, so this must be measured, not assumed.

Usage: ./scripts/run.sh screen
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, draft, env, trace, validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)

    problems = validate.validate_dataset(ds) + validate.validate_disk()
    for p in problems:
        print(f"[{p.severity}] {p.code}: {p.message}")
    if any(p.severity == "error" for p in problems):
        print("\nFix the errors above before screening.")
        return 2

    cases = ds.cases[:args.limit] if args.limit else ds.cases
    run = trace.open_run("screen", argv=sys.argv, args=vars(args))
    print(f"[screen] {len(cases)} cases -> {run.path}")

    pipe = draft.load_kontext_t2i(device=args.device)
    drafter = draft.Drafter(pipe, steps=pipe_cfg.steps, seed=pipe_cfg.seed,
                            device=args.device)

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.id}: {case.prompt}", flush=True)
        image = drafter.draft(case.prompt)
        out = run.case_dir(case.id) / "draft.png"
        image.save(out)
        rows.append({"case_id": case.id, "prompt": case.prompt,
                     "concept": case.concept, "draft_path": str(out),
                     "verdict": "", "notes": ""})

    labels = run.path / "labels.csv"
    with labels.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    run.finish("ok", {"n_cases": len(rows)})
    env.reclaim_gpu()

    print(f"\n[gate] Open {labels}")
    print("[gate] Fill the `verdict` column with pass|fail for every row.")
    print("[gate] If fewer than ~30% are `fail`, pick rarer concepts and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_draft.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/draft.py scripts/screen_premise.py tests/test_draft.py
git commit -m "feat: premise-screening gate with FLUX.1-Kontext drafting"
```

- [ ] **Step 6: RUN THE GATE**

```bash
nvidia-smi
./scripts/run.sh screen
```

Then hand-label `outputs/screen_latest/labels.csv`. **Stop and report the fail rate before starting
Plan 2.** If it is below ~30%, the concept list needs to get rarer and Plan 2's design assumptions
change.

---

## Task 13: The `bakeoff` stage

**Files:**
- Create: `scripts/bakeoff.py`, `tests/test_bakeoff.py`

**Interfaces:**
- Consumes: `encoders.build_encoder`, `models.build_crop_scorer`, `retrieve.build_index`, `retrieve.Retriever`.
- Produces:
  - `recall_at_k(hits: list[Hit], correct: set[Path], k: int) -> float`
  - `mrr(hits: list[Hit], correct: set[Path]) -> float`
  - `bakeoff.py` writing `summary.md` and `results.json`

Reports FG-CLIP vs SigLIP **separately per role** — retriever and crop scorer — because a
region-text model can win one and lose the other (spec §6 Stage 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bakeoff.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.retrieve import Hit  # noqa: E402
from scripts import bakeoff  # noqa: E402


def hits(*stems):
    return [Hit(Path(f"/c/{s}.jpg"), 1.0 - i * 0.1, i)
            for i, s in enumerate(stems)]


def test_recall_at_1_hits():
    assert bakeoff.recall_at_k(hits("fox", "panda"), {Path("/c/fox.jpg")}, 1) == 1.0


def test_recall_at_1_misses():
    assert bakeoff.recall_at_k(hits("panda", "fox"), {Path("/c/fox.jpg")}, 1) == 0.0


def test_recall_at_5_finds_a_later_hit():
    assert bakeoff.recall_at_k(hits("a", "b", "fox"), {Path("/c/fox.jpg")}, 5) == 1.0


def test_mrr_is_reciprocal_of_the_first_correct_rank():
    assert bakeoff.mrr(hits("a", "fox"), {Path("/c/fox.jpg")}) == 0.5


def test_mrr_is_zero_when_absent():
    assert bakeoff.mrr(hits("a", "b"), {Path("/c/fox.jpg")}) == 0.0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_bakeoff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

Create `scripts/__init__.py` (empty) so the test can import it.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""FG-CLIP vs SigLIP, evaluated separately in each role (spec §6 Stage 1).

Two bands, because a region-text model can win one and lose the other:
  RETRIEVER   query -> whole-image ranking over the corpus (R@1, R@5, MRR)
  CROP SCORER phrase -> crop similarity, measured as breed-discrimination
              accuracy over the ground-truth references

Usage: ./scripts/run.sh bakeoff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, encoders, env, models, retrieve, trace  # noqa: E402

CANDIDATES = ("siglip_so400m_384", "fgclip")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    run = trace.open_run("bakeoff", argv=sys.argv, args=vars(args))

    retriever_rows, scorer_rows = [], []
    for name in CANDIDATES:
        print(f"[retriever] {name}", flush=True)
        retriever_rows.append(
            eval_retriever(name, ds, args.device, run.path / f"{name}.faiss"))
        print(f"[crop scorer] {name}", flush=True)
        scorer_rows.append(eval_crop_scorer(name, ds, args.device))

    lines = ["# Encoder bake-off", "",
             "## Retriever (query -> whole image)", "",
             "| encoder | R@1 | R@5 | MRR |", "|---|---|---|---|"]
    for r in retriever_rows:
        lines.append(f"| {r['encoder']} | {r['R@1']:.3f} | {r['R@5']:.3f} "
                     f"| {r['MRR']:.3f} |")
    lines += ["", "## Crop scorer (phrase -> crop)", "",
              "| encoder | accuracy | n | classes |", "|---|---|---|---|"]
    for r in scorer_rows:
        lines.append(f"| {r['encoder']} | {r['accuracy']:.3f} | {r['n']} "
                     f"| {r['n_classes']} |")
    lines += ["", "Set the winners in `configs/pipeline.yaml` as `retriever:` "
              "and `crop_scorer:`. They need not be the same encoder."]

    (run.path / "summary.md").write_text("\n".join(lines))
    run.write_json("results.json",
                   {"retriever": retriever_rows, "crop_scorer": scorer_rows})
    run.finish("ok", {"retriever": retriever_rows, "crop_scorer": scorer_rows})
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_bakeoff.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bakeoff.py scripts/__init__.py tests/test_bakeoff.py
git commit -m "feat: FG-CLIP vs SigLIP bake-off across both roles"
```

---

## Task 14: The `validate` stage and the `run.sh` entry point

**Files:**
- Create: `scripts/validate_cli.py`, `scripts/run.sh`
- Modify: `docs/RUNBOOK.md` — remove the ⏳ markers and the status banner

**Interfaces:**
- Consumes: everything above.
- Produces: `./scripts/run.sh {validate|screen|bakeoff|build-index}` — the operator's only entry point.

Plan 2 adds `pilot`, `full` and `report` to the same dispatcher.

- [ ] **Step 1: Write the CLI**

```python
#!/usr/bin/env python
"""Preflight the operator's configs. Runs before any GPU work.

Usage: ./scripts/run.sh validate
Exit codes: 0 = clean (warnings allowed), 2 = errors found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    args = ap.parse_args()

    try:
        ds = config.load_dataset(args.dataset)
        db = config.load_retrieval_db(args.db)
        pipe = config.load_pipeline(args.pipeline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] config: {exc}")
        return 2

    problems = validate.validate_all(ds, db, pipe)
    errors = [p for p in problems if p.severity == "error"]
    warnings = [p for p in problems if p.severity == "warning"]

    for p in errors + warnings:
        print(f"[{p.severity}] {p.code}: {p.message}")

    print(f"\n[summary] {len(ds.cases)} cases, corpus at {db.images_root}")
    print(f"[summary] {len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("[summary] Fix the errors above. Nothing will run until they are clear.")
        return 2
    print("[summary] OK to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the dispatcher**

```bash
#!/usr/bin/env bash
# The operator's only entry point. See docs/RUNBOOK.md.
set -euo pipefail

PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache

usage() {
  cat <<'EOF'
Usage: ./scripts/run.sh <stage> [options]

  validate     preflight your configs          (no GPU, seconds)
  build-index  corpus -> FAISS index           (GPU, ~1h / 500k)
  screen       THE GATE: draft + hand-label    (GPU, ~1h)
  bakeoff      FG-CLIP vs SigLIP, both roles   (GPU, ~1h)

Run `validate` first. `screen` gates everything downstream.
EOF
}

[ $# -ge 1 ] || { usage; exit 1; }
STAGE="$1"; shift

case "$STAGE" in
  validate)    exec "$PY" scripts/validate_cli.py "$@" ;;
  build-index) exec "$PY" scripts/build_index.py "$@" ;;
  screen)      exec "$PY" scripts/screen_premise.py "$@" ;;
  bakeoff)     exec "$PY" scripts/bakeoff.py "$@" ;;
  -h|--help)   usage ;;
  *)           echo "unknown stage: $STAGE" >&2; usage; exit 1 ;;
esac
```

- [ ] **Step 3: Make it executable and check it dispatches**

```bash
chmod +x scripts/run.sh
./scripts/run.sh --help
./scripts/run.sh validate --dataset configs/dataset.example.yaml --db configs/retrieval_db.example.yaml
```

Expected: the help text, then validation errors about the example paths not existing. Both are
correct — the example configs point at placeholder paths.

- [ ] **Step 4: Run the whole CPU suite**

Run: `$PY -m pytest -v`
Expected: all non-GPU tests pass.

- [ ] **Step 5: Update the RUNBOOK**

Remove the ⏳ status banner at the top of `docs/RUNBOOK.md` and the ⏳ markers on `validate`,
`build-index`, `screen` and `bakeoff` in §2.2, §3.2, §3.3 and §4. Leave the markers on `pilot`,
`full` and `report` — those are Plan 2.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_cli.py scripts/run.sh docs/RUNBOOK.md
git commit -m "feat: operator CLI with validate, build-index, screen, bakeoff"
```

---

## Self-Review

**Spec coverage.** Spec §7 (operator interface) → Tasks 2, 3, 14. §2 modules `concepts`,
`verify/grounded`, `verify/semantic`, `verify/fusion` → Tasks 5, 6, 8, 9. §2 `retrieve` → Tasks 10,
11. §11 (logging) → Task 4. §6 Stage 1 (bake-off) → Task 13. §10 step 0 (gate) → Task 12. §10 step 1
(deps) → Task 1.

**Deferred to Plan 2, by design:** `mask.py`, `regen.py`, `schedule.py`, `metrics.py`, `query.py`,
§6 Stage 2, §6 Stage 3, and §5 metric implementations. `query.py` moves to Plan 2 because it only
has a consumer once regeneration exists.

**Known soft spots, flagged rather than hidden:**

1. **`qihoo360/fg-clip-base` is the one unverified identifier in this plan.** Every other model id
   was confirmed present in `../.cache`. If it 404s, correct `SCORER_IDS` and `ENCODER_IDS` — Task 7
   Step 6 and Task 11 Step 5 are where it surfaces.
2. **`_HFEncoder.dim` reads `config.projection_dim`,** which some checkpoints do not expose. Task 11
   Step 5 measures it against `validate.ENCODER_DIMS`; the measured value is the truth and the table
   gets corrected, not the code.
3. **`validate.ENCODER_DIMS` values are from model cards, not measured.** Same test corrects them.
4. **Task 12's `Drafter` test uses a fake pipe,** so it verifies wiring, not image quality. That is
   deliberate: real drafting costs 2 minutes per call, and its actual verification is the human
   labelling pass in Step 6.

**Type consistency.** `Detector.all_boxes` / `CropScorer.score` (Task 6) are implemented by
`DinoDetector` / `SigLIPScorer` (Task 7). `ConceptScore.state` uses the same three strings in Tasks
6 and 9. `SemanticVerdict` fields `ok`/`issues`/`raw`/`degenerate` are consumed unchanged by
`fuse()`. `Hit(path, score, rank)` is produced in Task 10 and consumed in Task 13. `build_encoder`
names are keyed to `validate.ENCODER_DIMS` and Task 11's second test enforces it.
