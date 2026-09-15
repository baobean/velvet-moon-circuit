# LAION Common-Concept Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer whether the full retrieve-and-repair pipeline damages images of concepts FLUX already renders correctly, by giving the never-executed `--arm full` branch a real 100k LAION corpus and a 92-case set whose answer key comes from ImageNet.

**Architecture:** Two new pure modules — `ragregen/corpus.py` (which LAION rows to keep) and `ragregen/casegen.py` (which ImageNet classes become cases) — each driven by a thin CLI in `scripts/`, matching the existing `ragregen/c1.py` ↔ `scripts/c1_report.py` split. `scripts/report.py` gains a paired within-case DINO delta, which is the headline. `run_pipeline.py` gains a VRAM preflight and `scripts/supervise.sh` re-acquires the card after eviction, so a 15-hour run on a shared 4090 survives neighbours. Everything except encoder loads is CPU-testable.

**Tech Stack:** Python 3.11, numpy, Pillow, pyarrow, img2dataset, faiss, transformers, pytest, bash.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Bare `python` is conda 3.13 and fails on faiss. Verified 2026-08-05: Python 3.11.15, faiss and pytest import.
- `export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache` before anything touching HuggingFace. `run.sh` already does this; standalone `$PY` invocations must do it too.
- **ImageNet is never in the corpus.** `gt_refs` live under `data/gt_refs/`, the corpus under `data/laion100k/`. `validate.py:88` raises `gt_ref_in_corpus` if they overlap — do not "fix" that error by relaxing the check.
- **The non-inferiority margin is −0.02 DINO cosine and is fixed** (spec §2). Never tune it after seeing deltas.
- **The paired delta scores draft and output with the SAME mask and the SAME held-out refs.** Any divergence makes the headline meaningless while still printing a plausible number.
- **Cohorts never merge.** `bridge` cases have full-resolution refs, `common` cases have 256px refs. Never one DINO mean across both.
- **Healthy cases are excluded from the delta mean**, reported separately as delta-zero-by-construction.
- Repair rate still comes from `scores.json`'s draft verdict, never `queue.json`'s `status` (doc 4 §4).
- Held-out references are never edited with. Round count clamps to `len(edit refs)`.
- Eval encoders must differ from `retriever` and `crop_scorer`; config load fails otherwise.
- All metrics compute at the draft's size. Size mismatch raises; never resize.
- Verifier pass-rate appears nowhere near a headline.
- `git add <exact paths>` — never `git add -A`. `outputs/` and `data/` are gitignored.
- The existing suite (399 test functions across 34 files) must stay green.
- **GPU etiquette:** `nvidia-smi` before anything that loads a model. Kill background runs with `kill -9 $(pgrep -f 'bin/python scripts/run_pipeline.py')`, never `$!`.
- Branch is `doc5-laion-benchmark`. Commit there, not `master`.

## File Structure

| file | responsibility | new? |
|---|---|---|
| `ragregen/corpus.py` | caption matching, seeded row selection, per-concept counting | create |
| `ragregen/casegen.py` | class ranking, prompt framing, dataset YAML emission | create |
| `scripts/fetch_corpus.py` | CLI: `select` \| `download` \| `manifest` | create |
| `scripts/make_cases.py` | CLI: ImageNet parquet → `gt_refs` + `dataset_common.yaml` | create |
| `configs/imagenet_coarse.yaml` | curated class → groundable coarse term map | create |
| `scripts/supervise.sh` | re-acquire the card and resume after eviction | create |
| `ragregen/config.py` | `Case.cohort` | modify |
| `ragregen/env.py` | `free_vram_gb()` | modify |
| `ragregen/schedule.py` | heartbeat line per case | modify |
| `scripts/run_pipeline.py` | VRAM preflight in `stage_with_model` | modify |
| `ragregen/metrics.py` | `paired_bootstrap_ci`, `MARGIN` | modify |
| `scripts/report.py` | `dino_cropped_draft`, delta columns, `(kind, cohort)` strata | modify |
| `ragregen/validate.py` | corpus-density warning | modify |
| `scripts/run.sh`, `docs/RUNBOOK.md` | operator surface | modify |

---

### Task 1: `cohort` on the case schema

The report must never average a 256px-reference case with a full-resolution one. That distinction has to be data before any code can honour it.

**Files:**
- Modify: `ragregen/config.py:97` (the `KINDS` block), `ragregen/config.py:22-32` (the `Case` dataclass)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Case.cohort: str` — `"bridge"` | `"common"`, default `"bridge"`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def _one_case_yaml(tmp_path, extra=""):
    p = tmp_path / "d.yaml"
    p.write_text(f"""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "African grey parrot"
    coarse: "parrot"
{extra}
    gt_refs: [x.jpg]
""")
    return p


def test_cohort_defaults_to_bridge(tmp_path):
    #: The 22 legacy cases carry no `cohort:` key and must keep behaving as
    #: they did, so the default is the legacy value, not the new one.
    ds = config.load_dataset(_one_case_yaml(tmp_path))
    assert ds.cases[0].cohort == "bridge"


def test_cohort_is_read_when_present(tmp_path):
    p = _one_case_yaml(tmp_path, "    cohort: common")
    assert config.load_dataset(p).cases[0].cohort == "common"


def test_an_unknown_cohort_is_refused(tmp_path):
    #: A typo silently merges a 256px-ref case into the full-res mean, which
    #: is exactly the comparison spec §5 forbids.
    p = _one_case_yaml(tmp_path, "    cohort: banana")
    with pytest.raises(ValueError, match="cohort"):
        config.load_dataset(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_config.py -k cohort -v`
Expected: FAIL with `AttributeError: 'Case' object has no attribute 'cohort'`

- [ ] **Step 3: Write minimal implementation**

In `ragregen/config.py`, add to the `Case` dataclass after `kind`:

```python
    #: "bridge" = one of the 22 legacy cases with full-resolution gt_refs;
    #: "common" = an ImageNet-derived case with 256px refs. The report never
    #: averages DINO across the two -- different reference resolutions
    #: (doc 5 §5).
    cohort: str = "bridge"
```

In `load_dataset`, beside the existing `KINDS`:

```python
    KINDS = ("target", "control")
    COHORTS = ("bridge", "common")
```

and after the `kind` validation block:

```python
        cohort = str(raw.get("cohort", "bridge"))
        if cohort not in COHORTS:
            raise ValueError(
                f"case {cid!r} has cohort {cohort!r}; expected one of "
                f"{COHORTS}")
```

then pass `cohort=cohort` to the `Case(...)` constructor.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: PASS, including every pre-existing config test.

- [ ] **Step 5: Commit**

```bash
git add ragregen/config.py tests/test_config.py
git commit -m "feat: cohort on the case schema, so resolutions cannot merge"
```

---

### Task 2: `ragregen/corpus.py` — which LAION rows to keep

The corpus is a random bulk (distractor mass, so retrieval is a real test) plus a caption-matched slice per concept (so the answer exists at all). This task is the pure selection logic; no network, no parquet.

**Files:**
- Create: `ragregen/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `Case.concept` strings from Task 1's schema
- Produces:
  - `caption_matches(caption: str, concept: str) -> bool`
  - `count_per_concept(captions: list[str], concepts: list[str]) -> dict[str, int]`
  - `select_rows(captions: list[str], concepts: list[str], *, n_random: int, n_per_concept: int, seed: int) -> list[int]` — sorted, de-duplicated row indices

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
"""Corpus selection. No network, no parquet -- just which rows we keep.

The corpus does two jobs (doc 5 §4): the seeded slice guarantees the answer
exists, the random bulk makes finding it non-trivial. Both are tested here
because getting either wrong produces a plausible-looking corpus that
measures nothing.
"""
import pytest

from ragregen import corpus


def test_a_word_boundary_match_hits():
    assert corpus.caption_matches("a red fox in the snow", "fox")


def test_matching_is_case_insensitive():
    assert corpus.caption_matches("A Golden Retriever puppy",
                                  "golden retriever")


def test_a_substring_inside_a_word_is_not_a_match():
    #: "foxglove" is a flower. Without word boundaries the fox slice fills
    #: with plants and retrieval looks broken for reasons that are ours.
    assert not corpus.caption_matches("purple foxglove blooms", "fox")


def test_multi_word_concepts_match_as_a_phrase():
    assert corpus.caption_matches("my golden retriever", "golden retriever")
    assert not corpus.caption_matches("a retriever of golden things",
                                      "golden retriever")


def test_count_per_concept_counts_every_matching_caption():
    caps = ["a fox", "another fox", "a violin", "nothing here"]
    got = corpus.count_per_concept(caps, ["fox", "violin", "panda"])
    assert got == {"fox": 2, "violin": 1, "panda": 0}


def test_select_takes_the_requested_number_per_concept():
    caps = [f"a fox number {i}" for i in range(50)] + ["unrelated"] * 50
    idx = corpus.select_rows(caps, ["fox"], n_random=0, n_per_concept=10,
                             seed=0)
    assert len(idx) == 10
    assert all(caps[i].startswith("a fox") for i in idx)


def test_select_adds_random_bulk_on_top_of_the_seeded_slice():
    caps = [f"a fox number {i}" for i in range(50)] + ["unrelated"] * 50
    idx = corpus.select_rows(caps, ["fox"], n_random=20, n_per_concept=10,
                             seed=0)
    #: The bulk is drawn from everything, so it may re-draw a fox row. The
    #: guarantee is >= the seeded slice and <= seeded + bulk, never double
    #: counting.
    assert 10 <= len(idx) <= 30
    assert len(idx) == len(set(idx))


def test_select_is_deterministic_for_a_seed():
    caps = [f"caption {i}" for i in range(200)]
    a = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    b = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    assert a == b


def test_a_different_seed_selects_differently():
    caps = [f"caption {i}" for i in range(200)]
    a = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    b = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=8)
    assert a != b


def test_a_concept_with_too_few_captions_takes_what_exists():
    #: Not an error. A thin concept is a finding the report must show, and
    #: raising here would abort a 100k download over one rare word.
    caps = ["a lone axolotl"]
    idx = corpus.select_rows(caps, ["axolotl"], n_random=0, n_per_concept=10,
                             seed=0)
    assert idx == [0]


def test_returned_indices_are_sorted():
    #: Parquet take() is much faster on sorted indices, and a sorted list
    #: makes the URL file diffable between runs.
    caps = [f"caption {i}" for i in range(200)]
    idx = corpus.select_rows(caps, [], n_random=40, n_per_concept=0, seed=3)
    assert idx == sorted(idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragregen.corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `ragregen/corpus.py`:

```python
"""Which LAION rows become the retrieval corpus.

Two jobs, sized separately (doc 5 §4). The per-concept slice guarantees
retrieval has something to find -- in random web images a given ImageNet class
appears at roughly 1-in-10^4, so a random 100k yields single digits for a
common concept. The random bulk is distractor mass: a corpus of only the
seeded concepts would make retrieval trivially correct and measure nothing.

Pure functions over caption strings. Parquet and network live in
scripts/fetch_corpus.py so this stays testable without either.
"""
from __future__ import annotations

import re

import numpy as np


def caption_matches(caption: str, concept: str) -> bool:
    """Whether `caption` mentions `concept` as a whole word or phrase.

    Word-bounded on purpose: a plain substring test puts "foxglove" in the
    fox slice, and retrieval then looks broken for a reason that is ours
    rather than the method's.
    """
    pattern = r"\b" + r"\s+".join(
        re.escape(w) for w in concept.lower().split()) + r"\b"
    return re.search(pattern, caption.lower()) is not None


def count_per_concept(captions, concepts) -> dict[str, int]:
    """How many captions mention each concept. Published in the manifest."""
    caps = list(captions)
    return {c: sum(1 for cap in caps if caption_matches(cap, c))
            for c in concepts}


def select_rows(captions, concepts, *, n_random: int, n_per_concept: int,
                seed: int) -> list[int]:
    """Row indices to download: the seeded slice, plus random bulk.

    Deterministic for a seed, so a failed download resumes against the same
    sample instead of silently re-rolling the corpus.
    """
    caps = list(captions)
    rng = np.random.default_rng(seed)
    keep: set[int] = set()

    for concept in concepts:
        hits = [i for i, cap in enumerate(caps)
                if caption_matches(cap, concept)]
        #: Fewer hits than asked for is a finding, not an error -- raising
        #: would abort a 100k download over one rare word.
        take = min(n_per_concept, len(hits))
        if take:
            keep.update(int(i) for i in rng.choice(hits, size=take,
                                                   replace=False))

    if n_random:
        bulk = rng.choice(len(caps), size=min(n_random, len(caps)),
                          replace=False)
        keep.update(int(i) for i in bulk)

    return sorted(keep)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_corpus.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add ragregen/corpus.py tests/test_corpus.py
git commit -m "feat: corpus selection, seeded slice plus distractor bulk"
```

---

### Task 3: `scripts/fetch_corpus.py` — acquire the 100k

Three subcommands, separable because selection is cheap and deterministic while download is slow and flaky. Conflating them means a network failure discards a reproducible sample.

**Files:**
- Create: `scripts/fetch_corpus.py`
- Create: `configs/retrieval_db.yaml`
- Test: `tests/test_fetch_corpus.py`

**Interfaces:**
- Consumes: `corpus.select_rows`, `corpus.count_per_concept` (Task 2)
- Produces:
  - `data/laion100k/urls.parquet` — columns `URL`, `TEXT`
  - `data/laion100k/images/` — 384px images from `img2dataset`
  - `data/laion100k/corpus_manifest.json` — `{"n_images": int, "per_concept": {concept: count}}`
  - `build_manifest(image_dir: Path, concepts: list[str]) -> dict`

- [ ] **Step 1: Install the two missing dependencies**

Verified missing on 2026-08-05: `img2dataset`, `pyarrow`. Present: `pandas 3.0.5`, `scipy 1.17.1`.

```bash
$PY -m pip install img2dataset pyarrow
```

Confirm: `$PY -c "import img2dataset, pyarrow; print('ok')"` → `ok`

- [ ] **Step 2: Write the failing test**

Create `tests/test_fetch_corpus.py`:

```python
"""Manifest arithmetic. No network -- img2dataset is never invoked here.

The manifest is what lets a bad result be attributed to retrieval failure
versus corpus sparsity (doc 5 §4), so it has to count what actually landed on
disk rather than what was requested.
"""
import json

from PIL import Image

from scripts.fetch_corpus import build_manifest


def _corpus(root, entries):
    """entries: {filename: caption}. Writes images plus img2dataset sidecars."""
    root.mkdir(parents=True, exist_ok=True)
    for name, caption in entries.items():
        Image.new("RGB", (8, 8), (1, 2, 3)).save(root / f"{name}.jpg")
        (root / f"{name}.json").write_text(json.dumps({"caption": caption}))
    return root


def test_manifest_counts_images_that_actually_landed(tmp_path):
    root = _corpus(tmp_path / "images",
                   {"000": "a red fox", "001": "a violin", "002": "nothing"})
    got = build_manifest(root, ["fox", "violin", "panda"])
    assert got["n_images"] == 3


def test_manifest_counts_captions_per_concept(tmp_path):
    root = _corpus(tmp_path / "images",
                   {"000": "a red fox", "001": "another fox",
                    "002": "a violin"})
    got = build_manifest(root, ["fox", "violin", "panda"])
    assert got["per_concept"] == {"fox": 2, "violin": 1, "panda": 0}


def test_a_concept_the_download_missed_reports_zero_not_absent(tmp_path):
    #: Absent would read as "not asked for". Zero reads as "asked for and the
    #: web had nothing", which is the finding.
    root = _corpus(tmp_path / "images", {"000": "a red fox"})
    got = build_manifest(root, ["fox", "axolotl"])
    assert got["per_concept"]["axolotl"] == 0


def test_an_image_without_a_sidecar_still_counts_as_an_image(tmp_path):
    #: img2dataset occasionally writes the jpg and not the json. Dropping the
    #: image would understate corpus size; counting its caption would crash.
    root = _corpus(tmp_path / "images", {"000": "a red fox"})
    Image.new("RGB", (8, 8)).save(root / "orphan.jpg")
    got = build_manifest(root, ["fox"])
    assert got["n_images"] == 2
    assert got["per_concept"]["fox"] == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `$PY -m pytest tests/test_fetch_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.fetch_corpus'`

- [ ] **Step 4: Write the implementation**

Create `scripts/fetch_corpus.py`:

```python
#!/usr/bin/env python
"""Build the 100k LAION retrieval corpus. Doc 5 §7.

Three subcommands, deliberately separate: `select` is cheap and deterministic,
`download` is slow and flaky, `manifest` reports what actually survived. Fusing
them would mean a network failure discards a reproducible sample.

Usage:
  ./scripts/run.sh fetch-corpus select
  ./scripts/run.sh fetch-corpus download
  ./scripts/run.sh fetch-corpus manifest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

from ragregen import config, corpus, env  # noqa: E402

#: One 64.7 MB shard holds far more rows than we need; a second is fetched
#: only if the first cannot fill the request.
LAION_REPO = "laion/laion2B-en-aesthetic"
LAION_SHARDS = [
    "part-00000-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy.parquet",
    "part-00001-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy.parquet",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _concepts(dataset_path: Path) -> list[str]:
    return [c.concept for c in config.load_dataset(dataset_path).cases]


def cmd_select(args) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    concepts = _concepts(args.dataset)
    tables = []
    for shard in LAION_SHARDS[:args.shards]:
        print(f"[select] fetching metadata shard {shard}", flush=True)
        local = hf_hub_download(LAION_REPO, shard, repo_type="dataset")
        tables.append(pq.read_table(local, columns=["URL", "TEXT"]))

    import pyarrow as pa
    table = pa.concat_tables(tables)
    captions = [c or "" for c in table.column("TEXT").to_pylist()]
    print(f"[select] {len(captions)} candidate rows, "
          f"{len(concepts)} concepts", flush=True)

    idx = corpus.select_rows(captions, concepts, n_random=args.n_random,
                             n_per_concept=args.n_per_concept, seed=args.seed)
    out = table.take(idx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out)
    print(f"[select] {len(idx)} rows -> {args.out}", flush=True)
    return 0


def cmd_download(args) -> int:
    """Shell out to img2dataset. Dead links are the norm, not an error."""
    args.image_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "img2dataset",
        "--url_list", str(args.urls),
        "--input_format", "parquet",
        "--url_col", "URL",
        "--caption_col", "TEXT",
        "--output_folder", str(args.image_dir),
        "--output_format", "files",
        "--image_size", "384",
        "--resize_mode", "keep_ratio",
        "--processes_count", str(args.processes),
        "--thread_count", str(args.threads),
        "--retries", "1",
        "--encode_quality", "90",
        "--encode_format", "jpg",
        "--disable_all_reencoding", "False",
    ]
    print("[download] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def build_manifest(image_dir: Path, concepts) -> dict:
    """What actually landed, counted per concept.

    Counts images and captions separately: img2dataset sometimes writes the
    jpg without its json sidecar, and dropping such an image would understate
    the corpus while reading its caption would crash.
    """
    image_dir = Path(image_dir)
    images = [p for p in image_dir.rglob("*")
              if p.suffix.lower() in IMAGE_SUFFIXES]
    captions = []
    for img in images:
        sidecar = img.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            captions.append(json.loads(sidecar.read_text()).get("caption")
                            or "")
        except (OSError, json.JSONDecodeError):
            continue
    return {
        "n_images": len(images),
        "n_captioned": len(captions),
        "per_concept": corpus.count_per_concept(captions, list(concepts)),
    }


def cmd_manifest(args) -> int:
    manifest = build_manifest(args.image_dir, _concepts(args.dataset))
    args.out.write_text(json.dumps(manifest, indent=2))
    thin = [c for c, n in manifest["per_concept"].items() if n < args.k]
    print(f"[manifest] {manifest['n_images']} images -> {args.out}")
    if thin:
        #: Printed, never raised. "The corpus had nothing here" is a result
        #: the report must carry, not a reason to abort (doc 5 §10).
        print(f"[manifest] {len(thin)} concepts below k={args.k}: "
              f"{', '.join(sorted(thin)[:10])}")
    return 0


def main() -> int:
    env.setup()
    root = env.PROJECT_ROOT / "data" / "laion100k"

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=config.DEFAULT_DATASET_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--out", type=Path, default=root / "urls.parquet")
    s.add_argument("--n-random", dest="n_random", type=int, default=90_000)
    s.add_argument("--n-per-concept", dest="n_per_concept", type=int,
                   default=200)
    s.add_argument("--shards", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_select)

    d = sub.add_parser("download")
    d.add_argument("--urls", type=Path, default=root / "urls.parquet")
    d.add_argument("--image-dir", type=Path, default=root / "images")
    d.add_argument("--processes", type=int, default=8)
    d.add_argument("--threads", type=int, default=32)
    d.set_defaults(func=cmd_download)

    m = sub.add_parser("manifest")
    m.add_argument("--image-dir", type=Path, default=root / "images")
    m.add_argument("--out", type=Path, default=root / "corpus_manifest.json")
    m.add_argument("--k", type=int, default=3)
    m.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

Note `--n-random 90000` and `--n-per-concept 200`: 92 concepts × 200 = 18,400 seeded rows requested, plus 90,000 bulk, against ~50% img2dataset attrition lands near 100k. `manifest` reports the truth.

- [ ] **Step 5: Run test to verify it passes**

Run: `$PY -m pytest tests/test_fetch_corpus.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Write the retrieval DB config**

Create `configs/retrieval_db.yaml`:

```yaml
# The 100k LAION corpus the `full` arm searches. See doc 5 §3.
#
# This is the ONLY image set the pipeline may consult. ImageNet gt_refs live
# under data/gt_refs/ and are the answer key -- validate.py refuses to run if
# they ever land inside images_root below.

name: laion100k
images_root: /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen/data/laion100k/images
index_path: data/laion100k/index.faiss
encoder: siglip_so400m_384      # must match what build-index used
captions: null
```

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_corpus.py tests/test_fetch_corpus.py configs/retrieval_db.yaml
git commit -m "feat: fetch the corpus in three steps, and report what landed"
```

---

### Task 4: `configs/imagenet_coarse.yaml` — groundable superordinates

`config.load_dataset` refuses a case whose `coarse` equals its `concept` (line 111), and `mask_draft` grounds the coarse term with GroundingDINO. So every new case needs a superordinate that is both more general than the concept and reliably groundable. WordNet hypernyms produce terms like "teleost fishes" that no detector grounds; a curated map does not.

This map is also the candidate pool: Task 5 ranks by corpus frequency **within** it, so ranking still does real work at 120 candidates for 70 slots.

**Files:**
- Create: `configs/imagenet_coarse.yaml`
- Test: `tests/test_casegen.py` (the schema check; ranking arrives in Task 5)

**Interfaces:**
- Consumes: nothing
- Produces: `configs/imagenet_coarse.yaml` — a flat `concept: coarse` mapping, ≥120 entries

- [ ] **Step 1: Write the failing test**

Create `tests/test_casegen.py`:

```python
"""The coarse map and case generation.

`config.load_dataset` rejects coarse == concept because identical terms make
the fine-grained margin identically zero, silently disabling the test. Every
entry here has to clear that bar before it can become a case.
"""
from pathlib import Path

import yaml

COARSE_PATH = Path("configs/imagenet_coarse.yaml")


def _coarse_map():
    return yaml.safe_load(COARSE_PATH.read_text())


def test_the_map_offers_more_candidates_than_the_70_we_need():
    #: Ranking picks 70 by corpus frequency. A pool of exactly 70 would make
    #: the ranking decorative.
    assert len(_coarse_map()) >= 120


def test_no_entry_has_coarse_equal_to_concept():
    #: config.load_dataset:111 raises on this, so a bad entry aborts
    #: make_cases rather than producing a degenerate case.
    bad = [k for k, v in _coarse_map().items()
           if k.strip().lower() == str(v).strip().lower()]
    assert bad == []


def test_every_coarse_term_is_a_non_empty_string():
    assert all(isinstance(v, str) and v.strip() for v in _coarse_map().values())


def test_concepts_are_lowercase_and_unique():
    #: Ranking matches captions case-insensitively; storing mixed case here
    #: would make the map's keys disagree with the manifest's.
    m = _coarse_map()
    assert all(k == k.lower() for k in m)
    assert len(m) == len(set(m))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_casegen.py -v`
Expected: FAIL with `FileNotFoundError: configs/imagenet_coarse.yaml`

- [ ] **Step 3: Write the map**

Create `configs/imagenet_coarse.yaml`:

```yaml
# ImageNet class -> groundable superordinate. Doc 5 §5.
#
# This is the candidate pool for the `common` cohort: make_cases.py ranks
# these by LAION caption frequency and takes the top 70. A pool larger than
# the selection keeps the ranking meaningful.
#
# Two rules for every entry:
#   1. coarse != concept, or config.load_dataset:111 raises -- identical terms
#      make the fine-grained margin identically zero.
#   2. coarse must be a term GroundingDINO actually grounds. "dog" grounds;
#      "domestic animal" does not. WordNet hypernyms were rejected for this
#      reason ("teleost fishes").

goldfish: fish
great white shark: shark
tiger shark: shark
hammerhead: shark
stingray: fish
hen: bird
ostrich: bird
goldfinch: bird
house finch: bird
junco: bird
indigo bunting: bird
robin: bird
bulbul: bird
jay: bird
magpie: bird
chickadee: bird
water ouzel: bird
kite: bird
bald eagle: bird
vulture: bird
great grey owl: owl
fire salamander: salamander
common newt: salamander
eft: salamander
spotted salamander: salamander
bullfrog: frog
tree frog: frog
tailed frog: frog
loggerhead: turtle
leatherback turtle: turtle
mud turtle: turtle
terrapin: turtle
box turtle: turtle
banded gecko: lizard
common iguana: lizard
american chameleon: lizard
whiptail: lizard
agama: lizard
frilled lizard: lizard
alligator lizard: lizard
gila monster: lizard
green lizard: lizard
african crocodile: crocodile
american alligator: crocodile
triceratops: dinosaur
thunder snake: snake
ringneck snake: snake
hognose snake: snake
green snake: snake
king snake: snake
garter snake: snake
water snake: snake
vine snake: snake
night snake: snake
boa constrictor: snake
rock python: snake
indian cobra: snake
green mamba: snake
sea snake: snake
horned viper: snake
diamondback: snake
sidewinder: snake
trilobite: fossil
harvestman: spider
scorpion: arachnid
black and gold garden spider: spider
barn spider: spider
garden spider: spider
black widow: spider
tarantula: spider
wolf spider: spider
tick: insect
centipede: arthropod
black grouse: bird
ptarmigan: bird
ruffed grouse: bird
prairie chicken: bird
peacock: bird
quail: bird
partridge: bird
african grey: parrot
macaw: parrot
sulphur-crested cockatoo: parrot
lorikeet: parrot
coucal: bird
bee eater: bird
hornbill: bird
hummingbird: bird
jacamar: bird
toucan: bird
drake: duck
red-breasted merganser: duck
goose: bird
black swan: bird
tusker: elephant
echidna: mammal
platypus: mammal
wallaby: marsupial
koala: marsupial
wombat: marsupial
jellyfish: sea creature
sea anemone: sea creature
brain coral: coral
flatworm: worm
nematode: worm
conch: shell
snail: mollusc
slug: mollusc
sea slug: sea creature
chiton: mollusc
chambered nautilus: shell
dungeness crab: crab
rock crab: crab
fiddler crab: crab
king crab: crab
american lobster: lobster
spiny lobster: lobster
crayfish: crustacean
hermit crab: crab
isopod: crustacean
white stork: bird
black stork: bird
spoonbill: bird
flamingo: bird
little blue heron: bird
american egret: bird
bittern: bird
crane: bird
limpkin: bird
european gallinule: bird
american coot: bird
bustard: bird
ruddy turnstone: bird
red-backed sandpiper: bird
redshank: bird
dowitcher: bird
oystercatcher: bird
pelican: bird
king penguin: penguin
albatross: bird
grey whale: whale
killer whale: whale
dugong: sea mammal
sea lion: seal
chihuahua: dog
japanese spaniel: dog
maltese dog: dog
pekinese: dog
shih-tzu: dog
blenheim spaniel: dog
papillon: dog
toy terrier: dog
rhodesian ridgeback: dog
afghan hound: dog
basset: dog
beagle: dog
bloodhound: dog
bluetick: dog
golden retriever: dog
labrador retriever: dog
german shepherd: dog
rottweiler: dog
siberian husky: dog
pug: dog
pomeranian: dog
chow: dog
samoyed: dog
dalmatian: dog
boston bull: dog
tabby: cat
persian cat: cat
siamese cat: cat
egyptian cat: cat
cougar: big cat
lynx: big cat
leopard: big cat
snow leopard: big cat
jaguar: big cat
lion: big cat
tiger: big cat
cheetah: big cat
brown bear: bear
american black bear: bear
ice bear: bear
sloth bear: bear
mongoose: mammal
meerkat: mammal
ladybug: insect
dragonfly: insect
monarch: butterfly
sulphur butterfly: butterfly
starfish: sea creature
sea urchin: sea creature
wood rabbit: rabbit
hare: rabbit
hamster: rodent
porcupine: rodent
fox squirrel: squirrel
marmot: rodent
beaver: rodent
guinea pig: rodent
sorrel: horse
zebra: equine
hog: pig
wild boar: pig
warthog: pig
hippopotamus: mammal
ox: cattle
water buffalo: cattle
bison: cattle
ram: sheep
bighorn: sheep
ibex: goat
impala: antelope
gazelle: antelope
arabian camel: camel
llama: mammal
weasel: mammal
mink: mammal
polecat: mammal
otter: mammal
skunk: mammal
badger: mammal
orangutan: ape
gorilla: ape
chimpanzee: ape
gibbon: ape
guenon: monkey
baboon: monkey
macaque: monkey
langur: monkey
capuchin: monkey
howler monkey: monkey
spider monkey: monkey
squirrel monkey: monkey
madagascar cat: lemur
indri: lemur
giant panda: bear
lesser panda: mammal
acoustic guitar: guitar
electric guitar: guitar
banjo: string instrument
cello: string instrument
violin: string instrument
harp: string instrument
grand piano: piano
accordion: instrument
harmonica: instrument
oboe: instrument
sax: instrument
trombone: instrument
cornet: instrument
french horn: instrument
drum: instrument
gong: instrument
maraca: instrument
steel drum: instrument
ambulance: vehicle
beach wagon: car
cab: car
convertible: car
jeep: car
limousine: car
minivan: car
model t: car
racer: car
sports car: car
fire engine: truck
garbage truck: truck
pickup: truck
tow truck: truck
trailer truck: truck
school bus: bus
trolleybus: bus
minibus: bus
moped: motorcycle
mountain bike: bicycle
tandem bicycle: bicycle
unicycle: cycle
motor scooter: scooter
airliner: aircraft
warplane: aircraft
balloon: aircraft
airship: aircraft
space shuttle: spacecraft
canoe: boat
catamaran: boat
fireboat: boat
gondola: boat
lifeboat: boat
speedboat: boat
yawl: boat
schooner: boat
container ship: ship
liner: ship
pirate: ship
submarine: vessel
aircraft carrier: ship
steam locomotive: train
electric locomotive: train
freight car: train
passenger car: train
bassinet: bed
cradle: bed
four-poster: bed
crib: bed
studio couch: couch
rocking chair: chair
folding chair: chair
barber chair: chair
throne: chair
desk: table
dining table: table
pool table: table
wardrobe: cabinet
chiffonier: cabinet
file: cabinet
bookcase: shelf
espresso maker: appliance
microwave: appliance
dishwasher: appliance
refrigerator: appliance
toaster: appliance
waffle iron: appliance
washer: appliance
vacuum: appliance
electric fan: appliance
space heater: appliance
coffee mug: mug
beer glass: glass
goblet: glass
wine bottle: bottle
beer bottle: bottle
water bottle: bottle
pop bottle: bottle
whiskey jug: jug
teapot: pot
coffeepot: pot
frying pan: pan
wok: pan
dutch oven: pot
caldron: pot
mixing bowl: bowl
soup bowl: bowl
plate: dish
tray: dish
spatula: utensil
wooden spoon: utensil
ladle: utensil
corkscrew: tool
can opener: tool
hammer: tool
screwdriver: tool
power drill: tool
chain saw: tool
lawn mower: machine
shovel: tool
plunger: tool
broom: tool
umbrella: accessory
backpack: bag
purse: bag
sleeping bag: bag
mailbag: bag
plastic bag: bag
running shoe: shoe
cowboy boot: boot
clog: shoe
sandal: shoe
sombrero: hat
cowboy hat: hat
bearskin: hat
bonnet: hat
crash helmet: helmet
football helmet: helmet
mitten: glove
sunglasses: glasses
necklace: jewellery
wall clock: clock
analog clock: clock
digital clock: clock
hourglass: timepiece
sundial: timepiece
barometer: instrument
stopwatch: timepiece
laptop: computer
desktop computer: computer
notebook: computer
hand-held computer: computer
cellular telephone: phone
dial telephone: phone
pay-phone: phone
television: screen
monitor: screen
computer keyboard: keyboard
mouse: computer accessory
joystick: controller
remote control: controller
printer: machine
photocopier: machine
projector: machine
loudspeaker: speaker
microphone: audio equipment
cassette player: audio equipment
cd player: audio equipment
radio: audio equipment
tape player: audio equipment
reflex camera: camera
polaroid camera: camera
binoculars: optical instrument
sunglass: glasses
candle: light
table lamp: lamp
spotlight: light
torch: light
jack-o'-lantern: pumpkin
pinwheel: toy
teddy: toy
soccer ball: ball
basketball: ball
volleyball: ball
rugby ball: ball
golf ball: ball
tennis ball: ball
ping-pong ball: ball
baseball: ball
croquet ball: ball
punching bag: sports equipment
dumbbell: sports equipment
barbell: sports equipment
balance beam: sports equipment
horizontal bar: sports equipment
parallel bars: sports equipment
ski: sports equipment
snowmobile: vehicle
dogsled: sled
bobsled: sled
paddle: oar
racket: sports equipment
bow: weapon
rifle: weapon
revolver: weapon
cannon: weapon
castle: building
church: building
mosque: building
palace: building
monastery: building
library: building
barn: building
greenhouse: building
boathouse: building
lighthouse: tower
bell cote: tower
obelisk: monument
triumphal arch: monument
viaduct: bridge
suspension bridge: bridge
steel arch bridge: bridge
pier: structure
dam: structure
fountain: structure
picket fence: fence
worm fence: fence
chainlink fence: fence
stone wall: wall
mountain tent: tent
parachute: fabric
flagpole: pole
street sign: sign
traffic light: signal
pedestal: support
altar: structure
throne room: room
strawberry: fruit
orange: fruit
lemon: fruit
fig: fruit
pineapple: fruit
banana: fruit
jackfruit: fruit
custard apple: fruit
pomegranate: fruit
granny smith: apple
bell pepper: vegetable
cucumber: vegetable
artichoke: vegetable
cauliflower: vegetable
broccoli: vegetable
zucchini: vegetable
acorn squash: vegetable
butternut squash: vegetable
spaghetti squash: vegetable
cardoon: vegetable
mushroom: fungus
agaric: fungus
corn: vegetable
cabbage head: vegetable
bagel: bread
pretzel: bread
french loaf: bread
cheeseburger: food
hotdog: food
pizza: food
burrito: food
carbonara: food
meat loaf: food
guacamole: food
consomme: food
trifle: dessert
ice cream: dessert
ice lolly: dessert
chocolate sauce: sauce
dough: food
red wine: drink
espresso: drink
eggnog: drink
cup: dish
daisy: flower
yellow lady's slipper: flower
cardoon flower: flower
rapeseed: plant
corn poppy: flower
buckeye: plant
coral fungus: fungus
earthstar: fungus
hen-of-the-woods: fungus
bolete: fungus
stinkhorn: fungus
alp: mountain
volcano: mountain
cliff: rock formation
promontory: rock formation
sandbar: shore
seashore: shore
lakeside: shore
valley: landscape
coral reef: reef
geyser: spring
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_casegen.py -v`
Expected: PASS, 4 tests. If `test_no_entry_has_coarse_equal_to_concept` fails, fix the offending entry — do not relax the assertion.

- [ ] **Step 5: Commit**

```bash
git add configs/imagenet_coarse.yaml tests/test_casegen.py
git commit -m "feat: a curated superordinate map, because hypernyms do not ground"
```

---

### Task 5: `ragregen/casegen.py` — rank classes and frame prompts

**Files:**
- Create: `ragregen/casegen.py`
- Test: `tests/test_casegen.py` (extend)

**Interfaces:**
- Consumes: `configs/imagenet_coarse.yaml` (Task 4), `corpus_manifest.json` `per_concept` counts (Task 3)
- Produces:
  - `rank_classes(counts: dict[str, int], coarse_map: dict[str, str], *, exclude: set[str], n: int) -> list[str]`
  - `frame_prompt(concept: str, index: int) -> str`
  - `SCENE_FRAMES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_casegen.py`:

```python
import pytest

from ragregen import casegen


def test_ranking_prefers_the_more_common_class():
    counts = {"fox": 500, "axolotl": 3, "violin": 120}
    coarse = {"fox": "canine", "axolotl": "salamander",
              "violin": "string instrument"}
    got = casegen.rank_classes(counts, coarse, exclude=set(), n=2)
    assert got == ["fox", "violin"]


def test_ranking_only_considers_classes_in_the_coarse_map():
    #: A class with no groundable superordinate cannot become a case at all,
    #: so it must not consume one of the 70 slots.
    counts = {"fox": 500, "nematode": 900}
    coarse = {"fox": "canine"}
    assert casegen.rank_classes(counts, coarse, exclude=set(), n=5) == ["fox"]


def test_ranking_excludes_the_bridge_concepts():
    #: The 22 legacy cases are carried forward verbatim; re-deriving one as a
    #: `common` case would duplicate the id and load_dataset would raise.
    counts = {"fox": 500, "violin": 400}
    coarse = {"fox": "canine", "violin": "string instrument"}
    got = casegen.rank_classes(counts, coarse, exclude={"fox"}, n=5)
    assert got == ["violin"]


def test_ranking_breaks_ties_alphabetically():
    #: Determinism matters more than which one wins: the same manifest must
    #: produce the same 70 classes on a re-run.
    counts = {"beta": 10, "alpha": 10}
    coarse = {"alpha": "a", "beta": "b"}
    assert casegen.rank_classes(counts, coarse, exclude=set(),
                                n=2) == ["alpha", "beta"]


def test_ranking_returns_fewer_than_asked_when_the_pool_is_small():
    counts = {"fox": 5}
    assert casegen.rank_classes(counts, {"fox": "canine"}, exclude=set(),
                                n=70) == ["fox"]


def test_a_prompt_mentions_its_concept():
    assert "golden retriever" in casegen.frame_prompt("golden retriever", 0)


def test_prompts_are_deterministic_for_an_index():
    assert (casegen.frame_prompt("fox", 3) == casegen.frame_prompt("fox", 3))


def test_consecutive_indices_use_different_frames():
    #: 70 identical sentence shapes would let the verifier key on the frame
    #: rather than the concept.
    a = casegen.frame_prompt("fox", 0)
    b = casegen.frame_prompt("fox", 1)
    assert a != b


def test_the_frame_pool_is_large_enough_to_vary():
    assert len(casegen.SCENE_FRAMES) >= 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_casegen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragregen.casegen'`

- [ ] **Step 3: Write minimal implementation**

Create `ragregen/casegen.py`:

```python
"""Turn ImageNet classes into cases. Doc 5 §5.

"Common" is measured, not asserted: classes are ranked by how often they
appear in the captions of the corpus we actually downloaded. That makes
commonness a property of the retrieval database rather than an opinion, and it
is honest about what the benchmark tests -- concepts the database knows.
"""
from __future__ import annotations

#: Prompts follow configs/dataset.yaml's house style: a subject doing
#: something somewhere. Varied so the verifier keys on the concept rather than
#: on a single repeated sentence shape.
SCENE_FRAMES: tuple[str, ...] = (
    "a {c} in soft afternoon light",
    "a {c} photographed against a plain background",
    "a {c} resting on a wooden table",
    "a {c} outdoors on an overcast day",
    "a {c} in sharp focus, close up",
    "a {c} standing in an open field",
    "a {c} lit from one side in a quiet room",
    "a {c} seen from a low angle",
    "a {c} on a bright summer morning",
    "a {c} against a dark neutral backdrop",
)


def rank_classes(counts, coarse_map, *, exclude, n: int) -> list[str]:
    """The `n` most common classes that can actually become cases.

    Restricted to `coarse_map`: a class with no groundable superordinate
    cannot pass config.load_dataset, so letting it win a slot would just
    shrink the case set. Ties break alphabetically so the same manifest
    yields the same selection on a re-run.
    """
    pool = [c for c in coarse_map if c not in set(exclude)]
    pool.sort(key=lambda c: (-int(counts.get(c, 0)), c))
    return pool[:n]


def frame_prompt(concept: str, index: int) -> str:
    """A prompt for `concept`, deterministic in `index`."""
    return SCENE_FRAMES[index % len(SCENE_FRAMES)].format(c=concept)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_casegen.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add ragregen/casegen.py tests/test_casegen.py
git commit -m "feat: rank classes by corpus frequency, so 'common' is measured"
```

---

### Task 6: `scripts/make_cases.py` — extract refs and emit the dataset

**Files:**
- Create: `scripts/make_cases.py`
- Test: `tests/test_make_cases.py`

**Interfaces:**
- Consumes: `casegen.rank_classes`, `casegen.frame_prompt` (Task 5); `Case.cohort` (Task 1)
- Produces:
  - `data/gt_refs/<class>/<class>_{0,1,2}.jpg`
  - `configs/dataset_common.yaml` — 92 cases, the 22 bridge merged in verbatim
  - `emit_yaml(entries: list[dict], images_root: Path) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_make_cases.py`:

```python
"""Dataset emission. No parquet, no HuggingFace -- just the YAML we produce.

The output has to survive config.load_dataset unchanged, so these tests round
-trip through it rather than asserting on strings.
"""
import pytest
import yaml

from ragregen import config
from scripts.make_cases import emit_yaml


def _entry(cid, cohort="common", kind="control"):
    return {"id": cid, "prompt": f"a {cid} somewhere", "concept": cid,
            "coarse": "thing", "kind": kind, "cohort": cohort,
            "gt_refs": [f"{cid}/{cid}_0.jpg", f"{cid}/{cid}_1.jpg",
                        f"{cid}/{cid}_2.jpg"]}


def test_emitted_yaml_loads_as_a_dataset(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    ds = config.load_dataset(p)
    assert [c.id for c in ds.cases] == ["fox"]


def test_common_cases_are_labelled_common(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    assert config.load_dataset(p).cases[0].cohort == "common"


def test_bridge_cases_keep_their_cohort(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("durian", cohort="bridge", kind="target")],
                           images_root=tmp_path))
    got = config.load_dataset(p).cases[0]
    assert got.cohort == "bridge"
    assert got.kind == "target"


def test_three_refs_survive_the_heldout_split(tmp_path):
    #: metrics.split_refs with heldout=1 needs >= 2 refs. Emitting 2 would
    #: leave exactly one to edit with and pass; emitting 1 raises at run time,
    #: hours in. Three is the floor this file guarantees.
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    assert len(config.load_dataset(p).cases[0].gt_refs) == 3


def test_a_duplicate_id_is_refused(tmp_path):
    #: A bridge concept re-derived as a common case would collide here.
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox"), _entry("fox")],
                           images_root=tmp_path))
    with pytest.raises(ValueError, match="duplicate"):
        config.load_dataset(p)


def test_images_root_is_written_absolute(tmp_path):
    #: gt_refs resolve against it, and validate.py compares the resolved path
    #: against the corpus root to catch leakage.
    text = emit_yaml([_entry("fox")], images_root=tmp_path)
    assert yaml.safe_load(text)["images_root"] == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_make_cases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.make_cases'`

- [ ] **Step 3: Write the implementation**

Create `scripts/make_cases.py`:

```python
#!/usr/bin/env python
"""ImageNet -> the 92-case set. Doc 5 §5.

ImageNet does two jobs here, both OUTSIDE the pipeline: class names become
prompts, and three val images per class become the held-out answer key. The
pipeline never sees these images -- validate.py:88 refuses to run if they ever
land inside the retrieval corpus.

Usage: ./scripts/run.sh make-cases
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

import yaml  # noqa: E402

from ragregen import casegen, config, env  # noqa: E402

IMAGENET_REPO = "evanarlian/imagenet_1k_resized_256"
IMAGENET_VAL = [
    "data/val-00000-of-00002-b5248be478d25e41.parquet",
    "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet",
]
REFS_PER_CASE = 3


def emit_yaml(entries, images_root: Path) -> str:
    """The dataset YAML. Round-trips through config.load_dataset unchanged."""
    doc = {
        "name": "laion_common_v1",
        "images_root": str(images_root),
        "cases": list(entries),
    }
    header = (
        "# Generated by scripts/make_cases.py -- doc 5 §5. Do not hand-edit;\n"
        "# re-run make-cases instead, or the seed no longer describes the set.\n"
        "#\n"
        "# cohort: bridge = the 22 legacy cases, full-resolution gt_refs.\n"
        "# cohort: common = ImageNet-derived, 256px gt_refs.\n"
        "# The report never averages DINO across the two.\n\n")
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def _class_names(schema) -> list[str]:
    """Label index -> class name, from the parquet's HuggingFace metadata."""
    meta = json.loads(schema.metadata[b"huggingface"].decode())
    feature = meta["info"]["features"]["label"]
    return list(feature["names"])


def _bridge_entries(ds) -> list[dict]:
    """The 22 legacy cases, carried forward verbatim with their own refs."""
    out = []
    for case in ds.cases:
        out.append({
            "id": case.id,
            "prompt": case.prompt,
            "concept": case.concept,
            "coarse": case.coarse,
            "kind": case.kind,
            "cohort": "bridge",
            "gt_refs": [str(p) for p in case.gt_refs],
        })
    return out


def main() -> int:
    env.setup()
    root = env.PROJECT_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=root / "data/laion100k/corpus_manifest.json")
    ap.add_argument("--coarse", type=Path,
                    default=root / "configs/imagenet_coarse.yaml")
    ap.add_argument("--bridge", type=Path,
                    default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--refs-dir", type=Path, default=root / "data/gt_refs")
    ap.add_argument("--out", type=Path,
                    default=root / "configs/dataset_common.yaml")
    ap.add_argument("--n", type=int, default=70)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    counts = json.loads(args.manifest.read_text())["per_concept"]
    coarse_map = yaml.safe_load(args.coarse.read_text())
    bridge = config.load_dataset(args.bridge)
    exclude = {c.concept.lower() for c in bridge.cases}

    chosen = casegen.rank_classes(counts, coarse_map, exclude=exclude,
                                  n=args.n)
    print(f"[make-cases] {len(chosen)} common classes chosen", flush=True)

    wanted = {c: [] for c in chosen}
    for shard in IMAGENET_VAL:
        local = hf_hub_download(IMAGENET_REPO, shard, repo_type="dataset")
        pf = pq.ParquetFile(local)
        names = [n.lower() for n in _class_names(pf.schema_arrow)]
        want_ids = {names.index(c): c for c in chosen if c in names}
        for batch in pf.iter_batches(batch_size=512,
                                     columns=["image", "label"]):
            for img, label in zip(batch.column("image").to_pylist(),
                                  batch.column("label").to_pylist()):
                concept = want_ids.get(int(label))
                if concept is None or len(wanted[concept]) >= REFS_PER_CASE:
                    continue
                wanted[concept].append(img["bytes"])

    entries = _bridge_entries(bridge)
    args.refs_dir.mkdir(parents=True, exist_ok=True)
    for i, concept in enumerate(chosen):
        blobs = wanted[concept]
        if len(blobs) < REFS_PER_CASE:
            #: Fewer than three refs cannot survive the held-out split, so the
            #: case is dropped here rather than raising hours into a run.
            print(f"[make-cases] skipping {concept!r}: only {len(blobs)} refs")
            continue
        cid = concept.replace(" ", "_").replace("-", "_").replace("'", "")
        d = args.refs_dir / cid
        d.mkdir(parents=True, exist_ok=True)
        refs = []
        for j, blob in enumerate(blobs):
            path = d / f"{cid}_{j}.jpg"
            Image.open(io.BytesIO(blob)).convert("RGB").save(path, quality=95)
            refs.append(f"{cid}/{path.name}")
        entries.append({
            "id": cid,
            "prompt": casegen.frame_prompt(concept, i),
            "concept": concept,
            "coarse": coarse_map[concept],
            "kind": "control",
            "cohort": "common",
            "gt_refs": refs,
        })

    args.out.write_text(emit_yaml(entries, images_root=args.refs_dir))
    print(f"[make-cases] {len(entries)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the bridge entries write **absolute** `gt_refs` while common entries write paths relative to `--refs-dir`. `config.load_dataset` does `root / r`, and `Path("/abs") / "/other/abs"` yields the second path, so both resolve correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_make_cases.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_cases.py tests/test_make_cases.py
git commit -m "feat: ImageNet becomes prompts and an answer key, never a corpus"
```

---

### Task 7: corpus-density warning in `validate.py`

A concept the corpus knows nothing about is a finding the report must carry, not a reason to abort. This makes it visible before the GPU is booked.

**Files:**
- Modify: `ragregen/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `corpus_manifest.json` from Task 3
- Produces: `validate_corpus_density(manifest_path: Path, ds: DatasetConfig, k: int) -> list[Problem]`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_validate.py`:

```python
import json

from ragregen import validate


class _Case:
    def __init__(self, cid, concept):
        self.id = cid
        self.concept = concept


class _DS:
    def __init__(self, cases):
        self.cases = cases


def test_a_thin_concept_warns_and_does_not_error(tmp_path):
    #: The run must proceed: "the corpus had nothing here" is a result the
    #: report carries (doc 5 §10). Erroring would abort a booked GPU block.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10,
                             "per_concept": {"axolotl": 1, "fox": 900}}))
    got = validate.validate_corpus_density(
        m, _DS([_Case("axolotl", "axolotl"), _Case("fox", "fox")]), k=3)

    assert [p.severity for p in got] == ["warning"]
    assert got[0].code == "corpus_thin"
    assert "axolotl" in got[0].message


def test_a_dense_corpus_reports_nothing(tmp_path):
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10, "per_concept": {"fox": 900}}))
    assert validate.validate_corpus_density(
        m, _DS([_Case("fox", "fox")]), k=3) == []


def test_a_concept_absent_from_the_manifest_is_treated_as_zero(tmp_path):
    #: Absent means the manifest predates the case, which is worse than thin,
    #: not better.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10, "per_concept": {}}))
    got = validate.validate_corpus_density(m, _DS([_Case("fox", "fox")]), k=3)
    assert got[0].code == "corpus_thin"


def test_a_missing_manifest_warns_rather_than_raising(tmp_path):
    #: The oracle arm needs no corpus at all, so a missing manifest must not
    #: block validate for a run that will never retrieve.
    got = validate.validate_corpus_density(
        tmp_path / "nope.json", _DS([_Case("fox", "fox")]), k=3)
    assert [p.code for p in got] == ["corpus_manifest_missing"]
    assert got[0].severity == "warning"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_validate.py -k corpus -v`
Expected: FAIL with `AttributeError: module 'ragregen.validate' has no attribute 'validate_corpus_density'`

- [ ] **Step 3: Write minimal implementation**

Append to `ragregen/validate.py`:

```python
def validate_corpus_density(manifest_path: Path, ds: DatasetConfig,
                            k: int) -> list[Problem]:
    """Which concepts the corpus cannot serve `k` references for.

    Warnings, never errors. A thin concept is a result the report publishes
    beside its case (doc 5 §4) -- aborting here would trade a known-weak row
    for no row at all, hours after a GPU block was booked.
    """
    import json

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return [_warn(
            "corpus_manifest_missing",
            f"no corpus manifest at {manifest_path}. Retrieval density is "
            f"unknown; run `./scripts/run.sh fetch-corpus manifest`. The "
            f"oracle arm does not need it.",
        )]

    counts = json.loads(manifest_path.read_text()).get("per_concept", {})
    thin = [c.concept for c in ds.cases if int(counts.get(c.concept, 0)) < k]
    if not thin:
        return []
    return [_warn(
        "corpus_thin",
        f"{len(thin)} concept(s) have fewer than k={k} caption matches in the "
        f"corpus: {', '.join(sorted(thin)[:10])}"
        f"{' ...' if len(thin) > 10 else ''}. Retrieval will return "
        f"off-concept references for these; the report records the count.",
    )]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_validate.py -v`
Expected: PASS, including every pre-existing validate test.

- [ ] **Step 5: Commit**

```bash
git add ragregen/validate.py tests/test_validate.py
git commit -m "feat: a thin corpus warns, because an empty concept is a finding"
```

---

### Task 8: the paired delta — `metrics.paired_bootstrap_ci`

The headline. Bootstrap rather than a t-test because n will be under 100 and deltas are likely skewed.

**Files:**
- Modify: `ragregen/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MARGIN: float = -0.02`
  - `paired_bootstrap_ci(deltas, *, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float] | None`
  - `delta_verdict(lo: float, hi: float) -> str` — `"no_harm"` | `"harm"` | `"inconclusive"`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
def test_a_clearly_positive_sample_gives_a_positive_lower_bound():
    deltas = [0.10, 0.12, 0.09, 0.11, 0.13, 0.10, 0.12, 0.11]
    lo, hi = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert lo > 0
    assert hi > lo


def test_the_interval_brackets_the_sample_mean():
    deltas = [0.10, -0.02, 0.04, 0.08, 0.00, 0.06]
    lo, hi = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert lo <= float(np.mean(deltas)) <= hi


def test_the_ci_is_deterministic_for_a_seed():
    #: A headline that moves between report runs is not a headline.
    deltas = [0.05, -0.01, 0.03, 0.00, 0.02]
    a = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=3)
    b = metrics.paired_bootstrap_ci(deltas, n_boot=2000, seed=3)
    assert a == b


def test_an_empty_sample_has_no_interval():
    #: Not (0.0, 0.0). An interval over nothing is not a number, and printing
    #: zeros would read as a measured null result.
    assert metrics.paired_bootstrap_ci([], n_boot=100, seed=0) is None


def test_a_single_case_yields_a_degenerate_interval():
    #: Every resample of one value is that value. Reported honestly rather
    #: than special-cased away.
    lo, hi = metrics.paired_bootstrap_ci([0.07], n_boot=100, seed=0)
    assert lo == pytest.approx(0.07)
    assert hi == pytest.approx(0.07)


def test_the_margin_is_the_value_the_spec_fixed():
    #: Doc 5 §2 fixes -0.02 before any data exists. Changing it after seeing
    #: deltas is the failure this constant prevents.
    assert metrics.MARGIN == -0.02


def test_a_lower_bound_above_the_margin_is_no_harm():
    assert metrics.delta_verdict(-0.01, 0.05) == "no_harm"


def test_an_upper_bound_below_zero_is_harm():
    assert metrics.delta_verdict(-0.09, -0.03) == "harm"


def test_an_interval_spanning_the_margin_is_inconclusive():
    assert metrics.delta_verdict(-0.06, 0.04) == "inconclusive"


def test_a_lower_bound_exactly_on_the_margin_is_no_harm():
    #: The boundary is decided here, not by whoever reads the table.
    assert metrics.delta_verdict(-0.02, 0.01) == "no_harm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_metrics.py -k "bootstrap or margin or verdict or interval" -v`
Expected: FAIL with `AttributeError: module 'ragregen.metrics' has no attribute 'paired_bootstrap_ci'`

- [ ] **Step 3: Write minimal implementation**

Append to `ragregen/metrics.py`:

```python
#: The non-inferiority margin, in DINO cosine, fixed by doc 5 §2 before any
#: data existed. Doc 4's measured cropped DINO was 0.626, so 0.02 is ~3% of
#: the operating point: below it a drop is not distinguishable from crop and
#: reference noise. Choosing this after seeing deltas would make the test
#: meaningless -- do not tune it.
MARGIN = -0.02


def paired_bootstrap_ci(deltas, *, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of per-case deltas.

    Bootstrap rather than a t-test: n is well under 100 and the deltas are
    likely skewed, so no normality is assumed. Seeded, because a headline that
    moves between report runs is not a headline.
    """
    vals = np.asarray([d for d in deltas if d is not None], dtype=np.float64)
    if vals.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def delta_verdict(lo: float, hi: float) -> str:
    """Which of doc 5 §2's three readings this interval supports.

    The boundary lives here rather than in whoever reads the table.
    """
    if lo >= MARGIN:
        return "no_harm"
    if hi < 0:
        return "harm"
    return "inconclusive"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: PASS, including every pre-existing metrics test.

- [ ] **Step 5: Commit**

```bash
git add ragregen/metrics.py tests/test_metrics.py
git commit -m "feat: the paired bootstrap, and a margin fixed before the data"
```

---

### Task 9: score the draft too, and report the delta

`score_case` currently scores only the selected image. The claim needs both sides of the pair, computed identically.

**Files:**
- Modify: `scripts/report.py:82-132` (`score_case`), `:140-160` (`summarise`), `:167-204` (`render`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `metrics.paired_bootstrap_ci`, `metrics.delta_verdict`, `metrics.MARGIN` (Task 8); `Case.cohort` (Task 1)
- Produces: row keys `dino_cropped_draft`, `dino_delta`, `cohort`; summary keys `dino_delta`, `delta_ci`, `delta_verdict`, `improved`, `unchanged`, `worsened`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py`. First extend the existing helpers — `_Case` gains a cohort and `_row` gains the new keys:

```python
class _CohortCase:
    id = "c"
    prompt = "a parrot"
    kind = "control"
    cohort = "common"


class _TwoEnc:
    """Draft and output embed differently, so a delta is actually visible."""

    def __init__(self):
        self.seen = []

    def embed(self, image):
        #: Keyed on the image's flat colour: the fixture paints the draft and
        #: the output different shades on purpose.
        px = image.convert("RGB").getpixel((0, 0))[0]
        return (np.array([1.0, 0.0], dtype=np.float32) if px > 100
                else np.array([0.6, 0.8], dtype=np.float32))

    def encode_pil(self, images, batch_size=32):
        return np.array([[1.0, 0.0]] * len(list(images)), dtype=np.float32)

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(list(texts)), dtype=np.float32)

    def free(self):
        pass


class _TwoHolder:
    def __init__(self):
        self.dino = _TwoEnc()
        self.clip = _Enc()
        self.siglip = _Enc()


def test_a_repaired_case_reports_both_sides_of_the_pair(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(
        {"draft": [False, 0.0], "attempt_1": [True, 1.0]}))
    #: bright output, dark draft, bright reference
    Image.new("RGB", (64, 64), (200, 200, 200)).save(d / "attempt_1.png")
    a = np.zeros((64, 64), dtype=np.uint8)
    a[8:56, 8:56] = 255
    Image.fromarray(a, "L").save(d / "mask.png")
    Image.new("RGB", (64, 64), (200, 200, 200)).save(ref)

    got = score_case(_CohortCase(), d,
                     Image.new("RGB", (64, 64), (10, 10, 10)),
                     _TwoHolder(), heldout_paths=[ref])

    assert got["dino_cropped"] == pytest.approx(1.0)
    assert got["dino_cropped_draft"] == pytest.approx(0.6)
    assert got["dino_delta"] == pytest.approx(0.4)
    assert got["cohort"] == "common"


def test_a_healthy_case_has_a_delta_of_exactly_zero(tmp_path):
    #: `best` is the draft, so output and draft are the same image. Zero by
    #: construction, and excluded from the delta mean rather than diluting it.
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [True, 1.0]}, with_mask=False, ref=ref)

    got = score_case(_CohortCase(), d,
                     Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref])

    assert got["dino_delta"] == 0.0
    assert got["dino_cropped_draft"] is None


def _drow(kind, cohort, group, dino=None, draft=None, delta=None,
          selected="attempt_1"):
    return {"case_id": "x", "kind": kind, "cohort": cohort, "bucket": group,
            "selected": selected, "dino_cropped": dino,
            "dino_cropped_draft": draft, "dino_delta": delta,
            "dino_whole": None, "preservation": 1.0, "clip": 0.3,
            "siglip": 0.4}


def test_strata_split_on_kind_and_cohort_together():
    #: A 256px-ref case and a full-res one must never share a DINO mean.
    rows = [_drow("control", "common", "repaired", dino=0.8, draft=0.7,
                  delta=0.1),
            _drow("control", "bridge", "repaired", dino=0.5, draft=0.4,
                  delta=0.1)]
    got = summarise(rows)
    assert set(got) == {"control/common", "control/bridge"}
    assert got["control/common"]["dino_cropped"] == pytest.approx(0.8)


def test_healthy_cases_are_excluded_from_the_delta_mean():
    #: Averaging them in dilutes a real effect toward zero and flatters the
    #: method -- the same reasoning doc 4 §4 applies to repair rate.
    rows = [_drow("control", "common", "healthy", delta=0.0,
                  selected="draft"),
            _drow("control", "common", "repaired", dino=0.8, draft=0.6,
                  delta=0.2),
            _drow("control", "common", "unrepairable", dino=0.4, draft=0.5,
                  delta=-0.1, selected="draft")]
    got = summarise(rows)["control/common"]
    #: mean over the two that entered repair: (0.2 + -0.1) / 2
    assert got["dino_delta"] == pytest.approx(0.05)


def test_the_three_counts_partition_the_repair_cases():
    rows = [_drow("control", "common", "repaired", delta=0.2),
            _drow("control", "common", "unrepairable", delta=-0.1),
            _drow("control", "common", "unrepairable", delta=0.0,
                  selected="draft")]
    got = summarise(rows)["control/common"]
    assert (got["improved"], got["worsened"], got["unchanged"]) == (1, 1, 1)


def test_unchanged_is_decided_by_best_not_by_a_threshold():
    #: `selected == "draft"` means the pipeline declined to act. A tiny
    #: non-zero delta on a real edit is still an edit.
    rows = [_drow("control", "common", "unrepairable", delta=0.0001,
                  selected="draft")]
    got = summarise(rows)["control/common"]
    assert got["unchanged"] == 1
    assert got["improved"] == 0


def test_a_stratum_with_no_repairs_has_no_interval():
    rows = [_drow("control", "common", "healthy", delta=0.0,
                  selected="draft")]
    got = summarise(rows)["control/common"]
    assert got["delta_ci"] is None
    assert got["delta_verdict"] is None


def test_render_names_the_margin_and_the_verdict():
    summary = summarise([_drow("control", "common", "repaired", dino=0.8,
                               draft=0.6, delta=0.2)])
    text = render(summary, {"run": "r", "arm": "full", "mechanism": "inpaint",
                            "heldout_refs": 1})
    assert "-0.02" in text or "−0.02" in text
    assert "control/common" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_report.py -k "pair or delta or cohort or strata or counts or margin or unchanged" -v`
Expected: FAIL — `KeyError: 'dino_cropped_draft'` and `KeyError: 'control/common'`

- [ ] **Step 3: Score the draft in `score_case`**

In `scripts/report.py`, add `metrics.MARGIN` usage later; first change `score_case`. Replace the `row = {...}` literal with:

```python
    row = {
        "case_id": case.id,
        "kind": case.kind,
        "cohort": getattr(case, "cohort", "bridge"),
        "bucket": group,
        "selected": label,
        "dino_cropped": None,
        "dino_cropped_draft": None,
        "dino_delta": None,
        "dino_whole": None,
        "preservation": None,
        "clip": metrics.prompt_alignment(output, case.prompt, encoders.clip),
        "siglip": metrics.prompt_alignment(output, case.prompt,
                                           encoders.siglip),
    }
```

and in the no-mask branch, before `return row`:

```python
        row["dino_whole"] = metrics.dino_identity(output, refs, encoders.dino)
        #: `best` is the draft here, so the pair is the same image twice.
        #: Zero by construction, and summarise excludes it from the mean.
        row["dino_delta"] = 0.0
        return row
```

Then replace the cropped-DINO block at the end with:

```python
    mask = Image.open(mask_path).convert("L")
    alpha = (np.array(mask).astype(np.float32) / 255.0 > 0.5).astype(
        np.float32)
    #: The SAME mask and the SAME held-out refs on both sides. Any divergence
    #: here makes the headline meaningless while still printing a plausible
    #: number (doc 5 §7), which is why it is pinned by a test.
    ref_crops = [metrics.crop_to_mask(r, mask) if r.size == mask.size else r
                 for r in refs]
    row["dino_cropped"] = metrics.dino_identity(
        metrics.crop_to_mask(output, mask), ref_crops, encoders.dino)
    row["dino_cropped_draft"] = metrics.dino_identity(
        metrics.crop_to_mask(draft, mask), ref_crops, encoders.dino)
    row["dino_delta"] = row["dino_cropped"] - row["dino_cropped_draft"]
    row["preservation"] = metrics.preservation(draft, output, alpha)["score"]
    return row
```

- [ ] **Step 4: Group by `(kind, cohort)` and add the delta columns**

Replace `summarise` in `scripts/report.py`:

```python
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
            "healthy": sum(1 for r in group if r["bucket"] == "healthy"),
            "repaired": sum(1 for r in group if r["bucket"] == "repaired"),
            "unrepairable": sum(1 for r in group
                                if r["bucket"] == "unrepairable"),
            "repair_rate": repair_rate([r["bucket"] for r in group]),
            "dino_cropped": _mean(r["dino_cropped"] for r in group),
            "dino_cropped_draft": _mean(r["dino_cropped_draft"]
                                        for r in group),
            "dino_delta": _mean(deltas),
            "delta_ci": ci,
            "delta_verdict": metrics.delta_verdict(*ci) if ci else None,
            #: `unchanged` is `best == "draft"` -- the pipeline declined to
            #: act -- not a threshold on the delta. Keeps the counts
            #: independent of MARGIN.
            "unchanged": sum(1 for r in repaired if r["selected"] == "draft"),
            "improved": sum(1 for r in repaired
                            if r["selected"] != "draft"
                            and (r["dino_delta"] or 0) > 0),
            "worsened": sum(1 for r in repaired
                            if r["selected"] != "draft"
                            and (r["dino_delta"] or 0) < 0),
            "dino_whole": _mean(r["dino_whole"] for r in group),
            "preservation": _mean(r["preservation"] for r in group),
            "clip": _mean(r["clip"] for r in group),
            "siglip": _mean(r["siglip"] for r in group),
        }
    return out
```

- [ ] **Step 5: Render the new table**

Replace `render` in `scripts/report.py`:

```python
def render(summary: dict, meta: dict) -> str:
    """The human-readable table, with its own caveats attached."""
    lines = [
        f"# Results — {meta['run']}",
        "",
        f"Arm: **{meta['arm']}**  ·  mechanism: **{meta['mechanism']}**  ·  "
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
        ci = ("—" if s["delta_ci"] is None
              else f"[{s['delta_ci'][0]:+.3f}, {s['delta_ci'][1]:+.3f}]")
        delta = "—" if s["dino_delta"] is None else f"{s['dino_delta']:+.3f}"
        lines.append(
            f"| {name} | {s['n']} | {s['healthy']} | {s['repaired']} | "
            f"{s['unrepairable']} | {rate} | {_fmt(s['dino_cropped_draft'])} "
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
        "- **The margin was fixed before any data existed** (doc 5 §2). It is "
        "not tuned to the result.",
        "- **Healthy cases are excluded from the delta.** Their delta is zero "
        "by construction — `best` is the draft — and averaging them in would "
        "dilute a real effect toward zero.",
        "- **`+/=/−` counts repair cases only.** `=` means the pipeline chose "
        "the draft; it is decided by `best`, not by a threshold on the delta.",
        "- **Strata never merge.** `bridge` cases carry full-resolution "
        "references and `common` cases 256px ones; one mean across both would "
        "read a resolution difference as identity.",
        "- **Read preservation next to the delta.** Identity bought by "
        "repainting the canvas is not a repair.",
        "- **DINO (whole) is not comparable to DINO (crop).** It covers "
        "healthy cases, which never ran `mask` and so have no box to crop to.",
        "- **Verifier pass-rate is absent on purpose.** The pipeline "
        "optimises against the verifier, so it is a development signal only.",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 6: Attach per-concept corpus density to each row**

Spec §6 requires the density beside each case, so a weak row can be checked against whether the corpus had anything to offer. Add to `tests/test_report.py`:

```python
def test_density_is_attached_per_case(tmp_path):
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"per_concept": {"fox": 420, "axolotl": 1}}))

    class _C:
        def __init__(self, cid, concept):
            self.id, self.concept = cid, concept

    rows = [{"case_id": "fox"}, {"case_id": "axolotl"}]
    attach_density(rows, m, [_C("fox", "fox"), _C("axolotl", "axolotl")])
    assert rows[0]["corpus_density"] == 420
    assert rows[1]["corpus_density"] == 1


def test_density_is_none_when_the_manifest_is_absent(tmp_path):
    #: None means unknown. Zero would claim the corpus was searched and found
    #: empty, which is a different and much stronger statement.
    class _C:
        id, concept = "fox", "fox"

    rows = [{"case_id": "fox"}]
    attach_density(rows, tmp_path / "nope.json", [_C()])
    assert rows[0]["corpus_density"] is None
```

Add `attach_density` to the import at the top of the test file, then implement in `scripts/report.py`:

```python
def attach_density(rows, manifest_path, cases):
    """Per-concept corpus counts, recorded beside each case's metrics.

    None means unknown -- no manifest -- which is not the same claim as zero
    ("searched, and the corpus had nothing"). Doc 5 §6 keeps this per-case so
    a weak row can be attributed to corpus sparsity rather than guessed at.
    """
    counts = {}
    manifest_path = Path(manifest_path)
    if manifest_path.is_file():
        counts = json.loads(manifest_path.read_text()).get("per_concept", {})
    by_id = {c.id: c.concept for c in cases}
    for row in rows:
        row["corpus_density"] = counts.get(by_id.get(row["case_id"], ""))
    return rows
```

and call it in `main()`, after the `finally: encoders.free()` block and before `summary = summarise(rows)`. `config` is already imported on `main()`'s first line:

```python
    db = config.load_retrieval_db()
    attach_density(rows, db.index_path.parent / "corpus_manifest.json",
                   ds.cases)
```

- [ ] **Step 7: Run the tests**

Run: `$PY -m pytest tests/test_report.py -v`
Expected: PASS. The pre-existing `test_summary_splits_targets_from_controls` will fail on the new key names — update it to `got["target/bridge"]` and `got["control/bridge"]`, since `_row` omits `cohort` and the default is `bridge`.

- [ ] **Step 8: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green, gpu-marked tests deselected.

- [ ] **Step 9: Commit**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat: the paired delta, scored on the same mask and the same refs"
```

---

### Task 10: VRAM preflight and the heartbeat

`stage_with_model` loads weights and discovers mid-allocation that a neighbour took the card. Yielding on purpose is cleaner than being evicted, and a 15-hour run needs a log line that distinguishes waiting from hung.

**Files:**
- Modify: `ragregen/env.py`, `ragregen/schedule.py:147-172` (`run_stage`), `scripts/run_pipeline.py:52-69` (`stage_with_model`)
- Test: `tests/test_env.py`, `tests/test_schedule.py`

**Interfaces:**
- Consumes: `schedule.StageAborted`
- Produces:
  - `env.free_vram_gb(device_index: int = 0) -> float | None`
  - `run_pipeline.STAGE_VRAM_GB: dict[str, float]`
  - `stage_with_model(..., need_gb: float | None = None)`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_env.py`:

```python
def test_free_vram_is_none_without_a_card(monkeypatch):
    #: None means "unknown", and the preflight must treat unknown as
    #: permission to proceed -- otherwise every CPU test aborts.
    import ragregen.env as e
    monkeypatch.setattr(e, "_cuda_mem_get_info", lambda i: None)
    assert e.free_vram_gb(0) is None


def test_free_vram_converts_bytes_to_gigabytes(monkeypatch):
    import ragregen.env as e
    monkeypatch.setattr(e, "_cuda_mem_get_info",
                        lambda i: (8 * 1024 ** 3, 24 * 1024 ** 3))
    assert e.free_vram_gb(0) == pytest.approx(8.0)
```

Add to `tests/test_schedule.py`:

```python
def test_the_heartbeat_names_every_completed_case(capsys):
    #: A stand-in rather than Queue.open, so the test needs no temp directory;
    #: run_stage only calls pending(), mark() and save().
    class _Q:
        def __init__(self):
            self.done = []

        def pending(self, stage, attempt=None):
            return ["a", "b"]

        def mark(self, cid, stage, state, attempt=None, **kw):
            self.done.append(cid)

        def save(self):
            pass

    queue = _Q()
    schedule.run_stage(queue, "verify", lambda cid: None)
    out = capsys.readouterr().out
    #: `tail -f` on a 15-hour run must distinguish "waiting for the card"
    #: from "hung". One line per case is that signal.
    assert "verify" in out and "a" in out and "b" in out
```

Add to `tests/test_run_pipeline.py`:

```python
def test_a_short_card_aborts_before_any_weights_load(monkeypatch):
    import scripts.run_pipeline as rp

    loaded = []

    class _Q:
        def pending(self, stage, attempt=None):
            return ["a"]

    monkeypatch.setattr(rp.env, "free_vram_gb", lambda i=0: 2.0)
    #: The whole point: `load` must never be called. An OOM partway through
    #: loading 16 GB of Qwen leaves the allocator worse off than a clean yield.
    with pytest.raises(rp.schedule.StageAborted, match="VRAM"):
        rp.stage_with_model(_Q(), "verify", lambda: loaded.append(1),
                            lambda m, cid: None, need_gb=17.0)
    assert loaded == []


def test_an_unknown_card_is_treated_as_permission_to_proceed(monkeypatch):
    #: free_vram_gb returns None on a CPU-only machine. Blocking there would
    #: make every CPU test abort.
    import scripts.run_pipeline as rp

    ran = []

    class _Q:
        def pending(self, stage, attempt=None):
            return ["a"]

        def mark(self, *a, **k):
            pass

        def save(self):
            pass

    monkeypatch.setattr(rp.env, "free_vram_gb", lambda i=0: None)
    rp.stage_with_model(_Q(), "verify", lambda: "model",
                        lambda m, cid: ran.append(cid), need_gb=17.0)
    assert ran == ["a"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_env.py tests/test_schedule.py tests/test_run_pipeline.py -k "vram or heartbeat or card" -v`
Expected: FAIL with `AttributeError: module 'ragregen.env' has no attribute 'free_vram_gb'`

- [ ] **Step 3: Add `free_vram_gb` to `ragregen/env.py`**

```python
def _cuda_mem_get_info(device_index: int):
    """(free, total) bytes, or None when there is no usable card.

    Split out so tests can substitute it without a CUDA context.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info(device_index)
    except (RuntimeError, AssertionError):
        return None


def free_vram_gb(device_index: int = 0) -> float | None:
    """Free VRAM in GB, or None when it cannot be read.

    None means unknown, and every caller must treat unknown as permission to
    proceed -- blocking on it would abort every CPU-only run.
    """
    info = _cuda_mem_get_info(device_index)
    if info is None:
        return None
    return float(info[0]) / (1024 ** 3)
```

- [ ] **Step 4: Add the heartbeat to `ragregen/schedule.py`**

In `run_stage`, replace the final `queue.mark(...)` line with:

```python
        queue.mark(case_id, stage, "done", attempt=attempt)
        #: One line per completed case. On a 15-hour run this is how `tail -f`
        #: distinguishes "waiting for the card" from "hung" (doc 5 §9).
        suffix = "" if attempt is None else f" attempt {attempt}"
        print(f"[{time.strftime('%H:%M:%S')}] {stage}{suffix}: {case_id} done",
              flush=True)
```

and add `import time` at the top of the module beside `import traceback`.

- [ ] **Step 5: Add the preflight to `scripts/run_pipeline.py`**

Add near the top, after the imports:

```python
#: Peak VRAM per stage, from spec §4's measured figures: Qwen-7B ~16 GB,
#: FLUX-nf4 ~12 GB, GroundingDINO+SAM ~6 GB, SigLIP ~4 GB. Headroom included,
#: because a stage that fits exactly is a stage that OOMs on fragmentation.
STAGE_VRAM_GB = {
    "verify": 17.0,
    "regen": 13.0,
    "mask": 6.0,
    "retrieve": 4.0,
}
```

Replace `stage_with_model`:

```python
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
    try:
        schedule.run_stage(queue, stage, lambda cid: work(model, cid),
                           attempt=attempt)
    finally:
        free_fn = getattr(model, "free", None)
        if callable(free_fn):
            free_fn()
```

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest tests/test_env.py tests/test_schedule.py tests/test_run_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add ragregen/env.py ragregen/schedule.py scripts/run_pipeline.py \
        tests/test_env.py tests/test_schedule.py tests/test_run_pipeline.py
git commit -m "feat: yield the card on purpose, and say so once per case"
```

---

### Task 11: `scripts/supervise.sh` — re-acquire and resume

`gpu_wait.sh` guards only the initial launch. Over 15 hours the dominant failure is an eviction at 3 a.m. that sits idle until a human notices.

**Files:**
- Create: `scripts/supervise.sh`
- Test: `tests/test_supervise.py`

**Interfaces:**
- Consumes: `gpu_wait.sh`, `run.sh pipeline --resume`, exit code 2 from `StageAborted`
- Produces: `scripts/supervise.sh --run <dir> [--max-retries N] [--deadline S] -- <command>`

- [ ] **Step 1: Write the failing test**

Create `tests/test_supervise.py`:

```python
"""The supervisor loop, driven with fake commands. No GPU, no pipeline.

Exit 2 is StageAborted -- the card was taken, the queue is checkpointed, and
resuming is correct. Every other non-zero code is a real failure and must not
be retried forever.
"""
import subprocess
from pathlib import Path

SUPERVISE = Path("scripts/supervise.sh").resolve()


def _run(tmp_path, script_body, *, max_retries=3):
    fake = tmp_path / "fake.sh"
    fake.write_text("#!/usr/bin/env bash\n" + script_body)
    fake.chmod(0o755)
    return subprocess.run(
        [str(SUPERVISE), "--max-retries", str(max_retries),
         "--interval", "0", "--no-gpu-wait", "--", str(fake)],
        capture_output=True, text=True, timeout=60)


def test_a_clean_exit_stops_immediately(tmp_path):
    got = _run(tmp_path, 'echo run >> "$0.log"\nexit 0\n')
    assert got.returncode == 0
    assert Path(f"{tmp_path / 'fake.sh'}.log").read_text().count("run") == 1


def test_exit_two_is_retried(tmp_path):
    #: The counter file makes the fake fail twice then succeed, which is what
    #: a contended card looks like.
    got = _run(tmp_path, f'''
n=$(cat "{tmp_path}/n" 2>/dev/null || echo 0)
echo $((n + 1)) > "{tmp_path}/n"
[ "$n" -ge 2 ] && exit 0
exit 2
''')
    assert got.returncode == 0
    assert (tmp_path / "n").read_text().strip() == "3"


def test_a_real_failure_is_not_retried(tmp_path):
    #: Exit 1 is a bug, not an eviction. Retrying it would loop on a
    #: deterministic crash and burn the whole booked block.
    got = _run(tmp_path, f'''
n=$(cat "{tmp_path}/n" 2>/dev/null || echo 0)
echo $((n + 1)) > "{tmp_path}/n"
exit 1
''')
    assert got.returncode == 1
    assert (tmp_path / "n").read_text().strip() == "1"


def test_retries_are_bounded(tmp_path):
    got = _run(tmp_path, "exit 2\n", max_retries=2)
    assert got.returncode != 0
    assert "max-retries" in (got.stdout + got.stderr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_supervise.py -v`
Expected: FAIL — `scripts/supervise.sh` does not exist.

- [ ] **Step 3: Write the implementation**

Create `scripts/supervise.sh`:

```bash
#!/usr/bin/env bash
# Keep a long pipeline run alive across GPU evictions.
#
# gpu_wait.sh guards only the initial launch. Over a 15-hour run the dominant
# failure is a neighbour landing at 3am: the pipeline checkpoints correctly and
# exits 2, then sits idle until a human notices. This turns "the run died" into
# "the run paused" (doc 5 §9).
#
# Exit 2 is schedule.StageAborted -- queue.json is saved and resuming is
# correct. Any other non-zero code is a real failure and is NOT retried, or a
# deterministic crash would loop until the deadline.
#
# Usage:
#   ./scripts/supervise.sh [options] -- <command> [args...]
#
#   --max-retries N   give up after N evictions       (default 40)
#   --interval S      seconds between attempts        (default 120)
#   --deadline S      total wall-clock budget, 0=none (default 0)
#   --need GB         passed to gpu_wait.sh           (default 17)
#   --no-gpu-wait     run the command directly (tests)
#   --log FILE        tee output here
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_RETRIES=40
INTERVAL=120
DEADLINE=0
NEED_GB=17
USE_GPU_WAIT=1
LOG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --interval)    INTERVAL="$2"; shift 2 ;;
    --deadline)    DEADLINE="$2"; shift 2 ;;
    --need)        NEED_GB="$2"; shift 2 ;;
    --no-gpu-wait) USE_GPU_WAIT=0; shift ;;
    --log)         LOG="$2"; shift 2 ;;
    --)            shift; break ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *)             echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ $# -ge 1 ] || { echo "error: no command after --" >&2; exit 2; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-logs/supervise_${TS}.log}"
mkdir -p "$(dirname "$LOG")"

say() { printf '[supervise %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

started=$(date +%s)
attempt=0

while :; do
  attempt=$((attempt + 1))
  say "attempt ${attempt}/${MAX_RETRIES}: $*"

  if [ "$USE_GPU_WAIT" -eq 1 ]; then
    ./scripts/gpu_wait.sh --need "$NEED_GB" --log "$LOG" -- "$@"
  else
    "$@"
  fi
  rc=$?

  if [ "$rc" -eq 0 ]; then
    say "command finished cleanly after ${attempt} attempt(s)"
    exit 0
  fi

  # Anything other than StageAborted is a bug, not an eviction. Retrying a
  # deterministic crash would burn the whole booked block.
  if [ "$rc" -ne 2 ]; then
    say "command exited ${rc} -- not an eviction, not retrying"
    exit "$rc"
  fi

  say "evicted (exit 2); queue.json is checkpointed"

  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    say "giving up: --max-retries ${MAX_RETRIES} reached"
    exit 3
  fi

  if [ "$DEADLINE" -gt 0 ]; then
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -ge "$DEADLINE" ]; then
      say "giving up: deadline ${DEADLINE}s reached after ${elapsed}s"
      exit 4
    fi
  fi

  [ "$INTERVAL" -gt 0 ] && sleep "$INTERVAL"
done
```

- [ ] **Step 4: Make it executable and run the tests**

```bash
chmod +x scripts/supervise.sh
$PY -m pytest tests/test_supervise.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/supervise.sh tests/test_supervise.py
git commit -m "feat: an eviction pauses the run instead of ending it"
```

---

### Task 12: wire the operator surface

**Files:**
- Modify: `scripts/run.sh:34-42` (the verb table and usage), `ragregen/validate.py:142-158` (`validate_all`), `docs/RUNBOOK.md`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: everything above
- Produces: `run.sh {fetch-corpus|make-cases|supervise}`; `validate_all` includes corpus density

- [ ] **Step 1: Write the failing test**

`scripts/validate_cli.py` calls `validate.validate_all(ds, db, pipe)` and prints whatever it returns, so the check belongs in `validate_all` — the CLI needs no edit at all. Add to `tests/test_validate.py`:

```python
def test_validate_all_includes_corpus_density(tmp_path, monkeypatch):
    #: The operator must learn a concept is unservable BEFORE booking a GPU
    #: block, not nine hours into one. validate_cli prints whatever
    #: validate_all returns, so wiring it here is the whole change.
    called = {}

    def _fake(manifest_path, ds, k):
        called["manifest"] = manifest_path
        called["k"] = k
        return [validate._warn("corpus_thin", "axolotl is thin")]

    monkeypatch.setattr(validate, "validate_corpus_density", _fake)
    monkeypatch.setattr(validate, "validate_dataset", lambda ds: [])
    monkeypatch.setattr(validate, "validate_leakage", lambda ds, db: [])
    monkeypatch.setattr(validate, "validate_db", lambda db, expected_dim: [])
    monkeypatch.setattr(validate, "validate_disk", lambda: [])

    db = config.RetrievalDBConfig(
        name="c", images_root=tmp_path / "images",
        index_path=tmp_path / "corpus" / "index.faiss",
        encoder="siglip_so400m_384")
    pipe = config.load_pipeline(Path("configs/pipeline.yaml"))

    got = validate.validate_all(_DS([_Case("fox", "fox")]), db, pipe)

    assert [p.code for p in got] == ["corpus_thin"]
    #: The manifest lives beside the index, so one config key locates both.
    assert called["manifest"] == tmp_path / "corpus" / "corpus_manifest.json"
    assert called["k"] == pipe.retry_budget
```

This needs `from pathlib import Path` and `from ragregen import config, validate` at the top of the file if they are not already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_validate.py -k validate_all_includes -v`
Expected: FAIL — `validate_all` never calls `validate_corpus_density`, so `got` is empty.

- [ ] **Step 3: Add the verbs to `scripts/run.sh`**

In the `usage()` heredoc, after the `build-index` line:

```
  fetch-corpus build the 100k LAION corpus     (no GPU, ~3h network)
  make-cases   ImageNet -> the 92-case set     (no GPU, ~10 min)
  supervise    keep a long run alive           (wraps pipeline)
```

In the `case "$STAGE"` table, after `build-index)`:

```bash
  fetch-corpus) exec "$PY" scripts/fetch_corpus.py "$@" ;;
  make-cases)  exec "$PY" scripts/make_cases.py "$@" ;;
  supervise)   exec ./scripts/supervise.sh "$@" ;;
```

- [ ] **Step 4: Call the density check from `validate_all`**

In `ragregen/validate.py`, inside `validate_all`, after the `validate_disk()` line:

```python
    #: The manifest lives beside the index, so retrieval_db.yaml's one path
    #: key locates both. A missing manifest warns rather than raising -- the
    #: oracle arm needs no corpus at all.
    problems += validate_corpus_density(
        db.index_path.parent / "corpus_manifest.json", ds, pipe.retry_budget)
```

`scripts/validate_cli.py` needs no change: it already prints everything `validate_all` returns and exits 2 only on errors, so a `corpus_thin` warning reports without blocking.

- [ ] **Step 5: Document the campaign in `docs/RUNBOOK.md`**

Add a section after §3, and update the status banner at the top — `fetch-corpus`, `make-cases` and `supervise` are now implemented:

```markdown
## 3.3 The common-concept benchmark (doc 5)

Answers: does the pipeline damage images of concepts FLUX already gets right?

Two datasets, two jobs, and they never touch. **LAION is the library the
system searches.** **ImageNet asks the questions and marks the answers** — the
pipeline never sees an ImageNet image. `validate` refuses to run if they
overlap.

```bash
./scripts/run.sh fetch-corpus select        # ~5 min, seeded and reproducible
./scripts/run.sh fetch-corpus download      # ~2-4h, network-bound, resumable
./scripts/run.sh fetch-corpus manifest      # ~2 min, reports what landed
./scripts/run.sh make-cases                 # ~10 min, writes dataset_common.yaml
./scripts/run.sh validate --dataset configs/dataset_common.yaml
./scripts/run.sh build-index                # ~20 min GPU
./scripts/run.sh screen --dataset configs/dataset_common.yaml   # ~2.3h GPU
```

Then the dry-run, which costs 2 hours and de-risks 15:

```bash
./scripts/run.sh supervise -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism stitch \
    --tag doc5dry
```

`stitch` pastes the reference in pixel space and needs no FLUX weights, so
this exercises retrieval, masking, scoring and the report end to end — code
that has never executed — without booking the card for a full day. Read its
report before continuing.

Then the real run:

```bash
./scripts/run.sh supervise -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism inpaint \
    --tag doc5
./scripts/run.sh report --run outputs/doc5_<TS> \
    --dataset configs/dataset_common.yaml
```

`supervise` re-acquires the card and resumes after an eviction. Exit 2 means
the card was taken and the queue is checkpointed; any other code is a real
failure and stops the loop.

**Reading the result.** The delta column is the claim: `DINO(best) −
DINO(draft)`, paired within each case. `no_harm` means the CI's lower bound
clears −0.02, a margin fixed before any data existed. A `harm` verdict is a
publishable negative, not a bug to fix.
```

- [ ] **Step 6: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green.

- [ ] **Step 7: Smoke the new verbs**

```bash
./scripts/run.sh 2>&1 | head -20
./scripts/run.sh fetch-corpus --help
./scripts/run.sh make-cases --help
```

Expected: the usage block lists the three new verbs; both `--help` calls print their arguments and exit 0.

- [ ] **Step 8: Commit**

```bash
git add scripts/run.sh ragregen/validate.py docs/RUNBOOK.md tests/test_validate.py
git commit -m "feat: the benchmark is three run.sh verbs and a runbook section"
```

---

### Task 13: run the campaign

Everything above is code. This is the experiment, and it is the first time the `full` arm touches real data.

**Files:**
- Create: `docs/findings/2026-08-05-common-concept-result.md`

**Interfaces:**
- Consumes: every task above
- Produces: `outputs/doc5dry_<TS>/report.md`, `outputs/doc5_<TS>/report.md`, and the findings document

- [ ] **Step 1: Build the corpus**

```bash
./scripts/run.sh fetch-corpus select
./scripts/run.sh fetch-corpus download
./scripts/run.sh fetch-corpus manifest
```

Expected: `data/laion100k/corpus_manifest.json` reports `n_images` near 100k. If it is far below, re-run `select` with a larger `--n-random` rather than accepting a thin corpus silently. Check `df -h .` first — the volume is at 97% and shared.

- [ ] **Step 2: Build the case set and validate**

```bash
./scripts/run.sh make-cases
./scripts/run.sh validate --dataset configs/dataset_common.yaml
```

Expected: 92 cases (or fewer, with each drop named). `validate` must report **zero errors**. A `corpus_thin` warning is expected for some concepts and is a result, not a blocker. A `gt_ref_in_corpus` **error** means ImageNet landed inside the corpus — stop and fix the paths; do not relax the check.

- [ ] **Step 3: Build the index, then spot-check for contamination**

```bash
nvidia-smi
./scripts/run.sh build-index
```

Expected: `~100000 vectors, dim=1152`. If `nvidia-smi` shows under 5 GB free, wait — do not lower the batch size to squeeze in.

Then spec §12 R1's mitigation. `validate.py:88` compares directories, not pixels, so it cannot catch a LAION image that happens to be a near-duplicate of a held-out reference. This embeds each held-out ref and asks the index for its nearest corpus neighbour. **Reported, never enforced** — a threshold that silently dropped cases would be worse than a number with a caveat.

```bash
HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache \
/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python - <<'EOF'
import json
from pathlib import Path
from ragregen import config, encoders, metrics, retrieve

ds = config.load_dataset(Path("configs/dataset_common.yaml"))
db = config.load_retrieval_db()
pipe = config.load_pipeline()
enc = encoders.build_encoder(db.encoder, device="cuda")
r = retrieve.Retriever.from_index(db.index_path, enc)

rows = []
for case in ds.cases:
    _, held = metrics.split_refs(case.gt_refs, pipe.eval_heldout_refs)
    vec = retrieve._l2_normalise(enc.encode_images(held))
    scores, _ = r.index.search(vec, 1)
    rows.append({"case": case.id, "max_sim": float(scores.max())})

rows.sort(key=lambda x: -x["max_sim"])
Path("data/laion100k/contamination.json").write_text(json.dumps(rows, indent=2))
print(f"highest cosine to any corpus image: {rows[0]['max_sim']:.3f} "
      f"({rows[0]['case']})")
print("cases above 0.95:",
      [x["case"] for x in rows if x["max_sim"] > 0.95] or "none")
EOF
```

Record the highest similarity in the findings whatever it is. A handful above 0.95 out of ~280 references is the expected order (spec §4); a large fraction means the seeded slice pulled the answer key in and the corpus needs rebuilding with a different seed.

- [ ] **Step 4: Draft the 70 new cases**

```bash
./scripts/gpu_wait.sh -- ./scripts/run.sh screen --dataset configs/dataset_common.yaml
```

Expected: ~2.3 h; a `draft.png` for every common case. The 22 bridge cases already have drafts in `outputs/screen_latest/`. `labels.csv` is written and deliberately left unlabelled for the new cohort — the paired delta needs no human verdict (doc 5 §5).

- [ ] **Step 5: The dry-run**

```bash
./scripts/run.sh supervise -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism stitch \
    --tag doc5dry
./scripts/run.sh report --run outputs/doc5dry_<TS> \
    --dataset configs/dataset_common.yaml
```

Expected, and each of these is a gate:

- Retrieval returned hits for most cases. Mass empty `refs.json` means the index or the encoder is wrong — stop.
- `preservation` is exactly **1.000** for `stitch`. Not 0.999: the compositor guarantees bit-identical pixels outside the mask, and anything less means it changed.
- Strata appear as `control/common`, `control/bridge`, `target/bridge` — never merged.
- The false-alarm rate (metric A, doc 5 §6) is readable from the `healthy` column of `control/common`.

**If almost every common case is `healthy`, stop and read doc 5 §12 R4**: that is the robustness result, the 15-hour run has little left to measure, and the finding should say so rather than being padded out.

- [ ] **Step 6: The real run**

```bash
nvidia-smi
./scripts/run.sh supervise --deadline 86400 -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism inpaint \
    --tag doc5
```

Expected: ~15 h, resuming across evictions. Monitor with `tail -f outputs/doc5_<TS>/run.log` — the heartbeat prints one line per completed case, so silence with a live supervisor means waiting for the card, not hung.

- [ ] **Step 7: Report**

```bash
./scripts/run.sh report --run outputs/doc5_<TS> \
    --dataset configs/dataset_common.yaml
```

- [ ] **Step 8: Write the findings**

Create `docs/findings/2026-08-05-common-concept-result.md` following the shape of `docs/findings/2026-07-29-orchestration-result.md`: what was run, the table, what broke, and what the number does and does not license. `outputs/` is gitignored, so this file is the durable record.

State plainly which of doc 5 §2's three readings the CI supports. **Do not re-describe a `harm` or `inconclusive` verdict as a success**, and do not adjust the margin — it was fixed at −0.02 before any data existed, and moving it now is the one change that would invalidate the whole benchmark.

- [ ] **Step 9: Commit**

```bash
git add docs/findings/2026-08-05-common-concept-result.md
git commit -m "docs: the common-concept result, and what it does not license"
```

---

## Notes for the executor

- **ImageNet is never in the corpus.** If `validate` raises `gt_ref_in_corpus`, the fix is to move files, never to weaken the check. The whole benchmark depends on the pipeline being unable to retrieve the image it is about to be graded against.
- **The margin is −0.02 and was fixed before any data existed.** If the result is `inconclusive`, report `inconclusive`. Tuning the margin after seeing deltas is the single change that would make the entire exercise worthless.
- **The paired delta must use the same mask and the same held-out refs on both sides.** If a test appears to want the draft scored against the edit pool, or cropped to a different box, the test is wrong.
- **Healthy cases are excluded from the delta mean, not counted as zeros.** Their delta is genuinely zero by construction, and averaging it in dilutes a real effect toward zero — which flatters the method.
- **`unchanged` is `best == "draft"`,** never a threshold on the delta. Keeping it independent of `MARGIN` is what lets the counts be read when the CI is ambiguous.
- **Exit 2 from the pipeline is not a failure.** It means the card was taken and `queue.json` is checkpointed. `supervise.sh` retries it; any other non-zero code is a real bug and must stop the loop.
- **Preservation must be exactly 1.0 for `stitch` composites.** Not 0.999. If it is not, the compositor changed and the metric is telling you so.
- **Do not lower `eval.heldout_refs` to 0** to make a thin case fit. `make_cases.py` guarantees 3 refs; a case with fewer is dropped at generation time on purpose.
- **A `corpus_thin` warning is a result, not a blocker.** The report carries the per-concept count so a bad case can be checked against whether the corpus had anything to offer.
- **`img2dataset` attrition of 30–50% is normal.** `select` over-requests for exactly this reason. A thin corpus is visible in the manifest rather than silent.
- **If the verifier passes nearly every common concept**, that is doc 5 §12 R4 and it is the answer, not a setback. Write it up as such.
