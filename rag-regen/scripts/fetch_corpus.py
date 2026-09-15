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

import yaml  # noqa: E402

from ragregen import config, corpus, env  # noqa: E402

#: One 64.7 MB shard holds far more rows than we need, and `--shards`
#: defaults to 1, so `select` reads exactly one. There is no escalation: if
#: the first shard cannot fill the request, re-run with `--shards 2` by hand.
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


def download_cmd(urls, image_dir, processes: int, threads: int) -> list[str]:
    """The img2dataset invocation, built so it can be asserted on.

    `-m img2dataset.main`, never `-m img2dataset`: the package ships no
    __main__.py, so the bare package name is not an executable module target
    and python exits 1 with "cannot be directly executed" before a single
    image is fetched. `main.py` carries the __main__ guard, and
    `fire.Fire(download)` behind it accepts exactly these flag names.
    """
    return [
        sys.executable, "-m", "img2dataset.main",
        "--url_list", str(urls),
        "--input_format", "parquet",
        "--url_col", "URL",
        "--caption_col", "TEXT",
        "--output_folder", str(image_dir),
        "--output_format", "files",
        "--image_size", "384",
        "--resize_mode", "keep_ratio",
        "--processes_count", str(processes),
        "--thread_count", str(threads),
        "--retries", "1",
        "--encode_quality", "90",
        "--encode_format", "jpg",
        "--disable_all_reencoding", "False",
    ]


def cmd_download(args) -> int:
    """Shell out to img2dataset. Dead links are the norm, not an error."""
    args.image_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = download_cmd(args.urls, args.image_dir, args.processes, args.threads)
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
    dataset_concepts = _concepts(args.dataset)
    coarse_concepts = list(yaml.safe_load(args.coarse.read_text()) or {})
    #: The pool make_cases.rank_classes will score is the whole candidate
    #: set -- the 22 bridge concepts plus the ~532 coarse-map keys -- not
    #: just the dataset's own concepts. Lowercased before the union: the
    #: dataset's concepts are mixed-case ("Boston bull") while every
    #: coarse-map key is already lowercase ("boston bull"), and a
    #: case-sensitive union would carry the same concept as two manifest
    #: keys. per_concept's keys are lowercase from here on -- validate.py's
    #: corpus-density warning (Task 7) reads this manifest and relies on
    #: that. Sorted so the manifest is byte-reproducible across runs.
    dataset_lower = {c.lower() for c in dataset_concepts}
    coarse_lower = {c.lower() for c in coarse_concepts}
    concepts = sorted(dataset_lower | coarse_lower)
    manifest = build_manifest(args.image_dir, concepts)
    args.out.write_text(json.dumps(manifest, indent=2))
    #: Scoped to the dataset's own concepts on purpose: those are the bridge
    #: cases that must work. Looked up and printed lowercased, matching
    #: per_concept's own key convention -- a case-sensitive lookup here
    #: would miss "Boston bull" against the lowercase key and misreport it
    #: as thin no matter its actual count.
    thin = [c for c in sorted(dataset_lower) if manifest["per_concept"][c] < args.k]
    print(f"[manifest] {manifest['n_images']} images -> {args.out}")
    if thin:
        #: Printed, never raised. "The corpus had nothing here" is a result
        #: the report must carry, not a reason to abort (doc 5 §10).
        print(f"[manifest] {len(thin)} concepts below k={args.k}: "
              f"{', '.join(thin[:10])}")
    #: How many of the coarse-map candidates cleared k -- this is what
    #: determines whether make-cases can actually fill its 70 common slots.
    cleared = sum(1 for c in coarse_lower if manifest["per_concept"][c] >= args.k)
    print(f"[manifest] {cleared}/{len(coarse_lower)} coarse-map "
          f"candidates cleared k={args.k}")
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
    m.add_argument("--coarse", type=Path,
                   default=env.PROJECT_ROOT / "configs/imagenet_coarse.yaml")
    m.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
