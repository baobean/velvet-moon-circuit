"""PlantCLEF2024 build adapter -- CSV `organ` column -> mmkg_store record.

Unlike CUB (point annotations only, crops must be *derived*), PlantCLEF's CSV
already labels every image by part type directly (`organ` in
{leaf, flower, fruit, bark, habit, branch, scan}), so a part crop here is just
"one whole image whose `organ` equals that part type" -- no
cropping/resizing at build time (``select_part_images``/``compose_reference``
downstream already resize to fit a tile). Candidates/medoid come from the
species' whole (uncropped) images, exactly like ``build_treevill.py``.

Species scope is FROZEN to the 13-species wk2 pilot list decided 2026-09-17
(see "Frozen wk2 pilot species list" in
``docs/superpowers/2026-09-17-cvpr-partgraph-handoff.md``) -- this module
never re-derives or expands that list. Matching is by exact string equality
on the CSV's `species` column (never fuzzy).

Image fetch: most images are not on local disk. Five of each frozen
species' images are already staged under ``wk1_screen_images/<species_id>/``
from the wk1 fidelity check (filenames there are the CSV's own `image_name`
value, NOT a name derived from the `url` column -- confirmed empirically
2026-09-17: iNaturalist-hosted rows all share the URL basename
`original.jpg`/`original.jpeg`, which cannot be the on-disk filename since it
would collide across images, while every staged file matches its row's
`image_name` exactly). Everything else is fetched via a generic
``requests.get(url, timeout=30)`` on the `url` column verbatim -- the CSV's
`url` column resolves to at least three different hosts
(``bs.plantnet.org``, ``inaturalist-open-data.s3.amazonaws.com``,
``observation.org``), so the fetch step never reconstructs or assumes a
single host. A row whose fetch fails (timeout, 404, ...) is skipped and
logged, never aborts the whole build; a species that ends up with zero
fetched images is skipped entirely (never fabricated).

Part crops reuse the already-computed candidate embedding for their chosen
image (same file, same deterministic encoder -> same vector) rather than
re-calling ``embed_fn`` a second time on an identical path -- a deliberate
efficiency deviation from ``build_cub.py``/``build_treevill.py``, both of
which crop to a *new* derived image file and so must re-embed. Every indexed
vector (candidate, medoid, part_crop) still gets its own
``index_add(vec, meta)`` call/ref, exactly like the other adapters.

PlantCLEF DOES carry taxonomy (`genus`/`family` CSV columns, unlike CUB), so
records get a real ``taxonomy={"genus": ..., "family": ...}``.
``species_key`` is the CSV's numeric ``species_id`` (stable, dedup/path-safe)
rather than a slugified scientific name.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

from ragregen.mmkg_store import schema, store
from ragregen.mmkg_store.build_treevill import _centroid_nearest_index
from ragregen.mmkg_store.embed_index import IndexBuilder

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    "/mnt/mmlab2024nas/ldtuan/data/partgraph/plantclef2024/"
    "PlantCLEF2024singleplanttrainingdata.csv"
)
DEFAULT_WK1_DIR = "/mnt/mmlab2024nas/ldtuan/data/partgraph/plantclef2024/wk1_screen_images"
DEFAULT_STORE_IMAGES_DIR = "/mnt/mmlab2024nas/ldtuan/data/partgraph/plantclef2024/store_images"
DEFAULT_OUT_DIR = "/mnt/mmlab2024nas/ldtuan/data/partgraph/partgraph_store/plantclef"

# --- frozen wk2 pilot species list (13, decided 2026-09-17, after wk1 screen) ---
# (species, species_id) -- exact `species` column strings, re-verified 2026-09-17
# directly against the CSV (the handoff doc's markdown table truncated one name
# to "Thapsia scabra (...)"; the full string below is the CSV ground truth).
# Do not re-derive this list -- it is frozen and pre-registered.
FROZEN_SPECIES: list[tuple[str, str]] = [
    ("Sisymbrium polyceratium L.", "1358432"),
    ("Narcissus viridiflorus Schousb.", "1360973"),
    ("Campanula petraea L.", "1398374"),
    ("Rostraria litorea (All.) Holub", "1361398"),
    ("Euphorbia akenocarpa Guss.", "1358523"),
    ("Astragalus turolensis Pau", "1359157"),
    ("Taraxacum oblongatum Dahlst.", "1390426"),
    ("Euphorbia graminifolia Vill.", "1392165"),
    ("Agrostis × murbeckii Fouill.", "1647575"),
    ("Desmazeria sicula (Jacq.) Dumort.", "1361240"),
    ("Teucrium turredanum Losa & Rivas Goday", "1564438"),
    ("Thapsia scabra (Cav.) Simonsen, Rønsted, Weitzel & Spalik", "1744638"),
    ("Hemionitis guanchica (Bolle) Christenh.", "1722699"),
]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _ensure_csv_field_size_limit() -> None:
    """The `field larger than field limit` error is Python's default 131072-byte
    cap, not row corruption (re-verified 2026-09-17 against the full 786 MB /
    1,408,033-row CSV -- no malformed rows found at this limit). Still called
    unconditionally as a guard before opening.
    """
    csv.field_size_limit(sys.maxsize)


def parse_plantclef_rows(csv_path, species_names) -> dict[str, list[dict]]:
    """Parse ``csv_path`` (`;`-delimited), grouping rows by exact `species` match.

    ``species_names`` is any iterable of exact CSV `species` strings to keep
    (never fuzzy-matched). Returns ``{species: [row_dict, ...]}`` for species
    with at least one matching row; a species with zero rows in the CSV is
    simply absent from the result (caller's responsibility to notice).

    A malformed CSV row (rare -- none observed in the full file as of
    2026-09-17, kept as a guard not the primary concern) is skipped and
    logged rather than aborting the whole parse.
    """
    _ensure_csv_field_size_limit()
    wanted = set(species_names)
    out: dict[str, list[dict]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                logger.warning("skipping malformed PlantCLEF CSV row: %s", exc)
                continue
            sp = row.get("species")
            if sp in wanted:
                out.setdefault(sp, []).append(row)

    return out


# ---------------------------------------------------------------------------
# Image staging (local reuse + generic-host fetch)
# ---------------------------------------------------------------------------

def _default_fetch(url: str, dest_path, *, max_attempts: int = 3, backoff_seconds: float = 1.0) -> None:
    """Generic GET on ``url`` verbatim -- never reconstructs or assumes a host
    (the CSV's `url` column spans `bs.plantnet.org`,
    `inaturalist-open-data.s3.amazonaws.com`, `observation.org`, ...).

    Retries up to ``max_attempts`` total on connection/timeout errors and 5xx
    responses, with exponential backoff (``backoff_seconds``, then ``2x``,
    ``4x``, ...) between attempts -- a single transient blip (timeout, brief
    5xx) should not permanently skip an image. A 4xx response is NOT retried
    (a 404 will not fix itself) and raises immediately. Raises on final
    failure (transient error persisting past ``max_attempts``, or any 4xx);
    the caller (``_stage_image``) catches and logs, preserving the existing
    skip-and-continue contract.
    """
    import time

    import requests

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                raise  # client error -- retrying will not help
            if attempt == max_attempts:
                raise
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == max_attempts:
                raise
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
            continue

        dest_path.write_bytes(resp.content)
        return


def _stage_image(row: dict, species_id: str, wk1_dir, store_dir, fetch_fn) -> str | None:
    """Resolve one CSV row to a local image path, reusing wk1-staged /
    previously-fetched files before ever calling ``fetch_fn``.

    Local filename is the row's own `image_name` (NOT derived from `url` --
    see module docstring: the `url` basename collides across iNaturalist-
    hosted rows, `image_name` does not). Returns ``None`` (never raises) on
    fetch failure, so one bad row never aborts the species' build.
    """
    filename = row["image_name"]

    wk1_path = Path(wk1_dir) / species_id / filename
    if wk1_path.exists():
        return str(wk1_path)

    store_path = Path(store_dir) / species_id / filename
    if store_path.exists():  # already fetched by a prior (possibly killed) run
        return str(store_path)

    try:
        fetch_fn(row["url"], store_path)
    except Exception as exc:  # noqa: BLE001 -- any fetch failure is skip-and-log
        logger.warning(
            "skipping PlantCLEF image %s (species_id=%s): fetch failed: %s",
            filename, species_id, exc,
        )
        return None

    if not store_path.exists():
        return None
    return str(store_path)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def plantclef_record(
    species_name: str,
    species_id: str,
    rows: list[dict],
    embed_fn,
    index_add,
    *,
    wk1_dir,
    store_dir,
    fetch_fn=_default_fetch,
    provenance_extra: dict | None = None,
) -> dict | None:
    """Assemble one PlantCLEF species record, or ``None`` if every row's
    image failed to stage (skip the species entirely -- never fabricated).

    Every successfully staged image is re-embedded once (batch) via
    ``embed_fn`` to pick the centroid-nearest medoid and to index every
    candidate. For each distinct `organ` present among the staged images,
    the first staged image of that organ (by ascending `image_name`) becomes
    that part_type's crop, reusing its already-computed embedding (see
    module docstring) rather than a second ``embed_fn`` call on the same
    file.
    """
    gid = schema.global_id("plantclef", species_id)

    rows_sorted = sorted(rows, key=lambda r: r["image_name"])

    fetched_rows: list[dict] = []
    fetched_paths: list[str] = []
    skipped: list[dict] = []
    for row in rows_sorted:
        path = _stage_image(row, species_id, wk1_dir, store_dir, fetch_fn)
        if path is None:
            skipped.append({"image_name": row["image_name"], "organ": row["organ"], "url": row["url"]})
            continue
        fetched_rows.append(row)
        fetched_paths.append(path)

    if not fetched_paths:
        logger.warning(
            "skipping PlantCLEF species %s (species_id=%s): 0/%d rows fetched",
            species_name, species_id, len(rows),
        )
        return None

    embeddings = np.asarray(embed_fn(fetched_paths))
    medoid_idx = _centroid_nearest_index(embeddings)
    medoid_path = fetched_paths[medoid_idx]

    candidates = []
    medoid_ref = None
    for i, (path, vec) in enumerate(zip(fetched_paths, embeddings)):
        ref = index_add(vec, {"global_id": gid, "kind": "candidate", "path": path})
        candidates.append({"image_path": path, "embedding_ref": ref})
        if i == medoid_idx:
            medoid_ref = index_add(
                embeddings[medoid_idx],
                {"global_id": gid, "kind": "medoid", "path": medoid_path},
            )

    medoid = {
        "image_path": medoid_path,
        "embedding_ref": medoid_ref,
        "k_images": len(fetched_paths),
        "selection": "siglip-centroid-nearest",
    }

    part_crops = []
    seen_organs: set[str] = set()
    for i, row in enumerate(fetched_rows):
        organ = row["organ"]
        if organ in seen_organs:
            continue
        seen_organs.add(organ)
        path = fetched_paths[i]
        # No cropping/resizing at build time -- the organ column already
        # labels the whole image, so reuse its already-computed candidate
        # embedding instead of re-calling embed_fn on the same file.
        ref = index_add(embeddings[i], {"global_id": gid, "kind": "part_crop", "path": path})
        part_crops.append({"part_type": organ, "image_path": path, "embedding_ref": ref})

    taxonomy = {"genus": rows[0]["genus"], "family": rows[0]["family"]}

    provenance = {
        "species_id": species_id,
        "n_rows_total": len(rows),
        "eligible_images": fetched_paths,
        "skipped_rows": skipped,
    }
    if provenance_extra:
        provenance.update(provenance_extra)

    return schema.build_record(
        dataset="plantclef",
        species_key=species_id,
        scientific_name=species_name,
        common_name=None,  # PlantCLEF CSV has no common-name column
        taxonomy=taxonomy,
        medoid=medoid,
        candidates=candidates,
        part_crops=part_crops,
        attributes={},
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_plantclef_records(
    csv_path,
    embed_fn,
    index_add,
    *,
    frozen_species: list[tuple[str, str]] | None = None,
    wk1_dir=DEFAULT_WK1_DIR,
    store_dir=DEFAULT_STORE_IMAGES_DIR,
    fetch_fn=_default_fetch,
) -> list[dict]:
    """Parse ``csv_path`` and build one record per frozen species present in it.

    ``frozen_species`` defaults to the module-level ``FROZEN_SPECIES`` (13,
    frozen 2026-09-17); a test overrides it with a tiny fixture list. A
    frozen species absent from the CSV, or whose every row failed to fetch,
    is skipped and logged -- never crashes the build.
    """
    frozen_species = frozen_species if frozen_species is not None else FROZEN_SPECIES
    species_names = [name for name, _sid in frozen_species]
    by_species = parse_plantclef_rows(csv_path, species_names)

    records = []
    for species_name, species_id in frozen_species:
        rows = by_species.get(species_name)
        if not rows:
            logger.warning(
                "skipping frozen PlantCLEF species %s (species_id=%s): 0 CSV rows",
                species_name, species_id,
            )
            continue
        rec = plantclef_record(
            species_name, species_id, rows, embed_fn, index_add,
            wk1_dir=wk1_dir, store_dir=store_dir, fetch_fn=fetch_fn,
        )
        if rec is not None:
            records.append(rec)
    return records


def build_plantclef_store(
    csv_path,
    out_dir,
    embed_fn,
    *,
    frozen_species: list[tuple[str, str]] | None = None,
    wk1_dir=DEFAULT_WK1_DIR,
    store_dir=DEFAULT_STORE_IMAGES_DIR,
    fetch_fn=_default_fetch,
    index_builder=None,
) -> list[dict]:
    """Build every frozen PlantCLEF species record and persist the store to ``out_dir``.

    ``index_builder`` defaults to a fresh ``embed_index.IndexBuilder`` (a real
    FAISS index); ``embed_fn``/``fetch_fn`` are injectable seams so tests
    never load a real model or touch the network.
    """
    if index_builder is None:
        index_builder = IndexBuilder()
    records = build_plantclef_records(
        csv_path, embed_fn, index_builder.add,
        frozen_species=frozen_species, wk1_dir=wk1_dir, store_dir=store_dir,
        fetch_fn=fetch_fn,
    )
    store.write_store(records, out_dir, index_builder=index_builder)
    return records
