"""CUB-200-2011 build adapter -- point annotations -> mmkg_store record.

CUB has no per-part bounding boxes (recon frozen 2026-09-17, see
``docs/superpowers/2026-09-17-cvpr-partgraph-handoff.md``), only 15 named
point annotations (``parts/part_locs.txt``) plus one whole-bird box per image
(``bounding_boxes.txt``). Part crops are therefore *derived*: the 15 points
are grouped into 6 coarser part-types (``head/back/breast/wing/leg/tail``,
frozen below), one "source photo" per species is picked as the image with
the most visible part-groups (ties broken by lowest ``image_id``, so every
part crop for a species comes from one consistent specimen/pose), and each
group's crop box is a square centered on the mean ``(x, y)`` of its visible
sub-parts in that photo, sized off the photo's whole-bird box. The medoid is
chosen independently (over every image of the species, not just the source
photo) via the same ``siglip-centroid-nearest`` rule as ``build_treevill.py``
-- medoid and part-crop source photo may therefore differ.

CUB carries no genus/family metadata, so records get ``taxonomy={}`` (no
``instance_of`` relations) rather than a fabricated genus/family, and
``scientific_name=None`` (CUB gives only an English common name).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ragregen.mmkg_store import schema, store
from ragregen.mmkg_store.build_treevill import _centroid_nearest_index
from ragregen.mmkg_store.embed_index import IndexBuilder

# --- frozen part-type grouping (15 CUB points -> 6 crop types) -------------
# See "Frozen decisions for build_cub.py" in the 2026-09-17 handoff doc.
PART_GROUPS: dict[str, set[str]] = {
    "head": {"beak", "crown", "forehead", "left eye", "right eye", "throat", "nape"},
    "back": {"back"},
    "breast": {"breast", "belly"},
    "wing": {"left wing", "right wing"},
    "leg": {"left leg", "right leg"},
    "tail": {"tail"},
}

PART_NAME_TO_GROUP: dict[str, str] = {
    name: group for group, names in PART_GROUPS.items() for name in names
}


# ---------------------------------------------------------------------------
# Raw CUB file parsing
# ---------------------------------------------------------------------------

def _read_lines(path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def parse_images(cub_root) -> dict[int, str]:
    """``images.txt``: ``<image_id> <species_dir>/<filename>`` -> id -> relpath."""
    out: dict[int, str] = {}
    for line in _read_lines(Path(cub_root) / "images.txt"):
        iid_str, rel = line.split(" ", 1)
        out[int(iid_str)] = rel
    return out


def parse_image_class_labels(cub_root) -> dict[int, int]:
    """``image_class_labels.txt``: ``<image_id> <class_id>`` -> id -> class_id."""
    out: dict[int, int] = {}
    for line in _read_lines(Path(cub_root) / "image_class_labels.txt"):
        iid_str, cid_str = line.split()
        out[int(iid_str)] = int(cid_str)
    return out


def parse_classes(cub_root) -> dict[int, str]:
    """``classes.txt``: ``<class_id> <NNN.Species_Name>`` -> class_id -> name."""
    out: dict[int, str] = {}
    for line in _read_lines(Path(cub_root) / "classes.txt"):
        cid_str, name = line.split(" ", 1)
        out[int(cid_str)] = name
    return out


def parse_bounding_boxes(cub_root) -> dict[int, tuple[float, float, float, float]]:
    """``bounding_boxes.txt``: one whole-bird box per image, ``<id> <x> <y> <w> <h>``."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for line in _read_lines(Path(cub_root) / "bounding_boxes.txt"):
        toks = line.split()
        iid = int(toks[0])
        x, y, w, h = (float(v) for v in toks[1:5])
        out[iid] = (x, y, w, h)
    return out


def parse_parts(cub_root) -> dict[int, str]:
    """``parts/parts.txt``: ``<part_id> <part_name>`` (name may contain spaces)."""
    out: dict[int, str] = {}
    for line in _read_lines(Path(cub_root) / "parts" / "parts.txt"):
        pid_str, name = line.split(" ", 1)
        out[int(pid_str)] = name.strip()
    return out


def parse_part_locs(cub_root, part_names: dict[int, str]) -> dict[int, dict[str, tuple[float, float, bool]]]:
    """``parts/part_locs.txt``: ``<image_id> <part_id> <x> <y> <visible>``.

    Returns ``{image_id: {part_name: (x, y, visible)}}``. An image absent
    here (or a part absent for a given image) simply has no visible points
    -- never fabricated as ``(0, 0, False)``.
    """
    out: dict[int, dict[str, tuple[float, float, bool]]] = {}
    for line in _read_lines(Path(cub_root) / "parts" / "part_locs.txt"):
        toks = line.split()
        iid, pid = int(toks[0]), int(toks[1])
        x, y = float(toks[2]), float(toks[3])
        vis = bool(int(float(toks[4])))
        name = part_names.get(pid)
        if name is None:
            continue
        out.setdefault(iid, {})[name] = (x, y, vis)
    return out


def _common_name(class_name: str) -> str:
    """``"001.Black_footed_Albatross"`` -> ``"Black footed Albatross"``."""
    name = class_name.split(".", 1)[1] if "." in class_name else class_name
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Part-group / crop-box derivation
# ---------------------------------------------------------------------------

def _visible_groups(part_locs_for_image: dict[str, tuple[float, float, bool]]) -> set[str]:
    groups: set[str] = set()
    for name, (_x, _y, vis) in part_locs_for_image.items():
        if not vis:
            continue
        group = PART_NAME_TO_GROUP.get(name)
        if group:
            groups.add(group)
    return groups


def _pick_source_image(image_ids, part_locs_by_image: dict[int, dict]) -> int:
    """The image_id with the most visible part-groups; ties -> lowest image_id.

    Iterating ascending and requiring strict improvement (``>``, not ``>=``)
    makes the first (lowest-id) image to reach the max count the winner.
    """
    best_id = None
    best_count = -1
    for iid in sorted(image_ids):
        count = len(_visible_groups(part_locs_by_image.get(iid, {})))
        if count > best_count:
            best_count = count
            best_id = iid
    return best_id


def _clamp_axis(center: float, side: float, size: float) -> tuple[float, float]:
    """Clamp a ``side``-length interval centered on ``center`` into ``[0, size]``.

    Clamps by *shifting* (never shrinking) the interval; if ``side >= size``,
    the full ``[0, size]`` extent is used on that axis.
    """
    if side >= size:
        return 0.0, float(size)
    half = side / 2.0
    lo = center - half
    hi = center + half
    if lo < 0:
        hi -= lo
        lo = 0.0
    if hi > size:
        lo -= (hi - size)
        hi = float(size)
    lo = max(lo, 0.0)
    hi = min(hi, float(size))
    return lo, hi


def _crop_box_for_group(
    part_locs_for_image: dict[str, tuple[float, float, bool]],
    group_parts: set[str],
    bbox: tuple[float, float, float, float],
    img_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Square crop box for one part-group in the source photo, or ``None``.

    ``None`` when the group has zero visible sub-parts in this photo (per
    the frozen rule: skipped for that species, never fabricated).
    """
    xs: list[float] = []
    ys: list[float] = []
    for name in group_parts:
        loc = part_locs_for_image.get(name)
        if loc is None:
            continue
        x, y, vis = loc
        if vis:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    _bx, _by, bbox_w, bbox_h = bbox
    side = max(32, round(0.45 * max(bbox_w, bbox_h)))
    img_w, img_h = img_size

    left, right = _clamp_axis(cx, side, img_w)
    top, bottom = _clamp_axis(cy, side, img_h)
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))


def _make_part_crops(species_key, source_path, source_part_locs, bbox, crops_dir, embed_fn, index_add, gid) -> list[dict]:
    part_crops: list[dict] = []
    with Image.open(source_path) as im:
        im = im.convert("RGB")
        img_size = im.size
        for group_name, group_parts in PART_GROUPS.items():
            box = _crop_box_for_group(source_part_locs, group_parts, bbox, img_size)
            if box is None:
                continue
            crop_img = im.crop(box)
            crop_path = Path(crops_dir) / species_key / f"{group_name}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_img.save(crop_path)

            crop_vec = np.asarray(embed_fn([str(crop_path)]))[0]
            ref = index_add(crop_vec, {"global_id": gid, "kind": "part_crop", "path": str(crop_path)})
            part_crops.append({
                "part_type": group_name,
                "image_path": str(crop_path),
                "embedding_ref": ref,
            })
    return part_crops


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def cub_record(
    species_key: str,
    common_name: str,
    images: dict[int, str],
    part_locs: dict[int, dict[str, tuple[float, float, bool]]],
    bboxes: dict[int, tuple[float, float, float, float]],
    embed_fn,
    index_add,
    crops_dir,
    *,
    provenance_extra: dict | None = None,
) -> dict:
    """Assemble one CUB species record.

    ``images``/``part_locs``/``bboxes`` are keyed by ``image_id``; every
    image in ``images`` is re-embedded (one batch, via ``embed_fn``) to pick
    the centroid-nearest medoid and to index every candidate, exactly like
    ``build_treevill.treevill_record``. Part crops are derived separately
    from the species' single "source photo" (see module docstring) and
    embedded/indexed one at a time.
    """
    gid = schema.global_id("cub", species_key)

    sorted_ids = sorted(images)
    eligible_paths = [images[iid] for iid in sorted_ids]

    embeddings = np.asarray(embed_fn(eligible_paths))
    medoid_idx = _centroid_nearest_index(embeddings)
    medoid_path = eligible_paths[medoid_idx]

    candidates = []
    medoid_ref = None
    for i, (path, vec) in enumerate(zip(eligible_paths, embeddings)):
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
        "k_images": len(eligible_paths),
        "selection": "siglip-centroid-nearest",
    }

    source_id = _pick_source_image(sorted_ids, part_locs)
    source_path = images[source_id]
    source_bbox = bboxes[source_id]
    source_part_locs = part_locs.get(source_id, {})

    part_crops = _make_part_crops(
        species_key, source_path, source_part_locs, source_bbox, crops_dir, embed_fn, index_add, gid,
    )

    provenance = {
        "eligible_images": eligible_paths,
        "source_photo": source_path,
        "source_image_id": source_id,
    }
    if provenance_extra:
        provenance.update(provenance_extra)

    # CUB has no genus/family metadata (recon frozen 2026-09-17) -- taxonomy={}
    # rather than a fabricated genus/family, and no scientific name.
    return schema.build_record(
        dataset="cub",
        species_key=species_key,
        scientific_name=None,
        common_name=common_name,
        taxonomy={},
        medoid=medoid,
        candidates=candidates,
        part_crops=part_crops,
        attributes={},
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_cub_records(cub_root, embed_fn, index_add, crops_dir) -> list[dict]:
    """Parse a CUB_200_2011-layout ``cub_root`` and build one record per species."""
    cub_root = Path(cub_root)

    images_rel = parse_images(cub_root)
    labels = parse_image_class_labels(cub_root)
    classes = parse_classes(cub_root)
    bboxes = parse_bounding_boxes(cub_root)
    part_names = parse_parts(cub_root)
    part_locs = parse_part_locs(cub_root, part_names)

    images_dir = cub_root / "images"
    resolved_images = {iid: str(images_dir / rel) for iid, rel in images_rel.items()}

    by_class: dict[int, list[int]] = {}
    for iid, cid in labels.items():
        by_class.setdefault(cid, []).append(iid)

    records = []
    for cid in sorted(by_class):
        class_name = classes[cid]
        image_ids = by_class[cid]
        species_images = {iid: resolved_images[iid] for iid in image_ids}
        species_bboxes = {iid: bboxes[iid] for iid in image_ids}
        species_part_locs = {iid: part_locs.get(iid, {}) for iid in image_ids}

        rec = cub_record(
            class_name, _common_name(class_name), species_images, species_part_locs,
            species_bboxes, embed_fn, index_add, crops_dir,
        )
        records.append(rec)
    return records


def build_cub_store(cub_root, out_dir, embed_fn, *, index_builder=None) -> list[dict]:
    """Build every CUB species record and persist the store to ``out_dir``.

    ``index_builder`` defaults to a fresh ``embed_index.IndexBuilder`` (a
    real FAISS index, not a fake) -- Phase-0 CUB is small enough that no
    separate encoder-loading orchestration (cf. ``build.py``) is needed yet;
    ``embed_fn`` is still an injectable seam so tests never load a real model.
    """
    if index_builder is None:
        index_builder = IndexBuilder()
    crops_dir = Path(out_dir) / "crops"
    records = build_cub_records(cub_root, embed_fn, index_builder.add, crops_dir)
    store.write_store(records, out_dir, index_builder=index_builder)
    return records
