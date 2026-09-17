import csv
import functools
import hashlib
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from ragregen.mmkg_store import build_plantclef as bp
from ragregen.mmkg_store.store import Store

CSV_COLUMNS = [
    "image_name", "organ", "species_id", "obs_id", "license", "partner", "author",
    "altitude", "latitude", "longitude", "gbif_species_id", "species", "genus",
    "family", "dataset", "publisher", "references", "url", "learn_tag", "image_backup_url",
]


def _row(image_name, organ, species_id, species, genus, family, url):
    return {
        "image_name": image_name, "organ": organ, "species_id": species_id,
        "obs_id": "1", "license": "cc-by", "partner": "p", "author": "a",
        "altitude": "", "latitude": "", "longitude": "", "gbif_species_id": "42",
        "species": species, "genus": genus, "family": family, "dataset": "plant",
        "publisher": "pub", "references": "", "url": url, "learn_tag": "train",
        "image_backup_url": "",
    }


# Two "real" frozen species + one always-fails-fetch species (species C) to
# exercise skip-and-log without aborting the whole build.
SPECIES_A = ("Testia fakea L.", "900001")
SPECIES_B = ("Testia bogusii L.", "900002")
SPECIES_C = ("Testia nihila L.", "900003")  # every row fails to fetch -> skipped

ROWS = [
    # species A: leaf (wk1-staged), leaf (fetched), flower (fetched)
    _row("a_img1.jpg", "leaf", "900001", SPECIES_A[0], "Testia", "Fakeaceae",
         "https://bs.plantnet.org/image/o/aaa1.jpg"),
    _row("a_img2.jpg", "leaf", "900001", SPECIES_A[0], "Testia", "Fakeaceae",
         "https://inaturalist-open-data.s3.amazonaws.com/photos/1/original.jpg"),
    _row("a_img3.jpg", "flower", "900001", SPECIES_A[0], "Testia", "Fakeaceae",
         "https://inaturalist-open-data.s3.amazonaws.com/photos/2/original.jpeg"),
    # species B: bark (fetched), habit (fetched), habit (fetched, 2nd -- not picked as crop)
    _row("b_img1.jpg", "bark", "900002", SPECIES_B[0], "Testia", "Fakeaceae",
         "https://observation.org/photos/10.jpg"),
    _row("b_img2.jpg", "habit", "900002", SPECIES_B[0], "Testia", "Fakeaceae",
         "https://observation.org/photos/11.jpg"),
    _row("b_img3.jpg", "habit", "900002", SPECIES_B[0], "Testia", "Fakeaceae",
         "https://observation.org/photos/12.jpg"),
    # species C: single row, fetch always raises -> species has 0 fetched images
    _row("c_img1.jpg", "leaf", "900003", SPECIES_C[0], "Testia", "Fakeaceae",
         "https://bs.plantnet.org/image/o/ccc1.jpg"),
    # a non-frozen species present in the CSV -- must never appear in output
    # (exact-match only, no fuzzy matching)
    _row("x_img1.jpg", "leaf", "999999", "Testia fakea", "Testia", "Fakeaceae",
         "https://bs.plantnet.org/image/o/xxx1.jpg"),
]

FROZEN = [SPECIES_A, SPECIES_B, SPECIES_C]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_fake_image(path: Path, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _fake_embed_fn(paths):
    """Deterministic, content-independent embedding: md5(path) -> 2D vector."""
    vecs = []
    for p in paths:
        h = hashlib.md5(str(p).encode()).hexdigest()
        v1 = int(h[:8], 16) % 1000 / 1000.0
        v2 = int(h[8:16], 16) % 1000 / 1000.0
        vecs.append([v1, v2])
    return np.array(vecs, dtype="float32")


class _FakeFetcher:
    """Records calls; raises for any url containing "ccc" (simulates a dead link)."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, url: str, dest_path) -> None:
        self.calls.append(url)
        if "ccc" in url:
            raise TimeoutError("simulated fetch failure")
        _write_fake_image(Path(dest_path))


def _build_fixture(tmp_path: Path):
    csv_path = tmp_path / "plantclef.csv"
    _write_csv(csv_path, ROWS)

    wk1_dir = tmp_path / "wk1_screen_images"
    store_dir = tmp_path / "store_images"

    # species A's first image ("a_img1.jpg") is pre-staged under wk1 -- must
    # be reused, never fetched.
    _write_fake_image(wk1_dir / "900001" / "a_img1.jpg")

    return csv_path, wk1_dir, store_dir


# --- unit tests: CSV parsing -------------------------------------------------

def test_parse_plantclef_rows_exact_match_only(tmp_path):
    csv_path, _wk1, _store = _build_fixture(tmp_path)
    by_species = bp.parse_plantclef_rows(csv_path, [SPECIES_A[0], SPECIES_B[0]])

    assert set(by_species) == {SPECIES_A[0], SPECIES_B[0]}
    assert len(by_species[SPECIES_A[0]]) == 3
    assert len(by_species[SPECIES_B[0]]) == 3
    # "Testia fakea" (no trailing " L.") must NOT fuzzy-match "Testia fakea L."
    assert "Testia fakea" not in by_species


def test_parse_plantclef_rows_missing_species_absent_from_result(tmp_path):
    csv_path, _wk1, _store = _build_fixture(tmp_path)
    by_species = bp.parse_plantclef_rows(csv_path, ["Nonexistent species X."])
    assert by_species == {}


# --- hermetic fixture: tiny PlantCLEF-layout CSV + real (fake-embedded) build ----

def test_build_plantclef_store_hermetic_fixture(tmp_path):
    csv_path, wk1_dir, store_dir = _build_fixture(tmp_path)
    out_dir = tmp_path / "store_out"
    fetcher = _FakeFetcher()

    records = bp.build_plantclef_store(
        csv_path, out_dir, _fake_embed_fn,
        frozen_species=FROZEN, wk1_dir=wk1_dir, store_dir=store_dir, fetch_fn=fetcher,
    )

    # species C (all fetches fail) is skipped entirely -- only A and B build.
    assert len(records) == 2

    loaded = Store.load(out_dir)
    all_ids = loaded.query()
    assert set(all_ids) == {"plantclef:900001", "plantclef:900002"}

    a = loaded.get("plantclef:900001")
    b = loaded.get("plantclef:900002")

    for rec in (a, b):
        assert rec["dataset"] == "plantclef"
        assert rec["common_name"] is None
        assert rec["taxonomy"] == {"genus": "Testia", "family": "Fakeaceae"}
        # taxonomy hubs present -- both genus and family instance_of relations
        assert {"type": "instance_of", "target": "plantclef:genus:Testia"} in rec["relations"]
        assert {"type": "instance_of", "target": "plantclef:family:Fakeaceae"} in rec["relations"]
        assert rec["medoid"]["image_path"]
        assert rec["medoid"]["selection"] == "siglip-centroid-nearest"
        assert isinstance(rec["medoid"]["embedding_ref"], int)

    assert a["scientific_name"] == SPECIES_A[0]
    assert a["species_key"] == "900001"
    assert len(a["candidates"]) == 3
    assert a["medoid"]["k_images"] == 3

    assert b["scientific_name"] == SPECIES_B[0]
    assert len(b["candidates"]) == 3

    # part_crops: species A has leaf + flower; species B has bark + habit
    a_types = {c["part_type"] for c in a["part_crops"]}
    b_types = {c["part_type"] for c in b["part_crops"]}
    assert a_types == {"leaf", "flower"}
    assert b_types == {"bark", "habit"}

    # species A's leaf crop must be the FIRST fetched leaf image by ascending
    # image_name ("a_img1.jpg", the wk1-staged one), not "a_img2.jpg".
    leaf_crop = next(c for c in a["part_crops"] if c["part_type"] == "leaf")
    assert Path(leaf_crop["image_path"]).name == "a_img1.jpg"
    assert str(wk1_dir) in leaf_crop["image_path"]  # reused from wk1, not store_dir

    # every part crop / candidate / medoid path is a real on-disk image
    for rec in (a, b):
        for crop in rec["part_crops"]:
            crop_path = Path(crop["image_path"])
            assert crop_path.exists()
            with Image.open(crop_path) as im:
                assert im.size[0] > 0 and im.size[1] > 0
            assert isinstance(crop["embedding_ref"], int)

    # the wk1-staged image was never fetched over the network
    assert not any("aaa1.jpg" in url for url in fetcher.calls)
    # the always-failing species-C url WAS attempted (and failed)
    assert any("ccc1.jpg" in url for url in fetcher.calls)

    # non-frozen species never leaks into the store
    assert loaded.get("plantclef:999999") is None

    # FAISS index + sidecar persisted and loadable
    assert (out_dir / "index.faiss").exists()
    assert (out_dir / "sidecar.json").exists()
    # A: 3 candidates + 1 medoid + 2 part crops (leaf/flower, reusing candidate
    #    embeddings but each still gets its own index_add ref) = 6
    # B: 3 candidates + 1 medoid + 2 part crops (bark/habit) = 6
    assert len(loaded._sidecar["entries"]) == 12


def test_build_plantclef_store_species_with_zero_fetched_images_is_skipped(tmp_path):
    csv_path, wk1_dir, store_dir = _build_fixture(tmp_path)
    out_dir = tmp_path / "store_out"
    fetcher = _FakeFetcher()

    bp.build_plantclef_store(
        csv_path, out_dir, _fake_embed_fn,
        frozen_species=FROZEN, wk1_dir=wk1_dir, store_dir=store_dir, fetch_fn=fetcher,
    )

    loaded = Store.load(out_dir)
    assert loaded.get("plantclef:900003") is None


def test_build_plantclef_store_frozen_species_absent_from_csv_is_skipped(tmp_path):
    csv_path, wk1_dir, store_dir = _build_fixture(tmp_path)
    out_dir = tmp_path / "store_out"
    fetcher = _FakeFetcher()

    frozen_with_ghost = FROZEN + [("Species Not In CSV At All.", "900099")]
    records = bp.build_plantclef_store(
        csv_path, out_dir, _fake_embed_fn,
        frozen_species=frozen_with_ghost, wk1_dir=wk1_dir, store_dir=store_dir, fetch_fn=fetcher,
    )

    assert len(records) == 2  # ghost species contributes no record, no crash
    loaded = Store.load(out_dir)
    assert loaded.get("plantclef:900099") is None


def test_stage_image_reuses_wk1_before_calling_fetch_fn(tmp_path):
    wk1_dir = tmp_path / "wk1"
    store_dir = tmp_path / "store"
    _write_fake_image(wk1_dir / "900001" / "a_img1.jpg")

    row = ROWS[0]  # a_img1.jpg, species 900001
    fetcher = _FakeFetcher()
    path = bp._stage_image(row, "900001", wk1_dir, store_dir, fetcher)

    assert path == str(wk1_dir / "900001" / "a_img1.jpg")
    assert fetcher.calls == []


def test_stage_image_returns_none_and_does_not_raise_on_fetch_failure(tmp_path):
    wk1_dir = tmp_path / "wk1"
    store_dir = tmp_path / "store"
    row = ROWS[6]  # c_img1.jpg, url contains "ccc" -> always fails
    fetcher = _FakeFetcher()

    path = bp._stage_image(row, "900003", wk1_dir, store_dir, fetcher)

    assert path is None
    assert fetcher.calls == [row["url"]]


# --- unit tests: _default_fetch retry/backoff (Important fix, review round 1) ----

class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` -- just what ``_default_fetch``
    touches (``status_code``, ``content``, ``raise_for_status``)."""

    def __init__(self, status_code=200, content=b"fake-bytes"):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            exc = requests.exceptions.HTTPError(f"{self.status_code} error")
            exc.response = self
            raise exc


def test_default_fetch_retries_transient_failure_then_succeeds(tmp_path, monkeypatch):
    """A connection error on attempt 1 followed by success on attempt 2 must
    be recovered -- the retry loop, not a raised exception or a permanent
    skip."""
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("simulated transient failure")
        return _FakeResponse(status_code=200, content=b"ok-bytes")

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "out" / "img.jpg"

    bp._default_fetch("https://example.org/img.jpg", dest, backoff_seconds=0)

    assert calls["n"] == 2
    assert dest.read_bytes() == b"ok-bytes"


def test_default_fetch_persistent_transient_failure_raises_after_bounded_retries(
    tmp_path, monkeypatch
):
    """A failure that never clears must NOT retry forever -- it should raise
    after exactly ``max_attempts`` tries, preserving the skip-and-continue
    contract via ``_stage_image``'s catch-all (tested below)."""
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        raise requests.exceptions.Timeout("simulated persistent failure")

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "out" / "img.jpg"

    with pytest.raises(requests.exceptions.Timeout):
        bp._default_fetch(
            "https://example.org/img.jpg", dest, max_attempts=3, backoff_seconds=0
        )

    assert calls["n"] == 3  # bounded -- not unbounded retry
    assert not dest.exists()


def test_default_fetch_retries_5xx_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(status_code=503)
        return _FakeResponse(status_code=200, content=b"ok-bytes")

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "out" / "img.jpg"

    bp._default_fetch("https://example.org/img.jpg", dest, backoff_seconds=0)

    assert calls["n"] == 2
    assert dest.read_bytes() == b"ok-bytes"


def test_default_fetch_does_not_retry_4xx(tmp_path, monkeypatch):
    """A 404 will not fix itself -- must raise on the first attempt, no retry."""
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "out" / "img.jpg"

    with pytest.raises(requests.exceptions.HTTPError):
        bp._default_fetch(
            "https://example.org/img.jpg", dest, max_attempts=3, backoff_seconds=0
        )

    assert calls["n"] == 1


def test_stage_image_skips_and_continues_after_default_fetch_exhausts_retries(
    tmp_path, monkeypatch
):
    """Integration of the fix with the existing skip-and-continue design:
    once ``_default_fetch``'s own retries are exhausted, ``_stage_image``
    still returns ``None`` rather than raising (same contract as the
    ``_FakeFetcher`` case above, now exercised through the real retry path)."""

    def fake_get(url, timeout):
        raise requests.exceptions.ConnectionError("simulated persistent failure")

    monkeypatch.setattr(requests, "get", fake_get)
    wk1_dir = tmp_path / "wk1"
    store_dir = tmp_path / "store"
    row = ROWS[6]  # c_img1.jpg
    fetch_fn = functools.partial(bp._default_fetch, backoff_seconds=0)

    path = bp._stage_image(row, "900003", wk1_dir, store_dir, fetch_fn)

    assert path is None
