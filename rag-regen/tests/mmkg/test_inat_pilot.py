"""Offline tests for the iNaturalist birds discrimination pilot: FAKE VLM,
FAKE encoder (never invoked -- ``--skip-baseline``). No torch, no faiss, no
GPU -- only PIL (already required transitively) and the frozen mmkg modules
under test.
"""
import json
import os

from PIL import Image

from ragregen.mmkg import evaluate, inat_pilot, schema


# ---------------------------------------------------------------------------
# §2 selection
# ---------------------------------------------------------------------------

def _cat(id_, family, klass="Aves"):
    return {"id": id_, "class": klass, "family": family, "image_dir_name": f"dir_{id_}"}


def test_select_pilot_species_frozen_order():
    categories = (
        # family "Zebra": 3 species (count=3), ids out of order to test
        # within-family id-ascending sort.
        [_cat(30, "Zebra"), _cat(10, "Zebra"), _cat(20, "Zebra")]
        # family "Aardvark": 4 species (count=4) -- must sort AFTER Zebra
        # despite the alphabetically-earlier name, because family order is
        # (species-count ascending, then name).
        + [_cat(103, "Aardvark"), _cat(100, "Aardvark"), _cat(102, "Aardvark"), _cat(101, "Aardvark")]
        # family "Excluded": below min_family_size -- must never appear.
        + [_cat(1, "Excluded"), _cat(2, "Excluded")]
        # a non-Aves category -- must never appear.
        + [_cat(999, "Zebra", klass="Mammalia")]
    )
    val_json = {"categories": categories}

    selected = inat_pilot.select_pilot_species(val_json, min_species=6)

    # whole families only, smaller family (count 3) before larger (count 4)
    # even though "Aardvark" < "Zebra" alphabetically; species sorted by id
    # ascending within each family; "Excluded" and the Mammalia row absent.
    assert [s["id"] for s in selected] == [10, 20, 30, 100, 101, 102, 103]
    assert {s["family"] for s in selected} == {"Zebra", "Aardvark"}


def test_select_pilot_species_family_name_tiebreak():
    # Two families tied on species-count (3 each) -> sorted by family name.
    categories = (
        [_cat(i, "Bravo") for i in (13, 11, 12)]
        + [_cat(i, "Alpha") for i in (23, 21, 22)]
    )
    val_json = {"categories": categories}
    selected = inat_pilot.select_pilot_species(val_json, min_species=6)
    assert [s["id"] for s in selected] == [21, 22, 23, 11, 12, 13]


# ---------------------------------------------------------------------------
# Shared fixture: 2 families x 3 species x 10 images
# ---------------------------------------------------------------------------

# One solid fill color per species -- reused across all 10 of its images,
# since every canned read for a species is identical regardless of which of
# its images is asked about. Opened+converted images lose PIL's .filename
# attribute (confirmed: Image.open(p).convert("RGB").filename is None), so
# FakeAsker decodes the species from the pixel color, not the path -- the
# same convention as tests/mmkg/test_run_smoke.py's FakeVLM.
SPECIES_COLORS = {
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
}


class FakeAsker:
    """.ask(image, prompt) -> str. Read prompts are answered from a
    per-pixel-color canned JSON reply table. Judge prompts ("essentially the
    same") always answer "no" -- safe here because every SAME-species
    comparison in this fixture is an exact lexical match (short-circuited by
    ``verifier.attribute_states`` before the judge is ever called), so the
    judge is only ever exercised on genuinely-differing SIBLING profiles."""

    def __init__(self, reads_by_color):
        self.reads_by_color = reads_by_color
        self.model_id = "fake-vlm"
        self.n_read_calls = 0

    def ask(self, image, prompt):
        if "essentially the same" in prompt:
            return "no"
        self.n_read_calls += 1
        pixel = image.getpixel((0, 0))
        return self.reads_by_color.get(pixel, "{}")


def _mk_image(path, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # format="PNG" explicitly: Image.save() otherwise infers JPEG from the
    # ".jpg" extension (kept for realism -- real val.json file_names are
    # .jpg), and JPEG's lossy DCT/YCbCr round-trip shifts a solid fill by
    # +/-1 per channel, breaking the pixel-exact color lookup FakeAsker
    # relies on.
    Image.new("RGB", (4, 4), color).save(path, format="PNG")


def _profile_json(tag):
    return json.dumps({
        "primary_plumage_color": f"{tag}-color",
        "secondary_plumage_color": f"{tag}-secondary",
        "plumage_pattern": f"{tag}-pattern",
        "bill_shape": f"{tag}-billshape",
        "bill_color": f"{tag}-billcolor",
        "distinctive_markings": f"{tag}-markings",
    })


def _build_fixture(tmp_path):
    """2 families ("F1": species 1,2,3; "F2": species 4,5,6), 10 images each
    (7 build + 3 test). Every image's canned read equals its species'
    consensus profile, so SAME pairs are exact-match PASS and every SIBLING
    pair is fully contradicted on all 6 slots -> FAIL."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    categories = [
        {"id": 1, "class": "Aves", "family": "F1", "image_dir_name": "sp1"},
        {"id": 2, "class": "Aves", "family": "F1", "image_dir_name": "sp2"},
        {"id": 3, "class": "Aves", "family": "F1", "image_dir_name": "sp3"},
        {"id": 4, "class": "Aves", "family": "F2", "image_dir_name": "sp4"},
        {"id": 5, "class": "Aves", "family": "F2", "image_dir_name": "sp5"},
        {"id": 6, "class": "Aves", "family": "F2", "image_dir_name": "sp6"},
    ]

    images, annotations, reads_by_path = [], [], {}
    image_id = 0
    for sp in categories:
        tag = f"sp{sp['id']}"
        color = SPECIES_COLORS[sp["id"]]
        for k in range(10):
            image_id += 1
            rel = f"val/{sp['image_dir_name']}/img{k}.jpg"
            abs_path = str(data_root / rel)
            _mk_image(abs_path, color)
            images.append({"id": image_id, "file_name": rel})
            annotations.append({"image_id": image_id, "category_id": sp["id"]})
            reads_by_path[abs_path] = _profile_json(tag)

    reads_by_color = {SPECIES_COLORS[sp["id"]]: _profile_json(f"sp{sp['id']}") for sp in categories}

    val_json = {"categories": categories, "images": images, "annotations": annotations}
    return val_json, str(data_root), reads_by_path, reads_by_color


def test_discrimination_cases_labels_and_reads_each_test_image_once(tmp_path):
    val_json, data_root, reads_by_path, _reads_by_color = _build_fixture(tmp_path)
    asker = FakeAsker({})  # judge-only here; reads come from a direct read_fn below

    species = inat_pilot.select_pilot_species(val_json, min_species=6)
    assert len(species) == 6

    read_calls = []

    def read_fn(path):
        read_calls.append(path)
        return json.loads(reads_by_path[path])

    def judge_fn(x, y, slot):
        return inat_pilot.judge_bird_match(x, y, slot, asker)

    profiles = inat_pilot.build_profiles(species, val_json, data_root, read_fn)
    for sp in species:
        assert schema.decidable(profiles[sp["id"]])

    read_calls.clear()  # only count test-image reads from here
    cases = inat_pilot.discrimination_cases(species, val_json, data_root, profiles, read_fn, judge_fn)

    # 6 species x 3 test images x (1 same + 2 siblings [family size 3]) = 54
    assert len(cases) == 54
    same_cases = [c for c in cases if c["candidate"] == c["species"]]
    sib_cases = [c for c in cases if c["candidate"] != c["species"]]
    assert len(same_cases) == 18 and len(sib_cases) == 36
    assert all(c["label"] == "pass" for c in same_cases)
    assert all(c["label"] == "fail" for c in sib_cases)
    assert all(c["verdict"] == "PASS" for c in same_cases)   # exact match, no judge needed
    assert all(c["verdict"] == "FAIL" for c in sib_cases)    # fully contradicted profiles
    assert all({"verdict", "label", "species", "candidate", "family"} == set(c) for c in cases)

    # each test image (18 total: 6 species x 3) read exactly once
    assert len(read_calls) == 18
    assert len(read_calls) == len(set(read_calls))

    conf = evaluate.confusion(cases)
    assert conf["recall"] == 1.0
    assert conf["fp_rate"] == 0.0


# ---------------------------------------------------------------------------
# §6 gate
# ---------------------------------------------------------------------------

def test_pilot_gate_exact_bar():
    def _conf(recall, fp):
        return {"recall": recall, "fp_rate": fp, "n_abstain": 0}

    g = inat_pilot.pilot_gate(_conf(0.5, 0.1), n_decidable=20)
    assert g["decision"] == "PASS"

    g = inat_pilot.pilot_gate(_conf(0.3, 0.1), n_decidable=20)  # recall < 0.40
    assert g["decision"] == "NEGATIVE"

    g = inat_pilot.pilot_gate(_conf(0.5, 0.1), n_decidable=17)  # under-powered
    assert g["decision"] == "NEGATIVE"

    g = inat_pilot.pilot_gate(_conf(0.55, 0.4), n_decidable=20)  # recall-fp < 0.20
    assert g["decision"] == "NEGATIVE"

    # boundary: exactly at every threshold -> PASS
    g = inat_pilot.pilot_gate(_conf(0.4, 0.2), n_decidable=18)
    assert g["decision"] == "PASS"


# ---------------------------------------------------------------------------
# main() end-to-end with injected fakes
# ---------------------------------------------------------------------------

def test_main_writes_result_and_traces(tmp_path):
    val_json, data_root, _reads_by_path, reads_by_color = _build_fixture(tmp_path)
    val_json_p = tmp_path / "val.json"
    val_json_p.write_text(json.dumps(val_json))

    asker = FakeAsker(reads_by_color)
    out_dir = tmp_path / "out"

    result = inat_pilot.main(
        ["--skip-baseline", "--min-species", "6"],
        vlm_factory=lambda: asker,
        val_json_path=str(val_json_p),
        data_root=data_root,
        out_dir=str(out_dir),
    )

    result_path = out_dir / "result.json"
    assert result_path.is_file()
    on_disk = json.loads(result_path.read_text())
    assert on_disk == result

    assert "gate" in result and "decision" in result["gate"]
    assert result["decision"] in {"PASS", "NEGATIVE"}
    # this fixture's signal is clean (sibling_recall=1.0, true_fp=0.0) --
    # pinned in the confusion assert below -- but the frozen gate also
    # requires n_decidable >= 18 (spec §6), which a 6-species fixture can
    # never reach, so the *decision* itself is correctly NEGATIVE here
    # (under-powered), exercised at the exact bar by test_pilot_gate_exact_bar.
    assert result["decision"] == "NEGATIVE"
    assert result["gate"]["sibling_recall"] == 1.0
    assert result["gate"]["true_fp"] == 0.0
    assert result["n_decidable"] == 6
    assert result["baseline"] is None  # --skip-baseline: no encoder touched

    trace_dir = out_dir / "trace"
    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) >= 1
    assert len(trace_files) == 18  # one per test image (6 species x 3)

    one = json.loads(trace_files[0].read_text())
    assert {"species", "family", "image_path", "read", "same_verdict", "sibling_verdicts"} <= set(one)

    # 42 build reads + 18 test reads = 60 distinct images; each image is
    # read exactly once even though discrimination_cases and the trace
    # builder both consume the (cached) reads.
    assert asker.n_read_calls == 60
