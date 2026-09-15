"""Report arithmetic. No models, no run directory -- just the bookkeeping.

The bug this file exists to prevent is documented in
findings/2026-07-29-orchestration-result.md §2: `status` says "unrepaired" both
when a case never needed repair and when repair failed.
"""
import json

import numpy as np
import pytest
from PIL import Image

from scripts.report import (_best_label, attach_density, bucket, render,
                            repair_rate, score_case, summarise)


def test_a_passing_draft_is_healthy():
    assert bucket({"draft": [True, 1.0]}) == "healthy"


def test_a_failing_draft_beaten_by_an_attempt_is_repaired():
    assert bucket({"draft": [False, 0.0],
                   "attempt_1": [True, 1.0]}) == "repaired"


def test_a_failing_draft_nothing_beat_is_unrepairable():
    assert bucket({"draft": [False, 0.0],
                   "attempt_1": [False, 0.0],
                   "attempt_2": [False, 0.0]}) == "unrepairable"


def test_repair_rate_excludes_healthy_cases_from_the_denominator():
    buckets = ["healthy", "repaired", "unrepairable"]
    #: 1 of the 2 that needed repair. `status` would give 1/3 or 2/3 -- both
    #: wrong, and both plausible enough to publish.
    assert repair_rate(buckets) == pytest.approx(0.5)


def test_repair_rate_is_undefined_when_nothing_needed_repair():
    #: Not 0.0. A rate over an empty denominator is not a number, and
    #: printing 0% would read as total failure.
    assert repair_rate(["healthy", "healthy"]) is None


def test_best_label_survives_a_score_with_no_grounded_evidence():
    """Scope extension: scores.json can now hold [ok, null] (no grounded
    evidence for that candidate). _best_label used to coerce every score
    through float(), which crashes on None; schedule.select_best already
    treats None as "cannot win", so the fix is to stop coercing it away.
    """
    assert _best_label({"draft": [False, None],
                        "attempt_1": [False, None]}) == "draft"


class _Enc:
    def embed(self, image):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_pil(self, images, batch_size=32):
        return np.array([[1.0, 0.0]] * len(list(images)), dtype=np.float32)

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(list(texts)), dtype=np.float32)

    def free(self):
        pass


class _Holder:
    def __init__(self):
        self.dino = _Enc()
        self.clip = _Enc()
        self.siglip = _Enc()


class _Case:
    id = "c"
    prompt = "a parrot"
    kind = "target"


def _write_case(d, scores, *, with_mask, ref):
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(scores))
    Image.new("RGB", (64, 64), (10, 20, 30)).save(d / "attempt_1.png")
    if with_mask:
        a = np.zeros((64, 64), dtype=np.uint8)
        a[8:56, 8:56] = 255
        Image.fromarray(a, "L").save(d / "mask.png")
    Image.new("RGB", (64, 64), (10, 20, 30)).save(ref)


def test_a_repaired_case_is_scored_on_the_crop(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 0.0], "attempt_1": [True, 1.0]},
                with_mask=True, ref=ref)

    got = score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref])

    assert got["bucket"] == "repaired"
    assert got["dino_cropped"] == pytest.approx(1.0)
    assert got["dino_whole"] is None
    assert got["preservation"] == 1.0


def test_open_loop_attempt_is_evaluated_without_fake_verifier_scores(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    d.mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(d / "attempt_1.png")
    a = np.zeros((64, 64), dtype=np.uint8)
    a[8:56, 8:56] = 255
    Image.fromarray(a, "L").save(d / "mask.png")
    Image.new("RGB", (64, 64), (10, 20, 30)).save(ref)

    got = score_case(
        _Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)), _Holder(),
        heldout_paths=[ref], selected_override="attempt_1")

    assert not (d / "scores.json").exists()
    assert got["selected"] == "attempt_1"
    assert got["bucket"] == "repaired"
    assert got["dino_delta"] == pytest.approx(0.0)


def test_a_healthy_case_has_no_mask_and_is_scored_whole(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [True, 1.0]}, with_mask=False, ref=ref)

    got = score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref])

    #: It never ran the `mask` stage, so there is no box to crop to. Its DINO
    #: goes in a separate column rather than into the cropped mean.
    assert got["bucket"] == "healthy"
    assert got["dino_cropped"] is None
    assert got["dino_whole"] == pytest.approx(1.0)


def test_a_mask_at_another_resolution_raises(tmp_path):
    #: The Global Constraint -- all metrics compute at the draft's size, size
    #: mismatch raises, never resizes -- was enforced on `output` only.
    #: crop_to_mask derives the bbox from the mask's grid and applies it to
    #: the draft's, clamped, so a mask at another resolution yields a wrong
    #: crop box on BOTH sides of the pair. Self-consistent, so the delta
    #: still prints a plausible number rather than failing visibly.
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 0.0], "attempt_1": [True, 1.0]},
                with_mask=False, ref=ref)
    a = np.zeros((32, 32), dtype=np.uint8)
    a[4:28, 4:28] = 255
    Image.fromarray(a, "L").save(d / "mask.png")

    with pytest.raises(ValueError, match="mask size"):
        score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                   _Holder(), heldout_paths=[ref])


def test_a_failed_draft_without_a_mask_raises(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 0.0], "attempt_1": [True, 1.0]},
                with_mask=False, ref=ref)

    #: This case should have run `mask`. A missing one means the run is
    #: damaged, and quietly scoring it whole-image would hide that.
    with pytest.raises(FileNotFoundError, match="mask"):
        score_case(_Case(), d, Image.new("RGB", (64, 64)), _Holder(),
                   heldout_paths=[ref])


def test_a_case_whose_mask_stage_failed_is_scored_not_raised(tmp_path):
    #: `mask` can fail honestly: GroundingDINO does not always find the coarse
    #: term in the draft, and run_pipeline records "ValueError: 'ship' did not
    #: ground in the draft" against stages.mask. That case has a failed draft
    #: and no mask.png -- the exact shape of a damaged run -- but it is a real
    #: outcome, not corruption. Raising on it aborts the whole report and
    #: loses the other 91 cases. doc-5's `pirate` and `model_t` are this.
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    _write_case(d, {"draft": [False, 1.0]}, with_mask=False, ref=ref)

    got = score_case(_Case(), d, Image.new("RGB", (64, 64), (10, 20, 30)),
                     _Holder(), heldout_paths=[ref], mask_failed=True)

    #: Unrepairable, not healthy: the draft failed and nothing repaired it.
    assert got["bucket"] == "unrepairable"
    assert got["mask_failed"] is True
    #: No box was ever found, so there is nothing to crop to. Whole-image
    #: DINO is the only honest number, and the delta stays unscored rather
    #: than defaulting to 0.0, which would read as "the repair changed
    #: nothing" when no repair was ever attempted.
    assert got["dino_cropped"] is None
    assert got["dino_delta"] is None
    assert got["dino_whole"] == pytest.approx(1.0)


def test_mask_failed_cases_are_counted_separately_in_the_summary():
    #: They sit inside `unrepairable`, so without their own count the table
    #: reads as "repair was attempted and lost" when repair never ran.
    rows = [_row("target", "unrepairable"), _row("target", "unrepairable")]
    rows[0]["mask_failed"] = True
    rows[1]["mask_failed"] = False

    got = summarise(rows)["target/bridge"]

    assert got["unrepairable"] == 2
    assert got["mask_failed"] == 1


def _row(kind, group, dino=None, whole=None, pres=None):
    #: `cohort`, `dino_cropped_draft`, `dino_delta` default to the values a
    #: pre-delta row would have -- no other test here exercises the delta --
    #: so summarise's direct indexing has something to find without changing
    #: any of these tests' assertions or outcomes.
    return {"case_id": "x", "kind": kind, "cohort": "bridge", "bucket": group,
            "selected": "d", "dino_cropped": dino,
            "dino_cropped_draft": None, "dino_delta": None,
            "dino_whole": whole, "preservation": pres,
            "clip": 0.3, "siglip": 0.4}


def test_summary_splits_targets_from_controls():
    rows = [_row("target", "repaired", dino=0.8, pres=1.0),
            _row("control", "healthy", whole=0.9)]
    got = summarise(rows)

    assert got["target/bridge"]["n"] == 1
    assert got["control/bridge"]["n"] == 1
    assert got["target/bridge"]["dino_cropped"] == pytest.approx(0.8)


def test_whole_image_dino_never_enters_the_cropped_mean():
    rows = [_row("target", "repaired", dino=0.8, pres=1.0),
            _row("target", "healthy", whole=0.2)]
    got = summarise(rows)

    #: 0.8, not 0.5. Averaging a whole-scene score into an object-crop mean
    #: compares two different measurements.
    assert got["target/bridge"]["dino_cropped"] == pytest.approx(0.8)
    assert got["target/bridge"]["dino_whole"] == pytest.approx(0.2)


def test_render_states_the_mechanism_and_refuses_a_pass_rate_headline():
    text = render(summarise([_row("target", "repaired", dino=0.8, pres=1.0)]),
                  {"mechanism": "stitch", "arm": "oracle", "run": "r",
                   "heldout_refs": 1})
    assert "stitch" in text
    #: The pipeline optimises against the verifier, so its pass-rate is a
    #: development signal only (RUNBOOK §5.2).
    assert "pass-rate" not in text.lower() or "development signal" in text.lower()


def test_a_passing_attempt_beats_an_equal_scoring_failed_draft():
    """The bucket must agree with the run's own `best`, tie or no tie.

    Observed on `axolotl` in outputs/doc4_20260731_095641: draft
    [False, 1.0], attempt_1 [True, 1.0]. select_best gives attempt_1 -- a
    passing attempt wins outright -- so the pipeline recorded `repaired`.
    Comparing scores alone calls the same case `unrepairable` and quietly
    understates the headline rate. Spec §4: repaired = best != "draft".
    """
    assert bucket({"draft": [False, 1.0],
                   "attempt_1": [True, 1.0]}) == "repaired"


def test_the_bucket_never_disagrees_with_select_best():
    from ragregen import schedule

    for scores in ({"draft": [False, 1.0], "attempt_1": [True, 1.0]},
                   {"draft": [False, 0.4], "attempt_1": [False, 0.9]},
                   {"draft": [False, 0.9], "attempt_1": [False, 0.4]},
                   {"draft": [False, 0.0], "attempt_1": [False, 0.0]}):
        attempts = [(k, bool(v[0]), float(v[1]))
                    for k, v in sorted(scores.items()) if k != "draft"]
        best = schedule.select_best(float(scores["draft"][1]), attempts)
        want = "repaired" if best != "draft" else "unrepairable"
        assert bucket(scores) == want, scores


class _CohortCase:
    id = "c"
    prompt = "a parrot"
    kind = "control"
    cohort = "common"


#: Canvas side length for the crop-geometry fixture below. Large enough that
#: the padded mask bbox lands strictly inside the canvas -- not clipped to
#: its edges -- so the crop is a genuine, off-centre sub-region rather than
#: the whole frame, and a wrong crop box actually captures different pixels.
_GEO_W = 100
_GEO_BG = 20  #: background grey level, identical on draft/output/ref


def _geo_mask():
    a = np.zeros((_GEO_W, _GEO_W), dtype=np.uint8)
    a[20:60, 20:60] = 255
    return Image.fromarray(a, "L")


def _geo_image(patch_val):
    #: A bright square on a dark field, at the SAME location the mask marks.
    #: crop_to_mask pads the mask bbox by ~13px on this geometry, giving a
    #: (7, 7, 72, 72) crop -- 65x65, not the full 100x100 canvas -- so the
    #: crop mean mixes 1600 patch pixels with 2625 background pixels. Shift
    #: or resize that box and the mix, hence the mean, changes.
    arr = np.full((_GEO_W, _GEO_W, 3), _GEO_BG, dtype=np.uint8)
    arr[20:60, 20:60] = patch_val
    return Image.fromarray(arr, "RGB")


class _CropAwareEnc:
    """Embeds a crop by what it actually contains: mean intensity and
    footprint (area as a fraction of the canvas). A crop taken at the wrong
    position captures a different patch/background mix -- a different mean --
    and a crop of the wrong size gets a different footprint. Either makes the
    embedding, and so the pairing invariant, observably different -- unlike a
    flat-colour fixture, where any crop of a solid fill is indistinguishable
    from any other.
    """

    def embed(self, image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        mean = arr.mean() / 255.0
        w, h = image.size
        footprint = (w * h) / (_GEO_W * _GEO_W)
        return np.array([mean, footprint], dtype=np.float32)

    def encode_pil(self, images, batch_size=32):
        return np.array([[1.0, 0.0]] * len(list(images)), dtype=np.float32)

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(list(texts)), dtype=np.float32)

    def free(self):
        pass


class _CropAwareHolder:
    def __init__(self):
        self.dino = _CropAwareEnc()
        self.clip = _Enc()
        self.siglip = _Enc()


def test_a_repaired_case_reports_both_sides_of_the_pair(tmp_path):
    d = tmp_path / "c"
    ref = tmp_path / "held.jpg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(
        {"draft": [False, 0.0], "attempt_1": [True, 1.0]}))
    #: bright output patch (220), dark draft patch (80), reference matches
    #: the output (220) -- so the pair actually diverges in identity, and a
    #: wrong crop box (different position or size than the mask's) would
    #: mix in a different amount of background and miss these numbers.
    _geo_image(220).save(d / "attempt_1.png")
    _geo_mask().save(d / "mask.png")
    _geo_image(220).save(ref)

    got = score_case(_CohortCase(), d, _geo_image(80),
                     _CropAwareHolder(), heldout_paths=[ref])

    #: Computed from the real crop_to_mask bbox on this geometry -- (7, 7,
    #: 72, 72), 65x65 -- mixing 1600 patch px with 2625 background px, then
    #: through _CropAwareEnc.embed. See the module docstring above for the
    #: derivation; verified against an actual metrics.crop_to_mask() call.
    assert got["dino_cropped"] == pytest.approx(0.31947, abs=1e-4)
    assert got["dino_cropped_draft"] == pytest.approx(0.24141, abs=1e-4)
    assert got["dino_delta"] == pytest.approx(0.07806, abs=1e-4)
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


def test_the_three_counts_are_a_total_partition_of_scored_repairs():
    #: A real edit whose delta lands at exactly 0.0 must still land
    #: somewhere: it fails `improved`'s `> 0`, `worsened`'s `< 0`, and (before
    #: this fix) `unchanged`'s `selected == "draft"` too, silently vanishing
    #: from the +/=/- column. Exact zero on a real edit reads as "=".
    rows = [_drow("control", "common", "repaired", delta=0.2),
            _drow("control", "common", "unrepairable", delta=-0.1),
            _drow("control", "common", "unrepairable", delta=0.0,
                  selected="draft"),
            _drow("control", "common", "repaired", delta=0.0)]
    got = summarise(rows)["control/common"]
    scored = [r for r in rows if r["dino_delta"] is not None]
    assert (got["improved"], got["worsened"], got["unchanged"]) == (1, 1, 2)
    assert got["improved"] + got["unchanged"] + got["worsened"] == len(scored)


def test_the_summary_carries_the_deltas_own_denominator():
    #: `n` is the whole stratum; the CI and dino_delta are computed over the
    #: repair cases with a scored delta. A row reading "n 70 | healthy 68 |
    #: [+0.01, +0.01]" invites attributing the interval to 70 cases when it
    #: came from 2, so the count travels with the interval.
    rows = [_drow("control", "common", "healthy", delta=0.0,
                  selected="draft"),
            _drow("control", "common", "healthy", delta=0.0,
                  selected="draft"),
            _drow("control", "common", "repaired", delta=0.2),
            _drow("control", "common", "unrepairable", delta=-0.1),
            _drow("control", "common", "unrepairable", delta=None)]
    got = summarise(rows)["control/common"]
    assert got["n"] == 5
    #: two non-healthy rows with a non-None delta; the third is unscored.
    assert got["n_delta"] == 2


def test_render_shows_the_cis_denominator_and_says_so():
    summary = summarise([_drow("control", "common", "repaired", dino=0.8,
                               draft=0.6, delta=0.2)])
    text = render(summary, {"run": "r", "arm": "full", "mechanism": "inpaint",
                            "heldout_refs": 1})
    assert "(n=1)" in text
    assert "repair cases only" in text


def test_a_stratum_below_the_bootstrap_floor_gets_no_interval():
    #: A 1-2 case repair stratum is a likely outcome with 11 bridge controls
    #: mostly expected healthy. It must render "—", not a zero-width
    #: interval carrying a verdict.
    rows = [_drow("control", "common", "repaired", dino=0.8, draft=0.79,
                  delta=0.01)]
    got = summarise(rows)["control/common"]
    assert got["n_delta"] == 1
    assert got["delta_ci"] is None
    assert got["delta_verdict"] is None


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


def test_density_lookup_lowercases_the_mixed_case_dataset_concept(tmp_path):
    #: corpus_manifest.json's per_concept keys are lowercase (fetch_corpus.py
    #: normalises them there, because configs/dataset.yaml's "Boston bull"
    #: and configs/imagenet_coarse.yaml's "boston bull" would otherwise
    #: collide into two keys). `case.concept` itself stays mixed-case for all
    #: 22 bridge cases, so the lookup must lowercase before it -- transcribed
    #: verbatim from the brief this collapses to None for every bridge case.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"per_concept": {"boston bull": 37}}))

    class _C:
        id, concept = "c", "Boston bull"

    rows = [{"case_id": "c"}]
    attach_density(rows, m, [_C()])
    assert rows[0]["corpus_density"] == 37


def test_render_explains_no_harm_is_not_zero_change():
    #: delta_verdict checks `lo >= MARGIN` before `hi < 0`, so an interval
    #: entirely below zero but above the margin -- e.g. (-0.015, -0.005) --
    #: still reads `no_harm`. That is the correct non-inferiority test; the
    #: legend must say so where the reader actually is.
    summary = summarise([_drow("control", "common", "repaired", dino=0.8,
                               draft=0.6, delta=0.2)])
    text = render(summary, {"run": "r", "arm": "full", "mechanism": "inpaint",
                            "heldout_refs": 1})
    assert "does not mean" in text.lower()
