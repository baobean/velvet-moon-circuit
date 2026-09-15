"""iNaturalist birds discrimination PILOT (frozen spec:
``docs/superpowers/specs/2026-09-09-mmkg-inat-birds-pilot-design.md``).

Re-tests the MMKG core hypothesis -- "a per-species attribute profile
discriminates a bird from its same-family siblings" -- on a clean,
name-blind substrate (iNaturalist 2021 birds, authoritative species labels
as ground truth) instead of the corpus-confounded LAION/ImageNet-holdout
Stage-1 run.

Flow (see ``main``):

1. Load ``val.json`` (categories/images/annotations) read-only.
2. :func:`select_pilot_species` -- deterministic §2 selection (whole
   families, ascending species-count then family name, until >=24 species).
3. :func:`build_profiles` -- one VLM read per build image (first 7/species,
   sorted by image id), consensus via :func:`bird_target_attributes` (the
   same rule as ``schema.target_attributes``, over the 6 bird slots).
4. :func:`discrimination_cases` -- for each decidable species' 3 test
   images (images 8-10), one SAME pair (label "pass") and up to
   ``N_SIBLINGS`` SIBLING pairs against in-set same-family siblings nearest
   by category-id distance (label "fail"), scored with
   :func:`ragregen.mmkg.verifier.case_verdict`. Each test image is read
   once.
5. :func:`ragregen.mmkg.evaluate.confusion` + :func:`pilot_gate` -- the
   pre-registered pilot bar (sibling_recall - true_fp >= 0.20, sibling_recall
   >= 0.40, n_decidable >= 18).
6. :func:`siglip_baseline` (descriptive, not gating) -- reported alongside.
7. ``result.json`` (selection, per-species decidable flags, confusion,
   gate, baseline) + one trace file per test image, under ``out_dir``.

``main`` takes injectable ``vlm_factory``/``encoder_factory`` seams (real
``QwenVLM``/SigLIP by default, imported lazily inside the factory) so an
offline smoke test can drive the whole flow with fakes and never import
torch or faiss (see ``tests/mmkg/test_inat_pilot.py``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

from ragregen.mmkg import evaluate, schema, verifier

# ---------------------------------------------------------------------------
# Frozen constants (spec §1-6)
# ---------------------------------------------------------------------------

BIRD_SLOTS = ["primary_plumage_color", "secondary_plumage_color", "plumage_pattern",
              "bill_shape", "bill_color", "distinctive_markings"]

BIRD_READ_PROMPT = (
    "Describe ONLY the single bird in this image. Do not name the species. "
    "Return a JSON object with exactly these keys: primary_plumage_color, "
    "secondary_plumage_color, plumage_pattern, bill_shape, bill_color, "
    "distinctive_markings. Each value is a short phrase, or \"not visible\" "
    "if you cannot tell. Output only the JSON."
)

JUDGE_PROMPT = (
    "You are comparing two descriptions of a bird's {slot}. A: \"{x}\". B: \"{y}\". "
    "Do A and B describe essentially the same {slot}? Answer with only 'yes' or 'no'."
)

N_BUILD = 7
N_TEST = 3
N_SIBLINGS = 3

DEFAULT_MIN_FAMILY_SIZE = 3
DEFAULT_MIN_SPECIES = 24

DEFAULT_VAL_JSON = "/mmlabworkspace_new/Students/tuanld/soict-2026-data/inat2021_birds/val.json"
DEFAULT_DATA_ROOT = "/mmlabworkspace_new/Students/tuanld/soict-2026-data/inat2021_birds"


# ---------------------------------------------------------------------------
# Read / judge (local -- mirrors ragregen.mmkg.vlm_read, bird ontology)
# ---------------------------------------------------------------------------

def _parse_json(text):
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def read_bird_slots(image_path, asker):
    """Open the image RGB, ask the bird READ_PROMPT, parse the JSON reply,
    fill any missing/blank key with "not visible". Mirrors
    ``ragregen.mmkg.vlm_read.read_slots`` over ``BIRD_SLOTS``."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    obj = _parse_json(asker.ask(img, BIRD_READ_PROMPT))
    return {slot: (str(obj[slot]) if slot in obj and str(obj[slot]).strip() else "not visible")
            for slot in BIRD_SLOTS}


def _yes(r):
    return str(r).strip().lower().startswith("yes")


def judge_bird_match(x, y, slot, asker):
    """Bidirectional yes/no judge over the bird JUDGE_PROMPT. Mirrors
    ``ragregen.mmkg.vlm_read.judge_match``."""
    if schema.norm(x) in schema.NOT_VISIBLE or schema.norm(y) in schema.NOT_VISIBLE:
        return False
    from PIL import Image

    blank = Image.new("RGB", (8, 8), "white")
    a = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=x, y=y))
    b = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=y, y=x))
    return _yes(a) and _yes(b)


# ---------------------------------------------------------------------------
# §2 selection
# ---------------------------------------------------------------------------

def select_pilot_species(val_json, *, min_family_size=DEFAULT_MIN_FAMILY_SIZE,
                          min_species=DEFAULT_MIN_SPECIES):
    """Deterministic bird-species selection (spec §2).

    Aves categories only; keep families with >= ``min_family_size`` species;
    sort those families by (species-count ascending, family name); take
    whole families in that order, species within a family sorted by iNat
    ``id`` ascending, accumulating until >= ``min_species`` species across
    the fewest whole families that reach it (never a partial family).
    """
    categories = val_json.get("categories", [])
    aves = [c for c in categories if c.get("class") == "Aves"]

    by_family = defaultdict(list)
    for c in aves:
        by_family[c["family"]].append(c)

    families = [fam for fam, specs in by_family.items() if len(specs) >= min_family_size]
    families.sort(key=lambda fam: (len(by_family[fam]), fam))

    selected = []
    for fam in families:
        specs = sorted(by_family[fam], key=lambda c: c["id"])
        selected.extend(specs)
        if len(selected) >= min_species:
            break
    return selected


# ---------------------------------------------------------------------------
# Image paths / build-test split
# ---------------------------------------------------------------------------

def species_image_paths(val_json, category_id, data_root):
    """Every ``val`` image path for ``category_id``, sorted by image id."""
    img_ids = {a["image_id"] for a in val_json.get("annotations", [])
               if a["category_id"] == category_id}
    imgs = [im for im in val_json.get("images", []) if im["id"] in img_ids]
    imgs.sort(key=lambda im: im["id"])
    return [os.path.join(data_root, im["file_name"]) for im in imgs]


def _split_build_test(paths):
    """First N_BUILD = build, next N_TEST = test. Asserts they never
    overlap (spec §3)."""
    build, test = paths[:N_BUILD], paths[N_BUILD:N_BUILD + N_TEST]
    assert not (set(build) & set(test)), "build/test image overlap"
    return build, test


# ---------------------------------------------------------------------------
# §4 consensus (bird slots -- NOT schema.target_attributes, which iterates
# schema.SLOTS, the arbitrary-object slots)
# ---------------------------------------------------------------------------

def bird_target_attributes(reads):
    """Same consensus rule as ``schema.target_attributes``
    (V_vis >= 3, plurality support >= max(2, ceil(0.5*V_vis))), applied over
    ``BIRD_SLOTS``. Output dict shape is identical, so
    ``ragregen.mmkg.verifier.case_verdict`` (which only iterates the keys
    of the targets dict) consumes it unchanged."""
    out = {}
    for slot in BIRD_SLOTS:
        vals = [schema.norm(r.get(slot)) for r in reads]
        vis = [v for v in vals if v not in schema.NOT_VISIBLE]
        if not vis:
            continue
        cnt = Counter(vis)
        value, support = cnt.most_common(1)[0]
        thr = max(2, math.ceil(0.5 * len(vis)))
        is_target = support >= thr and len(vis) >= 3
        out[slot] = {"value": value, "support": support,
                     "visible_count": len(vis), "is_target": is_target}
    return out


# ---------------------------------------------------------------------------
# §3/§4 build: per-species target profile from its 7 build images
# ---------------------------------------------------------------------------

def build_profiles(species, val_json, data_root, read_fn):
    """{category_id: targets} -- one consensus profile per selected species,
    read from its 7 build images only."""
    profiles = {}
    for sp in species:
        paths = species_image_paths(val_json, sp["id"], data_root)
        build_paths, _test_paths = _split_build_test(paths)
        reads = [read_fn(p) for p in build_paths]
        profiles[sp["id"]] = bird_target_attributes(reads)
    return profiles


def _siblings(by_family, sp):
    """Up to N_SIBLINGS other in-set same-family species, nearest by
    |id - id| (spec §5)."""
    return sorted(
        (s for s in by_family[sp["family"]] if s["id"] != sp["id"]),
        key=lambda s: abs(s["id"] - sp["id"]),
    )[:N_SIBLINGS]


# ---------------------------------------------------------------------------
# §5 discrimination protocol
# ---------------------------------------------------------------------------

def discrimination_cases(species, val_json, data_root, profiles, read_fn, judge_fn):
    """For each test image of each *decidable* species S: one SAME pair
    (label "pass") via case_verdict(profile[S], read(t)), and up to
    N_SIBLINGS SIBLING pairs (label "fail") against in-set same-family
    siblings nearest by category-id. Each test image is read exactly once
    -- the same read is scored against S and every sibling."""
    by_family = defaultdict(list)
    for sp in species:
        by_family[sp["family"]].append(sp)

    cases = []
    for sp in species:
        targets = profiles[sp["id"]]
        if not schema.decidable(targets):
            continue
        paths = species_image_paths(val_json, sp["id"], data_root)
        _build_paths, test_paths = _split_build_test(paths)
        siblings = _siblings(by_family, sp)

        for t in test_paths:
            read = read_fn(t)  # read once, reused for SAME + every SIBLING

            same = verifier.case_verdict(targets, read, judge_fn)
            cases.append({"verdict": same["verdict"], "label": "pass",
                          "species": sp["id"], "candidate": sp["id"], "family": sp["family"]})

            for sib in siblings:
                sib_verdict = verifier.case_verdict(profiles[sib["id"]], read, judge_fn)
                cases.append({"verdict": sib_verdict["verdict"], "label": "fail",
                              "species": sp["id"], "candidate": sib["id"], "family": sp["family"]})
    return cases


def _trace_records(species, val_json, data_root, profiles, read_fn, judge_fn):
    """One record per test image (species, family, image path, read, SAME
    verdict, per-sibling verdicts) -- the tracing counterpart of
    ``discrimination_cases``. ``read_fn``/``judge_fn`` are expected to be
    memoized by the caller, so this costs no extra VLM calls."""
    by_family = defaultdict(list)
    for sp in species:
        by_family[sp["family"]].append(sp)

    records = []
    for sp in species:
        targets = profiles[sp["id"]]
        if not schema.decidable(targets):
            continue
        paths = species_image_paths(val_json, sp["id"], data_root)
        _build_paths, test_paths = _split_build_test(paths)
        siblings = _siblings(by_family, sp)

        for t in test_paths:
            read = read_fn(t)
            same = verifier.case_verdict(targets, read, judge_fn)
            sib_results = [{"candidate": sib["id"],
                            "verdict": verifier.case_verdict(profiles[sib["id"]], read, judge_fn)}
                           for sib in siblings]
            records.append({"species": sp["id"], "family": sp["family"], "image_path": t,
                            "read": read, "same_verdict": same, "sibling_verdicts": sib_results})
    return records


def write_traces(records, out_dir):
    trace_dir = os.path.join(out_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    paths = []
    for idx, rec in enumerate(records):
        p = os.path.join(trace_dir, f"{rec['species']}_{idx}.json")
        with open(p, "w") as f:
            json.dump(rec, f, indent=2)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# §6 gate
# ---------------------------------------------------------------------------

def pilot_gate(conf, n_decidable):
    """Pre-registered pilot bar (spec §6): sibling_recall - true_fp >= 0.20,
    sibling_recall >= 0.40, and n_decidable >= 18."""
    recall = conf["recall"]
    fp = conf["fp_rate"]
    decision = "PASS" if (recall - fp >= 0.20 and recall >= 0.40 and n_decidable >= 18) else "NEGATIVE"
    return {"sibling_recall": recall, "true_fp": fp, "n_decidable": n_decidable,
            "n_abstain": conf.get("n_abstain"), "decision": decision}


# ---------------------------------------------------------------------------
# Baseline (descriptive, not gating) -- behind the encoder seam
# ---------------------------------------------------------------------------

def siglip_baseline(species, val_json, data_root, profiles, encoder):
    """Nearest build-centroid baseline: among {S} u siblings, does SigLIP
    cosine similarity of a test image to each candidate species' build-image
    centroid pick the true species S? Reported, not gating (spec §6)."""
    import numpy as np

    by_family = defaultdict(list)
    for sp in species:
        by_family[sp["family"]].append(sp)

    centroids = {}
    for sp in species:
        build_paths, _test_paths = _split_build_test(species_image_paths(val_json, sp["id"], data_root))
        if not build_paths:
            continue
        emb = np.asarray(encoder.encode_images(build_paths), dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids[sp["id"]] = (emb / norms).mean(axis=0)

    correct = total = 0
    for sp in species:
        if not schema.decidable(profiles.get(sp["id"], {})):
            continue
        if sp["id"] not in centroids:
            continue
        siblings = [s for s in _siblings(by_family, sp) if s["id"] in centroids]
        candidates = [sp] + siblings
        _build_paths, test_paths = _split_build_test(species_image_paths(val_json, sp["id"], data_root))
        if not test_paths:
            continue
        emb = np.asarray(encoder.encode_images(test_paths), dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        for te in emb:
            sims = {c["id"]: float(np.dot(te, centroids[c["id"]])) for c in candidates}
            pred = max(sims, key=sims.get)
            total += 1
            if pred == sp["id"]:
                correct += 1
    return {"accuracy": (correct / total) if total else None, "n": total}


# ---------------------------------------------------------------------------
# Default (real) factories -- imported lazily so importing this module pulls
# no torch/faiss.
# ---------------------------------------------------------------------------

def _default_vlm_factory():
    from ragregen.vlm import VLM_ID, QwenVLM

    vlm = QwenVLM()
    if not hasattr(vlm, "model_id"):
        vlm.model_id = VLM_ID
    return vlm


def _default_encoder_factory(name):
    from ragregen.encoders import build_encoder

    return build_encoder(name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv, val_json_path, data_root, out_dir):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-json", default=val_json_path)
    p.add_argument("--data-root", default=data_root)
    p.add_argument("--out", default=out_dir)
    p.add_argument("--encoder", default="siglip_so400m_384")
    p.add_argument("--min-species", type=int, default=DEFAULT_MIN_SPECIES)
    p.add_argument("--min-family-size", type=int, default=DEFAULT_MIN_FAMILY_SIZE)
    p.add_argument("--skip-baseline", action="store_true",
                    help="skip the SigLIP baseline (no encoder needed)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None, *, vlm_factory=None, encoder_factory=None,
         val_json_path=DEFAULT_VAL_JSON, data_root=DEFAULT_DATA_ROOT,
         out_dir="outputs/mmkg_inat_pilot"):
    args = _parse_args(argv, val_json_path, data_root, out_dir)
    os.makedirs(args.out, exist_ok=True)

    with open(args.val_json) as f:
        val_json = json.load(f)

    species = select_pilot_species(
        val_json, min_family_size=args.min_family_size, min_species=args.min_species)

    vlm = (vlm_factory or _default_vlm_factory)()
    vlm_id = getattr(vlm, "model_id", type(vlm).__name__)

    reads_cache = {}

    def read_fn(path):
        if path not in reads_cache:
            reads_cache[path] = read_bird_slots(path, vlm)
        return reads_cache[path]

    judge_cache = {}

    def judge_fn(x, y, slot):
        key = (x, y, slot)
        if key not in judge_cache:
            judge_cache[key] = judge_bird_match(x, y, slot, vlm)
        return judge_cache[key]

    # --- §3/§4 build ---------------------------------------------------------
    profiles = build_profiles(species, val_json, args.data_root, read_fn)
    n_decidable = sum(1 for sp in species if schema.decidable(profiles[sp["id"]]))

    # --- §5 discrimination + §6 offline eval ---------------------------------
    cases = discrimination_cases(species, val_json, args.data_root, profiles, read_fn, judge_fn)
    conf = evaluate.confusion(cases)
    gate = pilot_gate(conf, n_decidable)

    # --- baseline (descriptive, not gating) -----------------------------------
    baseline = None
    if not args.skip_baseline:
        encoder = (encoder_factory or (lambda: _default_encoder_factory(args.encoder)))()
        baseline = siglip_baseline(species, val_json, args.data_root, profiles, encoder)

    # --- traces (reuses the cached reads/judges -- no extra VLM calls) -------
    records = _trace_records(species, val_json, args.data_root, profiles, read_fn, judge_fn)
    trace_paths = write_traces(records, args.out)

    per_species = {
        str(sp["id"]): {
            "family": sp["family"],
            "decidable": schema.decidable(profiles[sp["id"]]),
            "n_target_attributes": schema.n_targets(profiles[sp["id"]]),
        }
        for sp in species
    }

    result = {
        "selection": {
            "n_species": len(species),
            "min_family_size": args.min_family_size,
            "min_species": args.min_species,
            "species": [{"id": sp["id"], "family": sp["family"],
                        "image_dir_name": sp.get("image_dir_name")} for sp in species],
        },
        "per_species": per_species,
        "n_decidable": n_decidable,
        "n_cases": len(cases),
        "n_traces": len(trace_paths),
        "confusion": conf,
        "gate": gate,
        "baseline": baseline,
        "provenance": {"vlm_id": vlm_id},
        "decision": gate["decision"],
    }

    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[inat_pilot] decision={gate['decision']} "
          f"sibling_recall={gate['sibling_recall']!r} true_fp={gate['true_fp']!r} "
          f"n_decidable={n_decidable}")

    return result


if __name__ == "__main__":
    main()
