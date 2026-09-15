# MMKG attribute-verifier — iNaturalist birds pilot finding

**Date:** 2026-09-09
**Decision:** **WIND DOWN — and this time it is decisive.** On the ideal substrate, the MMKG
attribute-verifier fails its pre-registered gate, over-fires, and is beaten by a plain embedding baseline.
**Spec (frozen):** `docs/superpowers/specs/2026-09-09-mmkg-inat-birds-pilot-design.md`
**Run:** `outputs/mmkg_inat_pilot/result.json` (+ 75 per-test-image traces). Ground truth = iNaturalist's
authoritative species/family labels (non-circular: the VLM is the detector, the labels are truth).

## Why this test counts (unlike the LAION run)

The LAION/ImageNet result was corpus-confounded (noisy name-retrieval, arbitrary objects, ~60% abstain).
This pilot removed every one of those confounds: a fine-grained **species** domain with a real visual
ontology, an **authoritative taxonomy**, **dense clean images** (10/species), and genuine same-family hard
negatives (3 treecreepers, 3 albatrosses, 3 fairywrens, 3 *Phylloscopus* warblers, 3 fantails…). **All 25/25
species were decidable** — the abstain/scarcity wall is gone. So this is a fair test of the hypothesis, and
the negative is about the *idea*, not the data.

## Result — fails the pre-registered gate

| pre-registered bar | required | observed | met? |
|---|---|---|---|
| sibling-recall − true-FP | ≥ 0.20 | **+0.147** | ✗ |
| sibling-recall | ≥ 0.40 | 0.827 | ✓ |
| decidable species | ≥ 18/24 | **25/25** | ✓ |

Confusion over 237 (test-image, candidate-species) pairs (verdict FAIL = "predicted different species"):
- **sibling-recall = 0.827** (134/162 same-family siblings correctly rejected),
- **true-FP = 0.68** (51/75 of the bird's *own* species wrongly rejected — only **24/75** correctly PASS),
- 10 abstains (counted as PASS, conservative).

The verifier discriminates a *little* — siblings are rejected ~15 pp more often than the true species — but
the margin (**+0.147**) misses the 0.20 bar, and the **68% false-rejection rate** makes it unusable as a gate
(it would reject most correct repairs). One condition failing is enough; two of three checks pass, the
decisive margin does not.

## The baseline settles it

**SigLIP embedding baseline accuracy = 0.707** (n=75): nearest build-image centroid among the true species +
its in-set siblings picks the *true* species 71% of the time — with **no attribute machinery**. Plain visual
similarity out-discriminates the attribute verifier on the verifier's own task. This is the program's
through-line, now shown on the best-case substrate: **the discriminative signal is visual similarity, which
retrieval already captures; the attribute-MMKG adds nothing over it.**

## Root cause

Within-species VLM attribute-read variation ≈ between-sibling variation. A held-out bird photo (a different
individual, pose, light, plumage state) reads different fine attributes than the 7-image build consensus of
its own species, so the strict per-attribute contradiction check fires **even for the correct species** (hence
68% FP). Fine visual attributes as read name-blind by a VLM and matched by a bidirectional judge are simply
too noisy per-image to separate a species from its close relatives — while the holistic embedding is not.

## Consequence

- **Wind down the MMKG attribute-verifier line.** Unlike LAION, this is a fair, well-powered negative on the
  ideal substrate: fine-grained-attribute verification via VLM reads does not beat plain embedding similarity
  even where the domain, taxonomy, and image density are all favourable.
- **Program-wide conclusion, now triple-confirmed** (GRAFT visual repair; GRAFT rare-attribute transfer; this
  iNat attribute-verifier): a relational/attribute MMKG gives **no advantage over retrieval**. The leverage is
  **retrieval/embedding quality**, not graph or attribute structure. Every arm has now pointed the same way,
  including this best-case one.
- **No scale-up** to the full-val bird run (gated on this pilot, which failed).

## Limitations / honesty

- The verifier under test is one design (name-blind 6-slot read + bidirectional equivalence judge + FAIL-on-
  any-contradiction). A different attribute encoding (e.g. structured attribute *embeddings*, or a learned
  attribute classifier) is not tested here; but note that any such variant that reduces to visual similarity
  collapses into the embedding baseline, which already wins — so the burden is on a variant that is both
  non-visual and discriminative, exactly what GRAFT's OracleHub also failed to find.
- The judge and reads share one VLM; both arms (SAME and SIBLING) are judged identically, so this is symmetric
  noise, not a directional bias. Ground truth is the iNat label, so the evaluation is not VLM-circular.

## Reproducibility

`ragregen/mmkg/inat_pilot.py` (+ `tests/mmkg/test_inat_pilot.py`, 5 offline tests; reviewed spec-✅/Approved);
reuses `ragregen.mmkg.{schema,verifier,evaluate}`. Data: iNaturalist 2021 `val` birds, 25 species / 8 families,
deterministic §2 selection, under `/mmlabworkspace_new/.../inat2021_birds/`. Every knob frozen pre-run in the
spec. Run artifact `outputs/mmkg_inat_pilot/result.json`; traces under `outputs/mmkg_inat_pilot/trace/`.
