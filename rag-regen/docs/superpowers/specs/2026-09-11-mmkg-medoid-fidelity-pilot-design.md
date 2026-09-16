# MMKG medoid-as-reference — matched fidelity pilot (design)

**Date:** 2026-09-11
**Status:** DESIGN ONLY. No code, no GPU yet. Execution is gated on this spec
being locked and an implementation plan approved. GPU (FLUX draft +
inpaint) is required to *run* it; it is deferred until then.
**Depends on:** the parked plumbing branch `mmkg-reference-integration`
(`Case.global_id`, `--reference-source mmkg`, `_reference_mmkg_one`) and the
MMKG species store (`ragregen/mmkg_store/`, built store at
`/mmlabworkspace_new/Students/tuanld/soict-2026-data/mmkg_store/`).

---

## 1. Research question (narrow, on purpose)

> Holding species, image source, draft, mask, prompt, seed, and generator
> constant, does a **SigLIP-centroid medoid** work better than a
> **deterministic non-medoid image of the same species** as the *repair
> reference* — measured on the **actual repair output**, not predicted?

This is a **retrieval-quality** question about **one mechanism** (medoid
*selection*). It is deliberately NOT "does MMKG work?" and NOT "does graph /
attribute reasoning improve generation?". Those were answered **negative**
already (see §2).

## 2. Relationship to prior evidence (why this is still open)

Three prior arms wound down the **relational/attribute MMKG** and concluded,
triple-confirmed, that *a relational/attribute MMKG gives no advantage over
retrieval; the leverage is retrieval/embedding quality*:
- `docs/superpowers/2026-09-09-mmkg-inat-birds-pilot-finding.md` (attribute
  verifier loses to embedding baseline on the ideal substrate);
- `docs/superpowers/2026-09-09-mmkg-stage1-finding.md` (attribute verifier on
  LAION; **and Role 1 "organized retrieval"**);
- GRAFT visual-repair / rare-attribute-transfer (referenced therein).

**The one gap this pilot fills:** Stage-1 Role 1 measured only *reference
divergence* — the medoid arm and the flat arm "never pick the same reference"
(0% identical) — and then **predicted** "parity-not-gain on fidelity … not
claimed as an improvement." It **never generated the two repairs and scored
them.** The build summary is explicit: "no repair-accuracy experiment was
run." So the medoid's effect on *actual repair fidelity* is untested.

Because the medoid is a `siglip-centroid-nearest` selection, this is a
retrieval-quality question — the dimension the program says *does* carry the
leverage — which is what makes it defensible to measure despite the negatives.
**Those same priors predict PARITY as the most likely outcome**; the pilot's
value is to convert Stage-1's *prediction* into a *measurement*.

## 3. Design — within-species matched pairs

**Substrate:** the 25 iNaturalist pilot bird species already in the store
(the locked pilot set; no auto-expand). TreeVill is **excluded** — it has no
held-out split (`source_split: ref_paths_only`), so a repair cannot be scored
without contamination risk.

**Independent statistical unit:** the **species** (n = 25). NOT the 75
(repair, GT-image) comparisons.

**One matched pair per species** (25 pairs total). For each species `s`:

```
prompt_s, seed_s                                       ]
draft_s  = FLUX text->image(prompt_s, seed_s)          }  identical across arms
mask_s   = existing masker(draft_s)                    ]
        |
        +-- arm A (medoid):   ref_A = store.get("inat:<s>").medoid.image_path
        +-- arm B (baseline): ref_B = deterministic non-medoid same-species
        |                             build image (rule frozen, §4)
        |
        v  same FLUX inpaint, same seed_s, same mask_s, same prompt_s;
           ONLY image_reference differs (ref_A vs ref_B)
   repair_A_s , repair_B_s
```

**Held constant across the two arms:** species, prompt, draft, mask, seed,
generator, mechanism, all inpaint knobs. **Varying:** the reference image
only, and within that, only its *selection* (both refs are same-species
build-pool images from the same source).

**Wiring / reuse:**
- Arm A reuses the committed `_reference_mmkg_one` (`--reference-source
  mmkg`, `global_id = "inat:<species_key>"`).
- Arm B writes the same `refs.json` contract with the frozen-rule baseline
  image path (a trivial writer; no FAISS, no NN).
- mask / inpaint / DINO / SigLIP scoring all already exist.
- **New:** a small experiment harness that, per species, builds the shared
  draft+mask, runs both arms, scores, and emits per-species records; plus
  per-species prompt authoring and draft generation.

## 4. Frozen baseline-selection rule (no eyeballing)

Arm B's reference is chosen by a **deterministic rule fixed before any run**;
a human MUST NOT pick a "representative-looking" image by eye.

```
eligible = the species' BUILD-pool images (the 7 build images), as recorded
           in the store for "inat:<species_key>"
exclude  = the medoid image_path
order    = sort remaining eligible image_paths lexicographically (ascending)
choose   = the first element
```

If the medoid is somehow the only build image (should not occur; birds have
7), the case is dropped and reported, never silently substituted.

**Interface dependency (verify in the plan):** this rule needs the per-species
build-image paths. If the store record persists them (e.g. `candidates`), read
them there; otherwise resolve them deterministically from the build manifest
recorded in the record's `provenance`. The *rule* is frozen regardless of
where the paths are read from.

## 5. Ground truth & contamination guard

- **GT for scoring** = the species' **3 held-out eval images**
  (`source_split: build_7_eval_3`, contamination-checked at build time).
- **Disjointness (by construction):** `ref_A` (medoid) and `ref_B` (baseline)
  are both in the **build** pool; GT is the **eval** pool; build ∩ eval = ∅.
  So neither arm's reference can appear in its own scoring set. The plan MUST
  assert this disjointness at run time (SHA-256 or path check) and abort on
  any overlap rather than score through it.

## 6. Scoring & aggregation

**Per (repair, GT-image):** cosine similarity in the encoder space.

**Primary metric — DINO** (repo's fidelity metric of record; non-circular
here because DINO ≠ the SigLIP used to select the medoid).

**Secondary metric — SigLIP**, reported as **descriptive and explicitly
selection-favorable**: the medoid is constructed by SigLIP-centroid-nearest,
so SigLIP may reward it by construction. It never enters the decision; it is
used only to see whether the two encoders agree. Divergence between DINO and
SigLIP is itself a reportable finding.

**Aggregation (frozen):** aggregate the 3 GT images *within a species first*,
then form one paired difference per species:

```
score_A_s = mean( DINO(repair_A_s, GT_s_i) for i in 1..3 )
score_B_s = mean( DINO(repair_B_s, GT_s_i) for i in 1..3 )
delta_s   = score_A_s - score_B_s          # medoid - baseline
```

The 25 `delta_s` are the statistical sample. (SigLIP is aggregated the same
way, separately, for the descriptive read.)

## 7. Pre-registered decision rule

**Margin — cited, pilot-independent, frozen before any repair is generated
or scored:**

```
MARGIN_SOURCE = ragregen/metrics.py:195  (metrics.MARGIN)
              = the project's DINO non-inferiority margin, fixed pre-data by
                doc 5 §2; justified as ~3% of the operating-point cropped
                DINO (0.626, doc 4). "Do not tune it after seeing deltas."
MARGIN        = 0.02  (DINO cosine; magnitude of metrics.MARGIN = -0.02)
```

**Read-out** over the 25 `delta_s` (report mean `delta`, its paired 95% CI,
and the sign split — number of species with `delta_s > 0`):

- **GAIN** (→ justifies a full controlled experiment): mean `delta` ≥ +MARGIN
  **and** the 95% CI lower bound > 0.
- **LOSS** (useful negative): mean `delta` ≤ −MARGIN **and** the 95% CI upper
  bound < 0.
- **NO LARGE EFFECT → close the line** (the expected outcome): anything else —
  i.e. the CI does not clear a margin away from 0. With n = 25 the pilot is
  powered only for a large, practically-meaningful effect; a null here means
  "no large effect detected," which — given the parity prior — is read as
  *the medoid selection does not meaningfully change repair fidelity* and
  **closes the fidelity direction**. This is honestly a "no large effect,"
  not a proof of exact equivalence, and the spec says so.

## 8. Power & the no-rescue rule

n = 25 species × 1 draft = **25 matched pairs**, powered for a **large**
effect only. This is intentional (the prior is parity/small-effect; YAGNI on
power).

**No-rescue rule (frozen):** the pilot MUST NOT add seeds or drafts because
its result "looks flat / ugly." If there is no large effect → **close**. If a
large effect appears → design a *separate* follow-up experiment (more
seeds/drafts, proper power) as new work. Adding data after seeing the deltas
would invalidate the pre-registration.

## 9. Boundaries (explicit)

- ❌ attributes, ❌ relational/graph signal, ❌ reranking, ❌ multi-reference,
  ❌ VLM identity judge. The **medoid path only**.
- ❌ TreeVill (no clean held-out split).
- Generator internals untouched (`ragregen/regen.py` read-only), consistent
  with the plumbing branch.
- No fallback: a missing `global_id` or a build/eval overlap **aborts**; it is
  never silently substituted.
- **No repair-time NN, either arm.** Both references are resolved to fixed
  paths *before* inpaint — arm A from the precomputed `medoid.image_path`
  (build-time SigLIP-centroid selection), arm B from the frozen §4 rule. No
  FAISS / `nearest_crops` / similarity search runs at repair time. If it did,
  the experiment would no longer test the stated hypothesis (precomputed
  selection), so this is an invariant the harness MUST enforce, not a
  preference.

## 12. Claim boundary (how results may be worded)

This pilot tests **one mechanism** — precomputed SigLIP-centroid *medoid
selection* as the repair reference — on a bird-only substrate. The wording of
any conclusion is frozen accordingly:

- A **GAIN** may be reported only as: *"precomputed SigLIP-centroid species
  medoids improve repair fidelity over a deterministic same-species non-medoid
  reference, under this controlled setup (25 iNat birds, DINO)."* It may **NOT**
  be generalized to *"MMKG improves generation"* — attributes, part crops,
  taxonomy/hubs, graph relations, FAISS, reranking, and VLM signals are all
  **untested here** (§9). It also may not be generalized beyond birds to
  TreeVill or other domains.
- The result is about a **precomputed, build-time** selection, not any
  repair-time retrieval (§11 invariant) — so it speaks to reference *curation*,
  not to a retrieval algorithm.
- A **null / LOSS** is worded as in §7 and closes the fidelity direction of
  the medoid mechanism; it likewise says nothing about the untested MMKG
  features, which the prior evidence (§2) has already judged separately.

## 10. Deliverables

1. A frozen run spec + implementation plan (before GPU).
2. On execution: per-species records (draft, mask, both refs, both repairs,
   DINO & SigLIP scores, `delta_s`), plus a run-level report with the mean
   `delta`, 95% CI, sign split, and the §7 verdict.
3. A finding doc recording the measured outcome — including, in the likely
   parity case, the explicit statement that Stage-1's *predicted* parity is
   now a *measured* "no large effect," closing the fidelity direction.

## 11. Open items / risks (for the plan to resolve, not re-decide)

- **Where build-image paths live** in the store record (§4 interface
  dependency) — verify `candidates` vs manifest resolution.
- **Prompt template**: one frozen, deterministic per-species template built
  from the stored scientific/common name (a *generation* prompt, not a
  verifier input); frozen before any draft is generated.
- **Draft realism**: FLUX may render some rare birds poorly; a draft that the
  masker cannot ground is dropped and **reported** (never hand-fixed), and the
  drop count is part of the result.
- **n after drops**: if drops shrink n materially, report it; do not backfill
  to restore 25 (that is a rescue — forbidden by §8).
