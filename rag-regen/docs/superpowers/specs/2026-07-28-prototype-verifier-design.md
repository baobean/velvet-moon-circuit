# Prototype verification — image-side contrast for fine-grained identity

**Date:** 2026-07-28
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md`
**Supersedes the mechanism in:** `docs/superpowers/specs/2026-07-27-finegrained-verifier-design.md`
**Prior findings:** `docs/findings/2026-07-27-c1-result.md`, `docs/findings/2026-07-27-finegrained-result.md`

---

## 1. The claim

Stream A can detect fine-grained identity error by comparing the detected crop against **reference
images** of the concept, rather than against the concept's **name**.

This is a different mechanism, not a re-tuning of the previous one. It is pre-registered the same
way, and it can fail the same way.

## 2. What failed, and why it constrains this design

Phrase-level contrast caught **0 of 6** previously-missed cases at all 81 δ values. The finding
identified two asymmetries, and this design is shaped by both.

**(a) The score was absolute, and dominated by the phrase.** `CropScorer.score` returns a sigmoid
over a single (crop, text) pair. SigLIP scores specific caption-like phrases far higher than generic
superordinates on the *same* crop — `"Boston bull"` 0.780 vs `"dog"` 0.017. The margin largely
measured how alt-text-like the fine phrase was, a constant of the phrase pair, not a statement about
the image.

**(b) The two sides were not measured the same way.** `best_box` is the argmax over candidates scored
against the **fine** phrase (`grounded.py:87-92`); `sim_coarse` is then a single score at that
already-fine-favoured crop (`grounded.py:94-98`). One side is a maximum over candidates, the other is
not. This biases the margin positive by construction.

The finding warned that prototype scoring removes (a) but **not** (b): box selection would still be
argmax against the fine phrase. This design must address (b) explicitly or it inherits the flaw.

## 3. Mechanism

Two prototypes per concept, both built from images, both scored the same way:

```
proto_fine   = normalise(mean(normalise(embed(refs("African grey parrot")))))
proto_coarse = normalise(mean(normalise(embed(refs("parrot")))))

sim_proto        = cos(embed(crop), proto_fine)
sim_proto_coarse = cos(embed(crop), proto_coarse)
margin           = sim_proto - sim_proto_coarse
```

`FINE_MISMATCH` fires when `round(margin, 9) < delta_proto`.

This is the structure that measured **0.917** box-selection accuracy in
`../ImageRAG/scripts/finegrained_seg.py` (`PrototypeClassifier`, L70-105) — nearest-prototype over
mean class embeddings. That result is *discriminative*: it compares a crop against competing class
prototypes. Plan 3's test was *absolute*. Preserving the discriminative structure is the point.

Asymmetry (a) is removed entirely: no text embedding participates, so phrase caption-likeness cannot
enter the margin. Asymmetry (b) is addressed in §5.

## 4. Encoder choice

**SigLIP image tower** (`ViT-SO400M-14-SigLIP-384`), already loaded for retrieval and crop scoring.

**Not DINOv3.** DINOv3 is the project's headline identity metric (`RUNBOOK.md` §5.2). Using it inside
the verifier would make the pipeline optimise against its own metric — the same circularity that
rules `gt_refs` out as a shipping prototype source. DINOv3 stays purely evaluative.

**Caveat, stated up front.** When prototypes come from retrieval, the retriever and the verifier
share an embedding space: the verifier partly scores *"does this crop resemble what SigLIP already
judged similar to the query."* That is mild self-confirmation and it is inherent to reusing one
encoder. The `gt_refs` ceiling arm (§6) does not have this property, which is a second reason it
earns its place.

## 5. Persisting every candidate, and why the live rule does not change

The obvious fix for asymmetry (b) — select the box by detector confidence instead of by fine-phrase
similarity — silently moves the τ = 0.25 baseline, because `sim` at the selected box is what decides
`MISSING`. That would destroy comparability with the pre-registered baseline this rule is graded
against.

So the live selection rule and the `state` logic in `grounded.py` are **unchanged**. Instead every
candidate box is scored against everything and persisted:

```python
@dataclass(frozen=True)
class CandidateScore:
    box: Box
    dino_conf: float
    sim: float                        # crop vs fine phrase   (text)
    sim_coarse: float | None          # crop vs coarse phrase (text)
    sim_proto: float | None           # crop vs fine prototype   (image)
    sim_proto_coarse: float | None    # crop vs coarse prototype (image)
```

`ConceptScore` gains `candidates: list[CandidateScore]` and the selected box's `sim_proto` /
`sim_proto_coarse`.

Two consequences:

1. The τ = 0.25 baseline reproduces bit for bit. The 9-caught / 6-missed split stays the graded
   target.
2. Box selection becomes an **offline** choice in `c1.py`, where alternative rules can be compared
   and reported. This closes a gap the previous finding named explicitly: *"`stream_a.json` persists
   only the winning box's score, not the eight candidate scores — so I am not claiming it is small,
   only that it is present and unquantified."*

The pre-registered grading uses the **detector-confidence** selection rule for the prototype margin —
phrase-neutral, and fixed before any number is looked at. Other selection rules may be reported as
diagnostics, never as the graded result.

Cost: 8 crops embedded once each, scored against 2 prototype vectors. One forward pass per crop on an
already-loaded encoder.

## 6. Two arms

| arm | prototype source | what it answers | reportable |
|---|---|---|---|
| **ceiling** | `gt_refs` from `dataset.yaml` | Does the mechanism work at all, given perfect references? | as a ceiling, never as a verifier result |
| **retrieved** | top-k from `retrieve.py` | Does it survive retrieval error? | yes — this is the number |

This mirrors the `oracle` / `full` split the project already uses (parent spec §6), and it separates
"the mechanism is dead" from "retrieval fed it bad references" — the exact ambiguity that made Plan 3
expensive to interpret.

**Both arms need coarse reference images, and neither source exists yet.** This is the single
scheduling fact that matters for this plan.

`gt_refs` supply the *fine* side only — they are photographs of the concept, and the dataset carries
nothing for `parrot` or `small mammal`. Deriving a coarse prototype from sibling cases sharing a
coarse term was considered and rejected on two grounds, both checked rather than assumed:

- Only **4 of the 6** graded cases have a sibling (`bonsai_tree`, `boston_bull`, `cardinal_bird`,
  `polar_bear`; `african_grey_parrot` and `hedgehog` have none). With a 4-of-6 pass threshold, the
  rule could then only pass if every available case caught — a degenerate grading setup.
- A `dog` prototype built from golden retriever photographs is another *fine* prototype, not a
  superordinate. It would bias the margin in an uncontrolled direction.

So each arm takes its coarse references from its own source:

| arm | fine prototype | coarse prototype | prerequisite |
|---|---|---|---|
| **ceiling** | `gt_refs` (on disk) | `coarse_refs` — **new, operator-supplied** | ~3 images per coarse term |
| **retrieved** | top-k for the concept | top-k for the coarse term | corpus + FAISS index |

`coarse_refs` is a new optional key in `dataset.yaml`, keyed by coarse term rather than by case so
the four shared terms are authored once. 18 distinct terms across the 22 cases.

**What this plan delivers now:** the full implementation, both arms wired, unit-tested against fakes,
and the baseline-invariance regression. **What it cannot deliver now:** either measured number.
Neither `coarse_refs` nor an index exists, and both are operator inputs. That is a clean handoff
boundary, not a shortfall — it is the same division of labour as the larger dataset.

**Leakage guards, both enforced in code:**

- A case's own draft or test image can never enter its own prototype — ported from
  `finegrained_seg.py:85`.
- The `retrieved` arm must not read `gt_refs`. The two sources are separate constructors and the
  arm is recorded in the artifact.

## 7. Pre-registration

Fixed before any prototype number is computed.

- **τ pinned at 0.25**, unchanged, not fitted. `MISSING` still decided by the text `sim`, preserving
  the baseline.
- **`delta_proto` swept** over the same 81-point grid, −0.20 to +0.20 in 0.005 steps.
- **Graded target:** the 6 cases missed by the τ = 0.25 baseline on `verdict_identity` —
  `african_grey_parrot`, `boston_bull`, `bonsai_tree`, `cardinal_bird`, `hedgehog`, `polar_bear`.

**RULE MET** requires all three, over a band of **3 or more consecutive** δ values:

1. **≥ 4 of the 6** previously-missed cases newly caught;
2. **no *new* false positives** relative to the δ-disabled baseline;
3. gross-category catches retained.

**Criterion 2 is a correction to the previous pre-registration.** Plan 3 required *zero* false
positives absolutely, but τ = 0.25 already produces two (`stack_of_books`, `sushi`, both flagged
`MISSING` by the threshold, not by the contrastive rule). That made the criterion unsatisfiable
before δ was ever swept. Grading against *new* false positives measures the mechanism instead of the
threshold it inherited.

`DELTA_OFF = -1.0` remains the value that disables the test. δ = 0.0 is an aggressive grid point, not
the mechanism off.

**If the rule is not met, the conclusion is that image-side contrast is insufficient** — not that
`delta_proto` needs a wider grid. The diagnostic widening table from the previous finding may be
reproduced to rule out grid width, but never to propose a value.

## 8. Out of scope

- Changing τ, the fusion rule, Stream B, or the published C1 numbers.
- Replacing the text-side `sim` / `sim_coarse` fields. They stay, so both mechanisms can be compared
  on one run.
- Retrieval quality work. The retrieved arm consumes `retrieve.py` as it stands.
- Any claim at n = 22. Both findings already state the sample cannot separate close alternatives.

## 9. Risks

**P1 — The graded target is 6 cases.** A 4-of-6 threshold on 6 items is coarse; one case moves the
verdict by 17 points. Unavoidable at this dataset size, and the reason the larger dataset matters
more than this mechanism does.

**P2 — Coarse prototype quality is unmeasured, and it is the load-bearing half.** `refs("parrot")`
from a corpus may return a mix of species, which is arguably what a superordinate prototype *should*
be — but nothing verifies it. A coarse prototype that collapses onto one species becomes a second
fine prototype and the margin stops meaning what §3 claims. The same risk applies to
operator-authored `coarse_refs`, where it is a curation instruction rather than a retrieval property:
**author coarse references as a spread across the category, never several photographs of one member.**
`validate` warns when a coarse term has fewer than 3 references; it cannot check the spread, so this
stays a documented operator responsibility and a caveat on any result.

**P3 — Self-confirmation between retriever and verifier** (§4). Mitigated only by the ceiling arm.

**P4 — n = 22, no held-out split, δ still selected in-sample.** Inherited from Plan 3 and unchanged.

## 10. Testing

Same standard as the existing suite: dependency injection throughout, fakes for every model, no
module-level loading. The 228 existing tests must stay green.

- `build_prototype` — normalise/mean/renormalise arithmetic against hand-computed vectors; unit-norm
  output; single-reference case; empty input raises.
- Leakage guards — a prototype built from a set containing the excluded image must equal one built
  without it.
- `CandidateScore` persistence — all candidates present in the artifact, not just the winner.
- Baseline invariance — **the τ = 0.25 split and every published C1 number reproduce exactly** with
  prototypes wired in. This is the regression that matters most.
- `delta_proto` sweep and the band rule — reuse the tested `evaluate_rule` / `robust_band` machinery.
- A guard test pinning that the graded selection rule is detector-confidence, so a later refactor
  cannot silently switch it.
- `coarse_refs` loading — keyed by coarse term, absent key tolerated, unreadable path named in the
  error, and `validate` warning when a term carries fewer than 3 references.
- Arm separation — a `retrieved`-arm run must not read `gt_refs`, asserted by injecting a source that
  fails if touched.

GPU paths are verified by one 1–2 case end-to-end smoke run before handoff, not by unit tests.
