# Fine-grained verification — contrastive crop scoring in Stream A

**Date:** 2026-07-27
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md`
**Motivating finding:** `docs/findings/2026-07-27-c1-result.md`
**Status:** design approved, plan pending

---

## 1. The problem this fixes

C1 measured the dual-stream verifier against 22 hand-labelled drafts and found a clean split in
what it catches:

- **Caught, 8/15** — the whole object is wrong: `amur_leopard`, `anas_platyrhynchos`, `axolotl`,
  `durian`, `love_in_a_mist`, `monkey_puzzle_tree`, `zalophus_californianus`, `red_ferrari`.
- **Missed, 7/15** — a recognisable member of the right category, wrong in its details:
  `african_grey_parrot`, `boston_bull`, `bonsai_tree`, `cardinal_bird`, `hedgehog`,
  `golden_retriever`, `polar_bear`.

The split is the wrong way round for this project. Reference-guided regeneration exists to repair
fine-grained identity — an Amur leopard that is merely a leopard, a Boston bull that is merely a
dog. The verifier passes exactly those cases and fires only on gross category error, which is the
easy case a generic pipeline already handles.

**The cause is that Stream A scores a phrase in isolation.** `GroundedVerifier` crops the detected
region and asks the scorer for `sim(crop, "African grey parrot")`, then compares it to an absolute
threshold τ. A passable parrot scores well on that phrase whether or not it has the white eye patch.
There is no point in the computation at which anything asks *which parrot*.

Stream B shares the blind spot for the same reason at a different granularity: it is asked a yes/no
question about the whole image. Because both streams miss the same cases, no reweighting of fusion
can recover them — which is why this design changes Stream A's computation rather than its
combination rule.

---

## 2. The fix

Score the crop **twice** — against the fine phrase and against its superordinate category — and
threshold the margin between them.

```
crop ──┬─ score(crop, "African grey parrot") ─→ sim_fine
       └─ score(crop, "parrot")              ─→ sim_coarse

no box                             → ABSTAIN         (unchanged)
sim_fine < τ                       → MISSING         (nothing of the sort is here)
sim_fine − sim_coarse < δ          → FINE_MISMATCH   (right family, wrong member)
otherwise                          → PRESENT
```

`FINE_MISMATCH` is a new state alongside the existing three. `fuse()` treats it as failure, so the
loop's `ok` semantics are unchanged and target selection still works — `fuse()` picks the
lowest-scoring failing phrase, and each case carries exactly one concept.

### Why a distinct state rather than reporting MISSING

Collapsing both into `MISSING` would produce identical verdicts and identical downstream behaviour,
so the argument is entirely about diagnosis. The decision rule in §5 is stated over the 7
fine-grained cases specifically. With one state, determining which mechanism fired on which case
requires re-running with δ disabled and diffing. With two, `trace.py` and the C1 report answer it
directly. The cost is a state constant, a fusion branch, and the tests that pin them.

### Where the mechanism has purchase, and where it does not

The margin test discriminates only when the concept is **subordinate to a visually similar
superordinate**. For basic-level concepts (`axolotl`, `durian`, `hedgehog`, `cactus`, `fox`,
`panda`, `sushi`, `violin`) there is no tighter category to contrast against, the margin carries
little signal, and τ remains the operative mechanism. This is a structural property of the approach
and is not a defect to be tuned away.

Applied to the 7 missed cases, 6 are subordinate terms with a natural superordinate
(`african_grey_parrot`→parrot, `boston_bull`→dog, `bonsai_tree`→tree, `cardinal_bird`→bird,
`golden_retriever`→dog, `polar_bear`→bear) and 1 is basic-level (`hedgehog`, which failed on
"spikes too sparse" — a rendering-quality defect with no categorical contrast available).
**The mechanism can reach at most 6 of the 7**, which is what makes the ≥4/7 bar in §5 a real test
rather than a formality.

---

## 3. Authoring the coarse term

`coarse:` is a new required field on each case in `dataset.yaml`, sitting next to `concept:`.
Per parent spec §7 the operator edits configs and not Python, so this belongs in the config surface.

**Integrity rule — the coarse term is authored from the concept name alone, before looking at the
draft.** It is the concept's superordinate category, not the thing the draft happens to resemble.
Choosing "dog" for `polar_bear` because the label notes say "face looks like a dog" would fit the
negative to the observed failure, which is answer-key leakage of exactly the kind parent spec §5
forbids elsewhere. The correct coarse term for `polar_bear` is `bear`.

`validate.py` enforces presence and rejects `coarse == concept` (case-insensitive, whitespace
normalised), which is the one mechanical error that silently disables the test — the margin would be
identically zero for every case.

Coarse terms for the current 22-case set:

| case | concept | coarse | subordinate? |
|---|---|---|---|
| african_grey_parrot | African grey parrot | parrot | yes |
| amur_leopard | Amur leopard | leopard | yes |
| anas_platyrhynchos | Anas platyrhynchos | duck | yes |
| axolotl | axolotl | salamander | basic-level |
| boston_bull | Boston bull | dog | yes |
| durian | durian | fruit | basic-level |
| love_in_a_mist | love-in-a-mist | flower | yes |
| monkey_puzzle_tree | monkey puzzle tree | tree | yes |
| zalophus_californianus | Zalophus californianus | sea lion | yes |
| bonsai_tree | bonsai tree | tree | yes |
| cardinal_bird | cardinal bird | bird | yes |
| hedgehog | hedgehog | small mammal | basic-level |
| cactus | cactus | plant | basic-level |
| flamingo | flamingo | bird | yes |
| fox | fox | canine | basic-level |
| golden_retriever | golden retriever | dog | yes |
| panda | panda | bear | yes |
| polar_bear | polar bear | bear | yes |
| red_ferrari | red Ferrari | sports car | yes |
| stack_of_books | stack of books | books | basic-level |
| sushi | sushi | food | basic-level |
| violin | violin | string instrument | yes |

---

## 4. Cost, and why it is four minutes

`score_a.py` persists **raw similarities and never thresholded verdicts** — states are recovered
offline by `c1.state_at_tau`. Extending it to persist `sim_coarse` alongside `sim` preserves that
property, so:

- Stream A runs **once**, about 4 minutes, and every (τ, δ) combination is recovered afterwards by
  arithmetic.
- Stream B is **not re-run**. The cached `outputs/screen_20260726_233320/stream_b.json` is reused
  verbatim — its computation is unchanged by this design. This saves the 13-minute nf4 Qwen pass.
- No drafting, no generation, no new model weights, no new disk. Per-crop cost is one extra text
  encoder forward pass, and coarse phrases repeat across cases so they cache.

Disk currently stands at 180 GB free (recovered from the 71 GB that triggered parent spec R2). This
design consumes none of it.

---

## 5. The decision rule, pre-registered

C1's τ\* was fitted to noise — it sat at the extreme edge of the swept grid on a peak two cells
wide. This design pre-registers its criteria to avoid repeating that.

**τ is pinned at the shipped default 0.25 and is not fitted.** δ is the only knob selected on this
data, so there is one fitted parameter rather than two.

**Baseline wrinkle, to settle first.** The 8-caught / 7-missed split in §1 is the published C1
baseline at τ\* = 0.02, not at τ = 0.25. Membership may differ at the pinned τ. The first step of
the re-run is therefore to recompute the baseline at τ = 0.25 with δ disabled and report both
splits. The rule below is evaluated against the **τ = 0.25 baseline**; the τ\* = 0.02 split remains
the published reference. If the two differ, say so plainly rather than quietly adopting whichever
is more favourable — and note that the §1 case names are then the reference split, not the graded
one.

**Primary criterion.** At a single δ, against `verdict_identity`:

1. **≥4 of the 7 fine-grained misses are caught.** The misses are those in the τ = 0.25 baseline,
   fixed and written down *before* any δ is applied, so the result cannot be reinterpreted
   afterwards. Expected to be the §1 set; if the baseline differs, the recomputed set is what
   counts and the bar stays at 4.
2. **Precision remains 1.000** — zero false positives across the 7 identity-pass cases
   (`cactus`, `flamingo`, `fox`, `panda`, `stack_of_books`, `sushi`, `violin`).
3. **The 8 already-caught gross-category cases stay caught.** δ can only add failures, so this
   should hold trivially; it is asserted rather than assumed because a τ interaction would be
   invisible otherwise.

**Robustness criterion.** Criteria 1–3 must hold across **at least 3 consecutive δ grid values**.
A result that survives at exactly one grid point is the τ\* mistake wearing a different letter.

Three of the identity-pass cases (`cactus`, `stack_of_books`, `sushi`) are the "right concept, wrong
aesthetics" controls the operator flagged. They are where precision will break first if δ is set
aggressively, which is what criterion 2 exists to catch.

**If the rule is not met**, the honest conclusion is that phrase-level contrast is insufficient and
the next candidate is image-prototype scoring — which cannot use `gt_refs` in the loop (§6) and so
becomes a different design, not a tuning pass.

### Sweep grid

`DELTA_GRID = tuple(round(0.005 * i, 3) for i in range(-40, 41))` — δ from −0.20 to +0.20 in steps
of 0.005. Negative δ is more conservative (fail only when the coarse term strictly beats the fine
term by a margin); δ = 0 fails when coarse ≥ fine; positive δ is more aggressive.

---

## 6. What this design deliberately excludes

**Image prototypes from `gt_refs`.** Mirroring `finegrained_seg.py`'s `PrototypeClassifier` would
likely score highest, but the loop verifies **before** retrieval (`v0 = verify(draft, prompt)`
precedes `retrieve`), so the only reference images available at v0 are the ground-truth refs. Those
are the answer key consumed by the DINO metric and the `oracle` arm. Putting them inside the in-loop
verifier is self-marking and would invalidate the main table. Viable only as an offline diagnostic.

**Closed-set argmax over all case classes.** `finegrained_seg.py` reaches 0.917 this way, but its
class set *is* the dataset. Accuracy inflates as the set shrinks, and the verifier would not
generalise to a case list it was not built for.

**Changes to Stream B.** Its blind spot is real but this design does not address it. Fixing one
stream at a time keeps the C1 re-run attributable to a single change.

**Enlarging the sample.** A bigger sample measures the same blind spot more precisely; it does not
close it. Deferred until after this result.

---

## 7. Testing

The prevailing failure mode on this project has been **tests that pass for the wrong reason** — four
occurrences on Plan 1 (Tasks 4, 5, the determinism test, and `FakePipe`). Two rules carry forward:

- Build fakes from the real signature via `inspect.signature`, never from the call site the plan
  happens to write.
- Mutation-test every assertion that claims to pin behaviour, and paste **real** pytest failure
  output as evidence. Hand-typed simulated output is not evidence.

Coverage required:

| area | test |
|---|---|
| `FINE_MISMATCH` boundary | margin exactly at δ; mutating `<` to `<=` must fail exactly one test |
| state precedence | `sim_fine < τ` **and** margin `< δ` reports `MISSING`, not `FINE_MISMATCH` |
| ABSTAIN unchanged | no box still abstains regardless of δ |
| fusion | `FINE_MISMATCH` fails the case; `box is None` is never the branch condition |
| config | missing `coarse` and `coarse == concept` both raise actionable `ValueError` |
| offline recovery | `state_at(sim, sim_coarse, τ, δ)` agrees with `GroundedVerifier` on the same inputs — the C1 report must not grade the verifier against a rule it does not use |
| δ sweep | a synthetic fixture with a known answer reproduces the expected catch counts |

The existing 127 CPU tests must stay green. GPU tests remain 6 passed / 2 skipped.

---

## 8. Deliverable

A re-run of the C1 report on the same 22 drafts, with the report split by mechanism (gross-category
catches vs fine-grained catches), the δ curve, and an explicit statement of whether the §5 rule was
met. That statement is what unblocks — or redirects — the repair loop.
