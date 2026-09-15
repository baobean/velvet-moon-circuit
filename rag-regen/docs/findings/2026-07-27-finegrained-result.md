# Fine-grained result — phrase-level contrast does not detect fine-grained identity error

**Date:** 2026-07-27
**Run:** `outputs/screen_20260726_233320/` (`c1.md`, `c1.json`)
**Spec:** `docs/superpowers/specs/2026-07-27-finegrained-verifier-design.md`
**Prior finding:** `docs/findings/2026-07-27-c1-result.md`
**Sample:** the same 22 drafts, ground truth `verdict_identity` = 15 fail / 7 pass (68% base rate)
**Stream A:** GroundingDINO-base + SigLIP-SO400M-384, re-run with `--force` · **Stream B:** cached, unchanged

---

## Headline

**RULE NOT MET.** No band. No δ anywhere in the pre-registered grid catches a single one of the
fine-grained failures.

```json
{
  "tau": 0.25,
  "baseline_caught": ["amur_leopard", "anas_platyrhynchos", "axolotl", "durian",
                      "love_in_a_mist", "monkey_puzzle_tree", "zalophus_californianus",
                      "golden_retriever", "red_ferrari"],
  "baseline_missed": ["african_grey_parrot", "boston_bull", "bonsai_tree",
                      "cardinal_bird", "hedgehog", "polar_bear"],
  "rule_met": false,
  "band": [],
  "chosen_delta": null,
  "caught_at_chosen": [],
  "false_positives_at_chosen": []
}
```

Across all 81 δ values from −0.20 to +0.20 the catch count is **0 out of 6**, at every single grid
point. The `verdict` (any-visible-defect) column is the same story: 0 out of 7.

Per spec §5, the conclusion is that **phrase-level contrast is insufficient**, not that δ needs more
tuning. This finding does not propose a wider grid, a different δ, or a different τ.

## 1. The τ = 0.25 baseline does not match the published τ\* = 0.02 split

It is one case smaller, and the difference is `golden_retriever`.

| | published (τ\* = 0.02) | this run (τ = 0.25) |
|---|---|---|
| missed | african_grey_parrot, boston_bull, bonsai_tree, cardinal_bird, hedgehog, **golden_retriever**, polar_bear | african_grey_parrot, boston_bull, bonsai_tree, cardinal_bird, hedgehog, polar_bear |
| count | 7 | **6** |

`golden_retriever` scores `sim = 0.081` against its own phrase. At τ = 0.02 that is `PRESENT` and the
case is missed; at the pinned τ = 0.25 it falls below threshold, becomes `MISSING`, and the case is
caught. It moves for a threshold reason, not a contrastive one — the grounded arm at τ = 0.25 is
saying "there is no golden retriever here at all," which is not the fine-grained judgement this plan
set out to add. The rule is graded against the τ = 0.25 split of **6**, as pre-registered.

Nothing about the models or the drafts moved. I diffed a copy of the pre-`--force` `stream_a.json`
and found 0 of 22 concept scores changed in `sim`, `state` or `box` — but that copy was scratch and
is not preserved, and `--force` overwrote the original, so **no artifact in the repo or the run
directory can now reproduce that check**. What remains independently checkable is every consequence
of it: the fused arm still reports TP 8 / recall 0.533 / bal-acc 0.767 at τ\* and the abstain rate is
still 0%, both matching the published C1 values exactly. Treat the model's stability as
well-corroborated rather than directly verified. What did change is τ, the new `sim_coarse` column,
and — unintentionally — the δ = 0 default discussed in the appendix.

## 2. Nothing was caught, and the reason is structural, not marginal

`FINE_MISMATCH` never changes a verdict anywhere inside the grid, and the mechanism is easy to state:

**Every case whose margin falls below +0.20 already has `sim < τ = 0.25` and is already `MISSING`.**
`MISSING` takes precedence over `FINE_MISMATCH`, so the set δ could newly flag is a strict subset of
the set the threshold already flags. The lowest `sim` above τ is `bonsai_tree` at 0.312, and its
margin is 0.291 — outside the grid. This is why the fused sweep is flat: the confusion matrix is
**exactly identical at all 81 δ** (TP 9, FP 2, TN 5, FN 6 — precision 0.818, recall 0.600, bal-acc
0.657, MCC +0.293), not merely flat to the printed precision.

The margins, sorted, with ground truth:

| margin | case | identity | sim | sim_coarse | coarse term |
|---|---|---|---|---|---|
| −0.403 | monkey_puzzle_tree | fail | 0.003 | 0.406 | tree |
| −0.231 | anas_platyrhynchos | fail | 0.213 | 0.445 | duck |
| −0.082 | love_in_a_mist | fail | 0.000 | 0.082 | flower |
| −0.049 | axolotl | fail | 0.019 | 0.068 | salamander |
| −0.017 | durian | fail | 0.000 | 0.017 | fruit |
| −0.000 | zalophus_californianus | fail | 0.000 | 0.001 | sea lion |
| +0.030 | golden_retriever | fail | 0.081 | 0.051 | dog |
| +0.037 | **sushi** | **pass** | 0.038 | 0.000 | food |
| +0.040 | **stack_of_books** | **pass** | 0.066 | 0.026 | books |
| +0.041 | red_ferrari | fail | 0.076 | 0.035 | sports car |
| +0.291 | bonsai_tree | fail *(missed)* | 0.312 | 0.021 | tree |
| +0.324 | **flamingo** | **pass** | 0.329 | 0.004 | bird |
| +0.338 | amur_leopard | fail | 0.434 | 0.096 | leopard |
| +0.383 | **cactus** | **pass** | 0.386 | 0.002 | plant |
| +0.443 | hedgehog | fail *(missed)* | 0.588 | 0.145 | small mammal |
| +0.503 | african_grey_parrot | fail *(missed)* | 0.707 | 0.204 | parrot |
| +0.515 | **fox** | **pass** | 0.516 | 0.001 | canine |
| +0.560 | **violin** | **pass** | 0.661 | 0.101 | string instrument |
| +0.577 | polar_bear | fail *(missed)* | 0.609 | 0.032 | bear |
| +0.637 | **panda** | **pass** | 0.639 | 0.002 | bear |
| +0.763 | boston_bull | fail *(missed)* | 0.780 | 0.017 | dog |
| +0.798 | cardinal_bird | fail *(missed)* | 0.827 | 0.029 | bird |

**The six target cases and the correct images are thoroughly interleaved.** Ranked by descending
margin, the top twelve rows run: `cardinal_bird` (missed), `boston_bull` (missed), **`panda` (pass)**,
`polar_bear` (missed), **`violin` (pass)**, **`fox` (pass)**, `african_grey_parrot` (missed),
`hedgehog` (missed), **`cactus` (pass)**, `amur_leopard` (already caught), **`flamingo` (pass)**,
`bonsai_tree` (missed). Three of the **six** largest margins belong to correct images, and three of
the six missed cases (`african_grey_parrot`, `hedgehog`, `bonsai_tree`) sit *below* all three of
`panda`, `violin` and `fox`. The margin separates fine-grained failures from correct images at
roughly chance.

**This is not a grid-width problem, and I checked rather than assumed.** Because `FINE_MISMATCH`
fires on *low* margins, catching a missed case at some δ also flags every case below it. So the
interleaving is directly a price list. Widening δ diagnostically to the full legal range [−1, +1]:

| δ above | missed caught | new false positives |
|---|---|---|
| 0.291 (`bonsai_tree`) | 1 of 6 | 0 |
| 0.443 (`hedgehog`) | 2 of 6 | 2 — `flamingo`, `cactus` |
| 0.503 (`african_grey_parrot`) | 3 of 6 | 2 |
| 0.577 (`polar_bear`) | **4 of 6** | **4** — adds `fox`, `violin` |
| 0.763 (`boston_bull`) | 5 of 6 | 5 — adds `panda` |
| 0.798 (`cardinal_bird`) | 6 of 6 | 5 (every remaining negative) |

The pre-registered rule needs 4 catches **and** zero false positives. The first δ reaching 4 catches
already costs 4 of the 5 available new negatives, and the widest δ range that catches anything at all
without a new false positive — δ ∈ (0.291, 0.324] — catches exactly 1. There is no δ in the entire
admissible range that satisfies the rule; at the
δ = 0.80 that catches all six, **all seven** true negatives are false positives. This table exists to
rule out "the grid was too narrow," not to propose a value — every row of it fails the rule.

## 3. Why the margin carries so little signal

Two asymmetries push the margin positive regardless of whether the identity is right. Neither is
about the image.

**(a) The score is absolute, not relative.** `CropScorer.score` returns `sigmoid(logit)` for a
**single (crop, text) pair**. So `sim − sim_coarse` differences two independent absolute scores, and
SigLIP's absolute scale is far higher for specific, caption-like phrases than for generic
superordinates on the same crop:

- a Boston-bull-ish dog crop: **"Boston bull" 0.780** vs **"dog" 0.017**
- a cardinal-ish bird crop: **"cardinal bird" 0.827** vs **"bird" 0.029**
- a correct fox crop: **"fox" 0.516** vs **"canine" 0.0005**

Nothing there is a statement about whether the animal really is a Boston bull. The margin largely
measures **how alt-text-like the fine phrase is** relative to its superordinate — a constant of the
phrase pair, which swamps the image evidence.

**(b) The two scores are not measured the same way.** `best_box` is the argmax of `sim` over up to 8
candidate boxes scored against the **fine** phrase (`grounded.py:87-92`); `sim_coarse` is then a
single score at that already-fine-favoured crop (`grounded.py:94-98`). `sim` is a maximum over candidates and `sim_coarse`
is not, which **biases the margin positive by construction**, on top of (a). I cannot size this
effect from the artifacts — `stream_a.json` persists only the winning box's score, not the eight
candidate scores — so I am not claiming it is small, only that it is present and unquantified.

The contrast fires only where the draft is grossly wrong and the coarse term wins outright. All six
cases with a negative margin (`monkey_puzzle_tree`, `anas_platyrhynchos`, `love_in_a_mist`,
`axolotl`, `durian`, `zalophus_californianus`) are already in `baseline_caught` — the gross-category
regime the verifier handled before this plan.

## 4. False positives: `stack_of_books` and `sushi`, and they are not δ's fault

Two false positives at **every** δ in the grid, both precision controls, both `verdict_identity =
pass`:

- **`stack_of_books`** — *"is a stack of books; book design not correct"* — `sim = 0.066`
- **`sushi`** — *"IS sushi; meat looks weird, needs enhancement"* — `sim = 0.038`

Both fall below the pinned τ = 0.25 and are flagged `MISSING` by the **threshold**, not by the
contrastive rule. They are already false positives of the δ-disabled baseline. `cactus`, the third
precision control, is clean at `sim = 0.386`.

**This means the rule's zero-false-positive criterion was unsatisfiable on the identity column at
τ = 0.25 before δ was ever swept.** That is a flaw in the pre-registration worth recording — pinning
τ at 0.25 to avoid the τ\* overfitting mistake imported two baseline false positives with it. It
neither rescues nor worsens the verdict: the catch criterion fails independently at 0 of 6 across the
whole grid, so the rule fails on the substance, not on a technicality. But had the mechanism worked,
this would have blocked a pass for the wrong reason, and it should be fixed before any future
pre-registration reuses this rule.

## 5. `hedgehog` was not caught — the planted control behaved as predicted

Spec §5 predicted it would not be: `hedgehog` is a basic-level concept whose failure is a rendering
defect (*"still recognisably a hedgehog; spikes too sparse, legs too long"*) with no categorical
contrast available. Its coarse term had to be authored as `small mammal`, and the margin is +0.443 —
well outside the pre-registered grid. It is not caught at any δ in that grid; in the widened
diagnostic it first appears at δ > 0.443, which already costs two correct images (`flamingo`,
`cactus`). The control did its job.

## What this means for the repair loop

1. **Do not proceed to the repair loop on this verifier.** The C1 finding's recommendation stands
   unchanged: the verifier passes most of the cases the repair step exists to fix, and the contrastive
   check does not move that.
2. **Move to image-prototype scoring as a separate design.** The failure here is specific and
   informative: text-side contrast fails because SigLIP's absolute score for a phrase is dominated by
   the phrase's own caption-likeness. Comparing the crop against **retrieved reference images** of the
   fine concept removes the text side of that asymmetry entirely, and the reference images are already
   in the pipeline. That is a different mechanism, not a re-tuning of this one. **But it removes only
   asymmetry (a).** Box selection would still be the argmax against the fine phrase, so asymmetry (b)
   survives the swap unchanged; a prototype design should score the *same* candidate boxes both ways,
   or select boxes by a phrase-neutral criterion, rather than inheriting this selection step.
3. **The δ machinery can stay.** `FINE_MISMATCH`, the state precedence, and the offline recovery path
   are correct and tested; it is the *signal* fed to them that carries no information. A prototype
   score can be dropped into the same slot.

## Reproducing

```bash
./scripts/run.sh score-a --labels outputs/screen_20260726_233320/labels.csv --force
./scripts/run.sh c1      --run    outputs/screen_20260726_233320
```

Stream A ~4 min on a free card; the report is seconds. Stream B was **not** re-run — this plan does
not change it, and its cached `stream_b.json` is reused as-is. `outputs/` is gitignored, so `c1.md`
and `c1.json` live with the run rather than in git.

## Caveats

- n = 22, 7 negatives on the identity column, 6 fine-grained failures in the graded set. Every
  interval is wide and no single case should be read as evidence on its own.
- τ was pinned at 0.25 and not fitted, but **δ was still selected in-sample**. The point is moot here
  — no δ passed — but a future MET result on this design would carry that weakness.
- **No held-out split exists at this size.** Both the baseline split and the rule are computed on the
  same 22 cases.
- The `coarse` terms were authored blind from the concept name, before any of these numbers existed
  (spec §3). A different author would pick different superordinates; `small mammal` for `hedgehog` and
  `canine` for `fox` are debatable. The failure is systematic across all 22 pairs, so it is unlikely to
  turn on wording, but this was not measured.
- Stream B's cached run is nf4-quantised, inherited from the C1 run along with that finding's caveat.
- The `verdict` (any-visible-defect) column is reported alongside in `c1.md` and reaches the same
  verdict: RULE NOT MET, 0 of 7 caught at every δ.

## Appendix: the τ-sweep tables briefly stopped reporting the pre-contrastive verifier — fixed

When this finding was first written, regenerating `c1.md` had moved a published C1 number with no
model change behind it.

| grounded @ τ\* | published C1 | the run as first regenerated | after the fix |
|---|---|---|---|
| τ\* | 0.02 | 0.01 | **0.02** |
| TP / recall | 5 / 0.333 | 6 / 0.400 | **5 / 0.333** |
| bal-acc / MCC | 0.667 / +0.370 | 0.700 / +0.418 | **0.667 / +0.370** |

Cause: `grounded_fails()` defaulted `delta=0.0`, and δ = 0.0 is **not** the mechanism turned off — it
fires `FINE_MISMATCH` whenever the coarse term merely ties or beats the fine one. `sweep_tau`
inherited that default, so the τ tables graded a contrastive verifier. The extra catch was
`anas_platyrhynchos` (`sim` 0.213 ≥ τ, margin −0.231). `c1.state_at_tau` existed and was documented
as *"pre-contrastive state recovery, kept for the tau report"* — but nothing outside the tests called
it.

**Fixed in `f472538`.** `c1.DELTA_OFF = -1.0` is now the value that disables the test — similarities
are in [0, 1], so the margin is ≥ −1 and the strict `<` can never fire — and it is the default for
`grounded_fails`, `fused_fails` and `abstain_rate`, and is passed explicitly by `baseline_split`,
whose docstring already claimed δ was disabled. `state_at_tau` was deleted as redundant with
`state_at(sim, None, tau, DELTA_OFF)`. A guard test now pins that `sweep_tau` does not apply the
contrastive rule; the absence of one is what let this through.

**The verdict above never depended on this, and the fix confirms it.** I had verified that at
τ = 0.25 the δ = 0.0 baseline split was identical to a genuinely disabled δ = −1.0, for both verdict
columns, because no case with `sim ≥ 0.25` has a negative margin. Re-running `c1` after the fix
reproduces `verdict_identity` and `verdict` `finegrained` artifacts **bit for bit** — `rule_met:
false`, `band: []`, `chosen_delta: null`, and the same 9-caught / 6-missed membership. The τ tables in
`c1.md` are once again comparable to the published C1 finding, whose grounded row now reproduces
exactly.
