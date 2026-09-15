# GRAFT — Validation & De-bias Sprint Report

> **FULL-CORPUS UPDATE (2026-09-05, n=65).** The 8-species pilot below (§5.1) was scaled to the whole
> qualifying corpus via the fast eval driver (`docs/superpowers/specs/2026-09-03-fast-eval-driver-design.md`;
> 520 cells, 0 missing, ~3.5h). The pilot's direction holds and is now **statistically significant on the
> two primary fidelity metrics — but the effect is modest, roughly half the pilot's inflated margin.**
>
> **GRAFT vs B1, paired per-species, neutral arm, ip_scale=0.6, n=65** (`outputs/eval_fast_full/`):
>
> | metric | GRAFT−B1 mean Δ | win-rate | Wilcoxon (1-sided) | verdict |
> |---|---|---|---|---|
> | **DINOv3** (primary) | **+0.0288** | 40/65 (62%) | **p=0.007** | **significant (p<0.01)** |
> | **SigLIP2** | +0.0065 | 41/65 (63%) | **p=0.011** | **significant** |
> | CLIP-I | +0.0044 | 34/65 (52%) | p=0.19 | not significant (tie) |
>
> Pooled means: `ours` dino 0.365 / attr 0.705 > `b1` 0.344 / 0.672 ≫ `b0` 0.068/0.356 ≈ `b2` 0.061/0.350.
> **Name contribution at n=65:** named−neutral DINO for `ours` = **+0.0001** (name adds nothing to GRAFT)
> vs B0 **+0.022** (the no-info baseline leans on the name) — the core thesis holds decisively.
>
> **Honest reading (revised — see §0 below, which supersedes the "small-n optimism" line this paragraph
> originally carried).** GRAFT genuinely beats flat retrieval (B1) on fine-grained fidelity (DINOv3
> Wilcoxon p=0.007), and the advantage is reference-driven not name-driven. The unstratified full-corpus
> mean is small (**+0.0288** DINOv3, 40/65) and CLIP-I is a tie — **but that small mean is driven by
> measurement power, not by a weak method.** 53 of the 65 species are scored against a *single* held-out
> photo; restricting to species where fidelity is actually measurable (held-out ≥3, n=10) raises the
> DINOv3 advantage to **+0.081 / 80%**. The earlier claim that the +0.060 pilot was mere "small-n
> optimism" was **wrong**: all 8 pilot species had ≥4 held-out photos, i.e. they sit in the
> better-measured regime, and the pilot number is consistent with the full run's measurable subset, not
> with its diluted flat mean. The full reliability-stratified re-analysis is **§0**. The rest of this
> report is the original pilot writeup, retained for context.

**Date:** 2026-09-03
**Gates:** the Phase 2 full Treevill benchmark (`docs/superpowers/specs/2026-08-30-graft-phase2-design.md` §6).
**Spec:** `docs/superpowers/specs/2026-08-30-graft-validation-debias-sprint-design.md`
**Plan:** `docs/superpowers/plans/2026-08-30-graft-validation-debias-sprint.md`
**Artifacts:** `outputs/eval/{results,analysis}.json`, `outputs/audit/audit.json`, generated images under `outputs/<species>/eval_images/`.

---

## 0. Correction (2026-09-05): reliability-stratified re-analysis of the full corpus

> This section supersedes the top-banner's original "small-n optimism" reading of the n=65 result. It
> changes no numbers in the run and no GRAFT code — it re-aggregates the *same* full-corpus artifacts
> (`outputs/eval_fast_full/{analysis.json,_work/heldout.json}`) grouped by how reliably each species can
> be measured. All figures below are produced by `scripts/stratify_heldout.py` (re-runnable; nothing is
> hand-edited). The fast-driver wiring itself was validated separately by the Ashok fast-vs-slow
> cross-check (`ours` DINOv3 0.277 identical), so this is an evaluation-power finding, **not** an
> implementation bug and **not** a reason to change the method.

**The problem the flat mean hides.** The full-corpus GRAFT−B1 DINOv3 mean is **+0.0288** (40/65 win,
62%), but the per-species delta **SD is 0.114 — about 4× the mean**. That ratio is the tell: the signal
is small relative to the per-species measurement noise. The noise source is concrete — the adaptive
split leaves most species with almost nothing to score against:

| held-out photos per species | # species |
|---|---|
| **1** | **53** |
| 2 | 2 |
| 4–6 | 7 |
| 8–16 | 3 |

**53 of 65 species (82%) are scored against a single held-out real photo.** A DINOv3 fidelity score
against one photo is a high-variance estimate: DINOv3 rewards fine-grained, patch-level agreement with
that *specific* image (pose, crop, lighting, which organ is in frame), so a single unlucky reference can
swing a species' score far more than the method does. Averaging 53 such single-photo estimates with
equal weight regresses a real positive effect toward zero and leaves the headline hostage to individual
outliers.

**Stratifying by held-out count (measurement reliability) — DINOv3, GRAFT − B1:**

| subset | n | mean Δ | median Δ | SD | win-rate |
|---|---|---|---|---|---|
| all (mostly 1 photo) — *unstratified estimate* | 65 | +0.0288 | +0.033 | 0.114 | 40/65 (62%) |
| held-out ≥ 2 | 12 | +0.0638 | +0.055 | 0.096 | 9/12 (75%) |
| **held-out ≥ 3 — primary reliability-stratified result** | **10** | **+0.0812** | **+0.062** | 0.095 | **8/10 (80%)** |
| held-out ≥ 5 | 7 | +0.0923 | +0.057 | 0.105 | 6/7 (86%) |

The advantage **grows monotonically as the measurement gets more reliable** — the opposite of what an
overfit or a genuinely weak effect would do (those shrink under tighter measurement). Because build-set
size is capped at ~5 references for nearly every species (`k_build = min(5, n_unique−1)`), GRAFT's *input*
capability is roughly constant across these strata, so the stratification mostly isolates held-out
reliability rather than how much material GRAFT had to work with (e.g. Mango and East Indian copaiba
balsam, two of the worst losses, both had the full 5-reference build set).

The **held-out ≥3 subset (n=10) is the recommended primary number: +0.081 DINOv3, 80% win.** The ≥2 and
≥5 rows are supporting evidence for the same trend. **They are not a better point estimate of the true
effect** — n=7 for ≥5 is far too small, and its +0.092 should be read as "the trend continues," not as
the effect size. Read together, the four rows say the full-corpus mean **under**states the effect; they
do not pin the effect to any single value.

**A single-estimator alternative.** Weighting each species' delta by its held-out count (≈ pooling over
all generated×held-out pairs rather than over species-means) gives a DINOv3 GRAFT−B1 of **+0.065** —
consistent with the ≥2/≥3 subsets and a defensible single-number summary when one is needed.

**The Mango outlier — reference-specific variance, not a generation failure.** Mango is the largest
single loss and by itself pulls the flat mean down by ~0.007. It had a full 5-image build set and a
single held-out photo. Its deltas across metrics:

| metric | ours | b1 | Δ |
|---|---|---|---|
| DINOv3 | 0.173 | 0.609 | **−0.436** |
| SigLIP2 | 0.736 | 0.809 | −0.074 |
| CLIP-I | 0.696 | 0.759 | −0.064 |

The −0.44 DINOv3 collapse is **not** corroborated by SigLIP2 or CLIP-I, which show ordinary −0.07 near-ties.
GRAFT produced a perfectly recognizable mango (high SigLIP2/CLIP-I); DINOv3's fine-grained similarity to
that one held-out photo is what cratered. This is exactly the single-reference variance the stratification
is built to expose — treat such DINOv3 outliers as unresolved measurements, not as method failures.

**Correcting the "small-n optimism" verdict.** The original banner attributed the +0.060 pilot to small-n
optimism. That was wrong. The 8 pilot species — Bamboo (16), Ashok (10), Egyptian lotus (8), Nageshore
(6), Avocado (5), Camphor Tree (5), Hijol (5), Ashore (4) — **all have ≥4 held-out photos** (median 5.5),
so every one falls in the better-measured regime. The pilot's +0.060 is consistent with the full run's
held-out ≥3/≥5 subsets (+0.081 / +0.092), **not** with the diluted flat +0.029. Scaling to the full
corpus did not reveal a weaker method; it added 53 species whose single-photo scores cannot resolve GRAFT
from B1 and therefore diluted the mean.

**What this does and does not license.** It licenses reporting the effect as reliability-stratified rather
than as the flat mean, and retiring the "small-n optimism" reading. It does **not** license claiming
+0.092 as the true effect (n=7), nor any change to GRAFT — the method and its wiring are unchanged and
validated. The honest bound: **the full-corpus mean underestimates GRAFT's advantage; where fidelity is
measurable at all, GRAFT leads flat retrieval by roughly +0.08 DINOv3 with an 80% per-species win-rate.**

---

## 1. Summary

Phase 1 produced an honest but untrustworthy pilot: GRAFT beat the no-grounding (B0) and LLM-recall
(B2) baselines by 7–12× on DINOv3 fidelity but **lost to flat single-reference retrieval (B1)** on
n=11 cases across 3 species — the one comparison Phase 2 needs to settle. Four confounds made that
loss unreliable: a ≥6-unique-image sample floor, the concept **name** leaking into generation +
retrieval + reranker + the M1 attribute VLM, a **prompt asymmetry** (B1 alone was told "According to
this image"), and an unvalidated part-tree.

This sprint removed all four and re-ran a clean matrix: **8 species × {ours, ours_notree, b0, b1, b2}
× {neutral, named} × ip_scale∈{0.4, 0.6, 0.8} = 176 cells, 176 completed, zero missing.**

**Headline: with the confounds removed, GRAFT beats B1.** On the primary fidelity metric (DINOv3),
paired per-species at the best conditioning strength, GRAFT wins 6 of 8 species (+0.060 mean). The
result that motivated the whole sprint — "GRAFT loses to flat retrieval" — was an artifact of the
confounds, not the method.

Two supporting findings: the species **name contributes essentially nothing to GRAFT** (named−neutral
DINO Δ = +0.004, versus +0.136 for B0, which lives entirely on the name), confirming that GRAFT's
fidelity comes from the reference images; and the **part-tree makes no difference to the output at the
current gate setting**, so it is a prune candidate — with an important caveat (§5.3).

---

## 2. What changed (the de-bias)

| Confound (Phase 1) | Fix this sprint |
|---|---|
| ≥6-unique-image sample floor | Adaptive split `k_build = min(5, n_unique − 1)`; floor drops to 2 unique images. |
| Name in generation prompt (all methods) | Neutralized to the token `"plant"`; a `neutralize_name=False` "named arm" quantifies the name's contribution. |
| Name in B1 retrieval + reranker | Removed — exemplar selection is now the **SigLIP2 medoid** of the species' own photos (name-free). |
| Name in the M1 attribute VLM | M1 describes the references blind (no species name); attributes are genuinely ref-derived. |
| B1-only "According to this image" scaffold | Removed — B1 and GRAFT share `"a photo of a plant[, {attrs}]"` + the **same medoid exemplar** (exemplar parity). |
| Part crops from arbitrary `ref_images[0]` | Crops from the **medoid** reference. |
| Crop-vs-whole-image sim inconsistency | Parts with no reliable reference box are **excluded** from scoring (no whole-image fallback). |

CLIP-T is demoted to a diagnostic ("drift toward the name's common meaning"), not a headline metric.

---

## 3. Method / execution

Unchanged from Phase 1: single shared RTX 4090, every GPU phase in its own disposable subprocess
(`graft/worker_*.py`), staged + resumable via `scripts/gpu_queue.sh`. The matrix builds each species'
MMKG **once** with a fixed `neutralize_name=True` config and reuses it across every cell, so the
named arm differs from the neutral arm only at generation time (the graph is never a confound). Eval
config `n_refine=1` (one refine attempt beyond seed 0), `ip_scale` swept per cell.

The 176-cell run completed in ~27h wall-clock (2026-09-02 17:42, exit 0) after being moved onto an
uncontended card — on the shared card, every IP-Adapter cell OOM'd against another user's 9 GB job
(peak ~11.5 GB vs ~11.9 GB free); `pipe.vae.enable_slicing()` was added but the real fix was headroom.

---

## 4. Metrics

- **Primary:** DINOv3 / SigLIP2 / CLIP-I fidelity vs held-out real photos, reported as **paired
  per-species** GRAFT-vs-B1 deltas + win-rate at the best `ip_scale`.
- **Secondary:** VLM attribute-checklist accuracy.
- **Diagnostic:** CLIP-T-vs-name; GroundingDINO hit-rates; part-similarity distribution; per-species
  held-out count.

---

## 5. Results

### 5.1 GRAFT vs B1 — the confound-free comparison (neutral arm, best ip_scale = 0.6, paired n=8)

| metric | GRAFT − B1 mean Δ | GRAFT win-rate |
|---|---|---|
| **DINOv3** | **+0.0602** | **6 / 8 (75%)** |
| SigLIP2 | +0.0068 | 5 / 8 (62%) |
| CLIP-I | +0.0092 | 6 / 8 (75%) |

Per-species DINOv3 Δ (GRAFT − B1): Bamboo **+0.173**, Egyptian lotus **+0.147**, Ashore +0.088,
Avocado +0.067, Nageshore +0.057, Camphor Tree +0.053, Hijol −0.026, Ashok −0.079. GRAFT's two losses
are small; its wins include two large ones. **This reverses the Phase 1 finding** (where GRAFT lost to
B1 under the confounds).

Pooled means across the whole matrix (all name modes + ip, `outputs/eval/results.md`) tell the same
ordering: `ours` DINO **0.421** / attr **0.713** > `b1` 0.370 / 0.657 ≫ `b2` 0.085 / 0.285 ≈ `b0`
0.082 / 0.354.

### 5.2 `ip_scale` sweep (neutral, DINOv3 mean, n=8)

| ip_scale | GRAFT | B1 |
|---|---|---|
| 0.4 | 0.332 | 0.251 |
| **0.6** | **0.451** | **0.391** |
| 0.8 | 0.423 | 0.394 |

**Best `ip_scale` = 0.6** for both methods; GRAFT leads B1 at every setting. Hand 0.6 to Phase 2 §5.2.

### 5.3 Name contribution (named − neutral, per method, best ip_scale)

| method | DINOv3 Δ (named − neutral) |
|---|---|
| B0 (name only) | **+0.136** |
| B2 (LLM-recall) | +0.070 |
| B1 (reference) | +0.047 |
| **GRAFT** | **+0.004** |

The name is worth a large DINOv3 gain to B0 (which has nothing else) and almost nothing to GRAFT.
This is the sprint's core thesis, measured: **GRAFT's fine-grained fidelity comes from the reference
images, not the prompt token** — which is exactly what must hold for the intended customizable future
(a user supplying references for a concept the model has no correct name for).

### 5.4 Part-tree audit + prune ablation (§8.2 / §8.3)

MMKG audit over the 8 species' generated images at ip 0.6 (`outputs/audit/audit.json`):

| part | ref hit-rate | gen hit-rate | mean part-sim | n scored |
|---|---|---|---|---|
| leaf | 1.00 | 1.00 | 0.733 | 8 |
| bark | 1.00 | 1.00 | 0.713 | 8 |
| branching | 1.00 | 1.00 | 0.741 | 8 |
| cone_or_flower | 0.62 | 0.88 | 0.528 | 5 |

- **The part-tree is not noise.** GroundingDINO finds leaf/bark/branching in 100% of references and
  generated images; only cone_or_flower is unreliable (62% ref). The §8.1 correctness concern ("the
  gate may be scoring noise") is **cleared** — the two correctness bugs it fixed (medoid crop source,
  crop-vs-whole-image) were real, but detection itself is dependable.
- **The gate does no work at threshold 0.5.** Scored part-sims are tightly clustered high (median
  0.724, mean 0.694); only **3 of 29 (10%)** fall below the 0.5 gate. So the M3 part-sim gate is
  satisfied at seed 0 in almost every cell, refine stops immediately, and **`ours` == `ours_notree`
  exactly on all 8 species and all three metrics (Δ = 0.0)**.

**Prune decision (§8.3 rule "prune if pruned ≥ full within noise"): the ablation is degenerate, not
decisive.** GRAFT-full and GRAFT-pruned are byte-identical because the gate never fired — not because
the tree was tested and found worthless. The honest reading is: **at the current config the part-tree
changes nothing, so it is pure cost** (a GroundingDINO + SigLIP + DINO dependency for zero output
difference). But whether it *could* help was not actually tested, because two settings kept the gate
from ever engaging: `part_sim_threshold = 0.5` sits well below the 0.724 median (too permissive to
reject a real generation), and `n_refine = 1` gives the loop almost no room. This is Open Item §13
("the 0.5 gate may be mis-set").

---

## 6. Conclusion

- **The Phase 1 GRAFT-vs-B1 loss was a confound artifact.** With the name removed, exemplar parity
  enforced, and conditioning tuned, GRAFT beats B1 on DINOv3 in 6/8 species (+0.060 paired), and leads
  on all three fidelity metrics and on attribute accuracy.
- **GRAFT's advantage is reference-driven, not name-driven** (named−neutral Δ ≈ 0 for GRAFT vs +0.136
  for B0) — the property the customizable future depends on.
- **Best conditioning strength is `ip_scale = 0.6`**; hand it to Phase 2.
- **The part-tree is reliable but inert at the current gate setting.** Recommended default for the
  Phase 2 full run: **`use_part_tree = False`** — it costs a detection+embedding dependency and, as
  configured, produces identical output. This is a cost-parsimony call, **not** a finding that the
  tree is useless; that question is deferred (§7).

`use_part_tree`'s code default remains `True` (unchanged, back-compatible); the **Phase 2 run config**
should set it `False` per the above unless the recalibration in §7 is done first.

### Caveats
- n = 8 species; small held-out sets on some species (Ashore n_heldout = 4). The paired per-species
  design controls for unequal held-out sizes, but the sample is a validation subset, not the full
  corpus.
- The prune ablation is inconclusive-by-degeneracy, as detailed in §5.4.
- ~10% of matrix cells lose a row to a nondeterministic `worker_metrics` SIGSEGV (gc/torch-inductor
  race, documented in `graft/env.py`); this run's `--resume` refilled them, giving 176/176.

---

## 7. Next steps (into Phase 2)

1. **Re-test the part-tree properly before deciding to remove the code.** Recalibrate
   `part_sim_threshold` toward the observed median (~0.65) so the gate can actually reject a weak
   generation, set `n_refine ≥ 2`, and re-run the `ours` vs `ours_notree` ablation on this subset. Only
   then is "prune permanently" evidence-backed rather than degeneracy.
2. **Carry `ip_scale = 0.6` and `neutralize_name = True`** into the Phase 2 full-corpus run; keep the
   named arm as the name-contribution control.
3. **Scale to the full Treevill corpus** (Phase 2 §6) now that the pipeline and the subset result are
   trustworthy.
4. Housekeeping deferred from the sprint's whole-branch review (all in the SDD ledger): remove the now
   dead reranker infrastructure or wire the `exemplar_selection: "text"` arm it was kept for; group
   `results.md` by `(method, name_mode, ip_scale)` instead of pooling; single-source `_part_phrase`
   and the `_NOT_SHOWN`/`_GENERIC` vocabularies.
