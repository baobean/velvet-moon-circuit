# Relational MMKG — Stage A result (2026-09-06)

**Run:** full 384-cell starvation/recovery experiment (8 concepts × build∈{1,2,3,5} × 3 draws ×
{isolated, random, rawnn, hub}), resident two-phase runner (`graft/run_starvation_resident.py`),
graph = 12 valid cross-concept PartType hubs (coherence 0.90–0.97, 31/32 (concept,part) pairs borrowable).
Artifacts: `outputs/mmkg/starve/` (rows + recovery.json + hub_vs_rawnn.json), images under `_imgs/`.

## Headline — the make-or-break test comes back NULL

**+Hub-k − +RawNN-k, build=1, DINOv3 (primary): mean Δ = +0.0053, win 3/8 (38%), n=8.**
Essentially zero, with a losing win-rate. Across levels it stays noise-around-zero
(+0.005 / +0.012 / −0.015 / +0.006 for build 1/2/3/5). SigLIP2: +0.005 (7/8); CLIP-I: −0.007 (4/8).
**Per §10, this is the "collapses into retrieval" outcome: the shared PartType hubs do NOT beat plain
nearest-neighbour retrieval.** We report it straight.

## Worse: borrowing of any kind slightly HURTS vs the isolated per-concept schema

**+Hub-k − +Isolated is NEGATIVE at every build level** (−0.016 / −0.027 / −0.002 / −0.023). The best
condition at build=1 is **isolated (0.317)** > hub (0.301) > rawnn (0.296) > random (0.271). Feeding
another concept's part-crops into the IP-Adapter reference pool pulls the output toward the neighbour's
appearance (identity dilution) despite the 2:1 own-weighting + 0.7 cap. Hub *does* beat **random**
(+0.030, 75% at build=1) — i.e. same-part hub selection beats random off-concept crops — but both are
worse than not borrowing at all.

## The manipulation didn't bite — the recovery test is partly degenerate

The recovery curve is **nearly flat across build 1→5** (isolated 0.317 → 0.290; build=1 is actually the
*highest*). Reducing the build set from 5 images to 1 barely changed DINOv3-vs-held-out fidelity. So the
core premise — "data-poor concepts have a fidelity deficit that transfer can recover" — **never
instantiated a deficit to recover.** Whatever the cause (single-reference IP-Adapter conditioning is
already robust to fewer refs; or DINOv3-vs-held-out is insensitive at this scale/noise), the experiment
could not demonstrate recovery because there was little to recover.

## Honest verdict

- The MMKG-as-exemplar-transfer mechanism, **as built (Stage A)**, does **not** earn its keep: it ≈
  retrieval on the primary metric and slightly underperforms the isolated schema.
- This is a **null with a caveat**, not a clean refutation of the idea: the starvation lever had almost
  no effect on the metric, so the test was underpowered by construction.

## Likely confounds to resolve before judging the idea (not spin — real setup issues)

1. **Starvation doesn't reduce fidelity** → no deficit → no recovery to measure. Diagnose why build=1
   isn't data-poor in this pipeline (metric insensitivity? robust single-ref conditioning?). Without a
   working deficit, the recovery design cannot test the hypothesis.
2. **Per-part Stage-A simplification** (borrow crops for `own[0]`'s part, mixed into whole-image
   IP-Adapter conditioning) is a prime suspect for hub < isolated: whole-image conditioning on a
   borrowed part-crop dilutes identity rather than fixing one part. The intended design edits the
   *specific* failing part (inpaint), which this run did not do.
3. **Exemplar transfer may be the wrong channel.** Borrowing pixels pulls appearance toward the
   neighbour. Attribute-prior transfer (Stage B) is a different channel that this run did not test.
4. **Measurement**: n=8, single held-out on some concepts, 3 draws; the deltas (~0.005–0.03) are within
   noise — same reliability problem as the sprint §0.

## Recommendation

Do NOT proceed to a big MMKG or the rag-regen swap on this evidence — the transfer mechanism has not
demonstrated value. Before abandoning the idea, fix the test: (a) establish a manipulation that actually
degrades fidelity (so "recovery" is measurable), and (b) implement per-part (inpaint) transfer rather
than whole-image conditioning on borrowed crops. If, with a real deficit and per-part transfer, +Hub
still ≈ +RawNN, that is the decisive negative.
