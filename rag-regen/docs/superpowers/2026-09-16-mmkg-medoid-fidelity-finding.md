# Finding — MMKG medoid fidelity pilot: NO LARGE EFFECT (close the fidelity direction)

**Date:** 2026-09-16
**Spec:** `docs/superpowers/specs/2026-09-11-mmkg-medoid-fidelity-pilot-design.md` (frozen 2026-09-11, pre-data)
**Plan:** `docs/superpowers/plans/2026-09-11-mmkg-reference-integration.md` (plumbing) + the fidelity-pilot plan on this branch
**Branch:** `mmkg-reference-integration` (worktree); pilot harness `ca350b7`, GPU runner `a109049` + uncommitted runner fixes
**Run:** 25/25 iNat pilot birds, 0 dropped. GPU: RTX A5000 on `192.168.20.245`. `outputs/mmkg_medoid_pilot/result.json`.

## Verdict: `NO_LARGE_EFFECT` → close the medoid fidelity direction

Pre-registered decision rule (spec §7): **GAIN** iff `mean ≥ +MARGIN` **and** `CI_low > 0`; **LOSS** iff `mean ≤ −MARGIN` **and** `CI_high < 0`; otherwise **NO LARGE EFFECT**. `MARGIN = 0.02` DINO cosine (`ragregen/metrics.py` `metrics.MARGIN`, fixed pre-data).

| Read-out (Δ = medoid − baseline, per species then aggregated) | Value |
|---|---|
| n (matched species pairs) | 25 (0 dropped) |
| **DINO mean Δ** (primary) | **+0.0215** |
| **DINO 95% CI** | **[−0.0063, +0.0494]** — includes 0 |
| DINO sign split | 17/25 species favor medoid |
| DINO delta range / median | −0.146 … +0.162 / +0.024 |
| DINO: cleared +MARGIN / in ±band / below −MARGIN | 13 / 7 / 5 |
| **SigLIP mean Δ** (secondary, descriptive) | **−0.0000** (flat), 14/25 |

**Why null, not GAIN:** the point estimate cleared the margin (mean +0.0215 ≥ 0.02), but the significance leg did **not** — the 95% CI lower bound is −0.006, i.e. the interval does not clear a margin away from 0. Per §7 that is `NO_LARGE_EFFECT`. The per-species deltas swing widely in **both** directions (−0.146 to +0.162), so the small positive mean sits inside high variance; there is no systematic large effect. SigLIP is dead flat.

## Interpretation (within the frozen §12 claim boundary)

- This measures **one mechanism only**: a precomputed, build-time SigLIP-centroid **medoid** as the repair reference, vs a deterministic same-species **non-medoid** build image (§4 rule), on a **birds-only** substrate, scored by DINO (primary) / SigLIP (secondary).
- The result is an honest **"no large effect detected"** — *not* a proof of exact equivalence. Given the parity prior (Stage-1, spec §2), it reads as: **medoid selection does not meaningfully change repair fidelity in this controlled setup**, and it **closes the fidelity direction of the medoid mechanism**. Stage-1's *predicted* parity is now a *measured* "no large effect."
- The directional DINO lean (17/25, mean +0.021) is **descriptive only**; it does **not** license a GAIN wording and must not be reported as an improvement.

**What this does NOT say (§9/§12):** nothing about MMKG attributes, part crops, taxonomy/graph relations, FAISS/`nearest_crops`, reranking, or VLM identity judges — all untested here. It does **not** generalize to "MMKG improves generation," and does **not** extend beyond birds (TreeVill excluded, no clean split). It speaks to reference **curation** (build-time selection), not to any repair-time retrieval algorithm (no repair-time NN ran in either arm — spec §9 invariant).

## No-rescue (frozen, honored)

n = 25 was powered for a **large** effect only, by design. Per spec §8 the pilot **must not** add seeds/drafts/species to chase significance. No large effect → **close**. No rescue was or will be attempted. A separate, properly-powered experiment would be *new* work, not a patch to this pre-registration — and the null here gives no reason to open one.

## Provenance / reproducibility notes

- Same frozen run as the aborted 2026-09-15 `.202` attempt (that GPU threw a hardware `CUDA error: unspecified launch failure` at ~2/25; no result). Re-run on `.245` recomputes all 25 seed-for-seed (`seed_base + i`), so it is the same pre-registered experiment, not a post-hoc variant.
- `.245` gotcha (recorded in project memory): pin the A5000 with **`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`** — default CUDA ordering (`FASTEST_FIRST`) selected the 12 GB P100 and core-dumped nf4/bf16 FLUX. The queue wrapper's `flock` also hangs on `.245` (NFS lock-manager stall on the `192.168.6.133` share); the runner was launched directly instead.
- Code implementing this pilot remains **uncommitted** on the parked branch pending an explicit decision on whether to keep it.
