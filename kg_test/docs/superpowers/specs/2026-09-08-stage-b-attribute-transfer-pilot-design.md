# Stage-B pilot — does taxonomy attribute-transfer beat retrieval on rare attributes? (design)

**Date:** 2026-09-08
**Status:** design, pre-registered. **Author sign-off pending.**
**Context:** The visual-repair MMKG hypothesis is refuted (`reports/2026-09-08-mmkg-hypothesis-verdict.md`):
SigLIP2 hubs are redundant with NN (HubNN), and an orthogonal taxonomy relation carries information but
that information has **no visual-restoration value** (OracleHub ≈ random, < NN). Stage-B asks a *different*
question that those results do **not** touch, because DINO visual fidelity is the wrong metric for it:

> **Hypothesis.** An independent relational signal — taxonomic relatedness — provides useful **attribute**
> information (attribute correctness / rare-attribute recall) beyond visual-NN retrieval.

This is a **gated pilot**, not a full benchmark. Both success bars are **pre-registered here** and must not
be changed after seeing results. If the pilot fails, the project winds down with the standing conclusion
intact (MMKG gives no advantage for visual repair, and none for attribute transfer either).

## 1. Two-gate structure

- **Gate 0 (premise; CPU + VLM; NO image generation).** For a **held-out target concept**, does taxonomy
  attribute-transfer predict its *rare* attributes better than visual-NN retrieval? Fail → **wind down** (no
  premise, no point generating).
- **Gate 1 (generation pilot; GPU; only if Gate 0 passes).** 20–40 cases: generate the held-out target
  concept's image under a fixed template that injects an attribute value — NN-derived vs taxonomy-derived;
  VLM reads the attribute back off the generated image; score rare-attribute correctness. Pass → proceed to
  full Stage-B; fail → wind down.

Gate 1 details finalize only after Gate 0 passes; §4 fixes its bar now so it cannot drift later.

## 2. Shared definitions (pre-registered)

**2.0 Concept universe.** **Evaluable concepts** (can have a held-out label) = the **25** concepts with a
held-out store in `outputs/mmkg_oh/` (all in multi-member families). **Prediction pool** (source of arm
predictions) = **all 65** concepts with kg.json part exemplars + attributes. So a case's target concept C is
one of the 25; its retrieval NN and taxonomy siblings are drawn from the 65 (excluding C). Rarity (§2.3) is
computed over the 25 evaluable concepts' held-out labels.

**2.1 Attributes (5, part-level only):** `leaf.shape`, `leaf.texture`, `bark.texture`, `bark.color`,
`branching.pattern`. Global attrs, `*.count`, and `cone_or_flower.appearance` (≈always "not visible") are
excluded.

**2.2 Held-out reference label** (the evaluation target — a *held-out pseudo-ground-truth*, NOT
human-verified truth). These are **held-out target concepts**, not an independently defined "data-poor"
split; the only claim is the held-out/build separation asserted below.

**Build/held-out disjointness (asserted, per case).** Let `B(C) = split_refs(load_species(C), k_build=5,
seed=0)[0]` — the exact build images that produced C's kg.json exemplar and attributes. The label is read
**only** from C's held-out images whose `ref_path ∉ B(C)`. The code **asserts** `H(C) ∩ B(C) = ∅` (the
mmkg_oh store was built at k_build=0, so this filter is what enforces disjointness). If C has **zero**
held-out images disjoint from `B(C)`, C is **dropped** (no clean label). Consequence: for low-unique
concepts the label may rest on a single held-out image; that is accepted (it is still strictly held out).

For concept C and attribute (part P, name A):
- Read **each unique disjoint held-out image of C** (dedupe records by `ref_path`, drop those in `B(C)`,
  read each remaining image once, taking all attributes from that one read) with **Qwen2.5-VL** using the
  **name-blind** `kg_build.SCHEMA_INSTRUCTION_NEUTRAL` instruction, `do_sample=False`,
  `max_new_tokens=512`; parse via `kg_build.parse_vlm_schema`; take `parts[P][A]`.
- **Normalize** a value = `str(v).strip().lower()`.
- **Label = strict plurality** across the held-out reads, over normalized values, **excluding** any read
  whose value ∈ {"not visible","none","n/a",""}.
- **Exclusions:** if no visible read, or the plurality is **tied**, the case is **dropped** (ambiguous /
  no evaluable label).
- The held-out images feed **only** the label. They never enter graph construction, retrieval, or transfer
  (those use other concepts' kg.json build attributes and build-image exemplar embeddings).

**2.3 Rarity** (deterministic, matcher-independent to avoid circularity). For attribute (P,A), over all
concepts' held-out reference labels (visible only), count concepts per **exact-normalized** value. A case is
**rare** iff its label value's concept-count ≤ `max(2, ceil(0.15 · N_visible))`, where `N_visible` = number
of concepts with a visible label for (P,A). (Exact-string bucketing is conservative: it splits near-synonyms
into separate buckets, if anything over-counting rares.)

**2.4 Arms** (both predict C's value from **other** concepts' kg.json *build* attributes; C's own data is
never used to predict C):
- **Retrieval (NN).** Rank concepts (≠C) by cosine between C's part-P SigLIP2 exemplar (kg.json) and theirs;
  prediction = the **nearest concept that has a visible build value** for (P,A) — walk the NN order until
  one exists.
- **MMKG (taxonomy).** Siblings = concepts (≠C) in C's family (`graft/taxonomy.py`) with a visible build
  value for (P,A). Prediction = **plurality** sibling value (normalized); ties broken by the sibling with
  the **highest part-P exemplar cosine** to C (deterministic). If **no** sibling has a visible value, the
  case is **dropped** (MMKG has no prediction — cannot form the pair).

**2.5 Case inclusion.** A (C,P,A) case is included iff: C is in a multi-member family (`taxonomy.py`); its
held-out label is valid, visible, non-tied (§2.2); ≥1 sibling has a visible value (§2.4); retrieval yields a
prediction. The **rare subset** is the included cases whose label is rare (§2.3).

**2.6 Semantic match** (scoring; **VLM-judge, no threshold**). `match(x, y)` for attribute name A:
- Judge = Qwen2.5-VL, text-only, `do_sample=False`, `max_new_tokens=8`.
- **Frozen prompt:** `You are comparing two descriptions of a plant's {A}. A: "{x}". B: "{y}". Do A and B
  describe essentially the same {A}? Answer with only 'yes' or 'no'.`
- Parse: strip, lowercase, `startswith("yes")` → yes, else no.
- **Order-symmetric:** evaluate both (x,y) and (y,x); `match` iff **both** return yes (conservative).
- A prediction whose value ∈ {"not visible",…} counts as an automatic **miss** (non-answer).

## 3. Gate 0 — metric, power check, decision (pre-registered)

- **Metric:** per arm, **rare-attribute recall** = (#rare cases where the arm's prediction `match`es the
  label) / (#rare cases).
- **Test:** paired **McNemar** on the per-case match/miss of MMKG vs retrieval over the rare cases.
- **Power/sanity (reported, not gating):** (a) retrieval rare-recall — a meaningful test needs the baseline
  to be failing on rares; (b) both arms' recall on **common** (non-rare) cases — both should be reasonable,
  confirming the pipeline works.
- **PASS bar (all three required):** `rare_recall(MMKG) − rare_recall(retrieval) ≥ 0.15`, **n_rare ≥ 30**,
  **McNemar p < 0.05**. If n_rare < 30 the gate is under-powered → treated as **not passed** (wind down),
  never widened post-hoc.
- **Decision:** PASS → run Gate 1. FAIL → **wind down**; record the result; the standing MMKG verdict stands.

## 4. Gate 1 — generation pilot (bar frozen now; mechanics finalized post-Gate-0)

- **Cases:** 20–40 rare (C,P,A) drawn from Gate-0-positive cases (concepts with held-out geometry already
  built).
- **Arms (fully matched — the ONLY difference is the attribute *value*):** one fixed conditioning template
  `T(value)` and one base generator. **NN arm** = `T(NN-derived value)`; **MMKG arm** = `T(taxonomy-derived
  value)`. Same prompt, same conditioning budget, same seed, **no extra reference image for either arm** — so
  a Gate-1 difference reflects *attribute-source quality* alone, not extra visual information. The exact
  template `T` is fixed after Gate 0, but the **arms, the "value-only difference" constraint, the metric, and
  the bar do not change**.
- **Metric:** Qwen-VL reads the target attribute off the **generated** image (§2.2 read + §2.6 match to the
  held-out label). **Rare-attribute correctness.**
- **PASS bar:** `correctness(MMKG) − correctness(retrieval) ≥ 0.15`, n = 20–40, paired **sign test p < 0.05**.
  PASS → full Stage-B benchmark. FAIL → wind down.

## 5. Anti-search / honesty guards

- Every knob above is **frozen before running** — attributes, label rule, rarity, arms, judge prompt,
  bars. No τ (VLM-judge). No metric or threshold shopping.
- A **single pre-registered analysis** per gate. Both arms are evaluated identically against the same
  held-out label using the same frozen judge; the evaluation is procedurally symmetric and paired.
  Because the label and the judge share one VLM, **residual VLM-dependent bias is treated as a limitation
  rather than assumed absent** (e.g. the judge could systematically favour phrasings closer to the label's
  own idiom). It is mitigated — not eliminated — by symmetry and pairing.
- Gate 0 is reported in full (both recalls, McNemar, n, the power/sanity numbers) whether it passes or fails.
- Gate 1 is entered **only** on a Gate-0 pass; a Gate-0 fail ends the line.

## 6. Scope, reuse, non-goals

- **Reuses:** `graft/taxonomy.py`, kg.json attributes + exemplar embeddings, held-out store `outputs/mmkg_oh/`,
  `kg_build` VLM read path, `models.vlm`. **No new graph build.**
- **MMKG signal = taxonomy siblings only.** No attribute-value hubs — building hubs from the attribute values
  we then predict would encode the answer into the graph (circular). Taxonomy is the independent relation.
- **Non-goals:** visual fidelity (closed), full Stage-B benchmark (gated behind the pilot), human-verified
  ground truth (held-out VLM pseudo-label is explicitly accepted for the pilot), attribute-hub variants.

## 7. Implementation outline

- `graft/stageb_gate0.py` — case construction (§2.5), arm predictions (§2.4), rarity (§2.3), metric + McNemar
  (§3). Pure/CPU except the two VLM steps.
- `graft/stageb_vlm.py` — `read_heldout_label(concept, records, models)` (§2.2, reuses `kg_build`) and
  `judge_match(x, y, attr, models)` (§2.6). Deterministic decoding.
- Reuse: `graft/taxonomy.py`, `graft/dataset.py`, `outputs/mmkg_oh/heldout_parts/`.
- Tests (CPU, mock VLM): case inclusion/exclusion rules, rarity bucketing, both arm predictions + tie-breaks,
  judge parse (yes/no/garbage), McNemar. VLM steps behind a thin interface so tests inject fixed reads.
- **Feasibility pre-check (first, cheap, no VLM):** using kg.json build attributes as a proxy for label
  values, estimate the rare-case count. If it cannot plausibly reach n_rare ≥ 30, surface it before spending
  the VLM pass — a data limit, reported like the OracleHub blocker, not a silent under-powered run.
- Run: one CPU/VLM pass → `outputs/mmkg/stageb_gate0/result.json` (both recalls, delta, McNemar p, n_rare,
  power numbers) → `reports/2026-09-08-stageb-gate0-result.md` with the PASS/FAIL decision.
