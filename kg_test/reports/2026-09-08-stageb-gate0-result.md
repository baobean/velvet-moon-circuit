# Stage-B Gate 0 result — does taxonomy attribute-transfer beat retrieval on rare attributes? NO

**Date:** 2026-09-08
**Spec (frozen, pre-registered):** `docs/superpowers/specs/2026-09-08-stage-b-attribute-transfer-pilot-design.md`
**Run:** `outputs/mmkg/stageb_gate0/result.json` (25 held-out target concepts, 5 part attributes,
Qwen2.5-VL held-out labels + frozen bidirectional judge).

## Decision: WIND_DOWN — pre-registered bar not met

| pre-registered condition | required | observed | met? |
|---|---|---|---|
| rare-attribute recall gain (MMKG − retrieval) | ≥ +0.15 | **−0.105** | ✗ (wrong sign) |
| n_rare | ≥ 30 | **19** | ✗ (under-powered) |
| McNemar p | < 0.05 | 0.50 | ✗ |

All three fail. Per the spec's honesty rule (n<30 → not passed, never widened post-hoc; single
pre-registered analysis), the pilot **does not pass** and the project **winds down**.

## Numbers

- **Rare cases (n=19):** recall MMKG **0.579** vs retrieval **0.684**, Δ = **−0.105**.
  Discordant rare cases: MMKG-only-correct = **0**, retrieval-only-correct = 2 (both-correct 11,
  neither 6). **Taxonomy never once recovered a rare attribute that retrieval missed.**
- **Common cases (n=56, sanity):** recall MMKG 0.804 vs retrieval 0.875 — the pipeline works (both
  arms match most common attributes), and retrieval edges MMKG here too.
- Concrete rare misses (judge working, MMKG off-family value): Haldu/leaf.shape label
  "elliptical with pointed tip" — retrieval "oval" (hit) vs MMKG "elliptical" (miss);
  Marking Nut tree/bark.color label "gray-brown" — retrieval "brown" (hit) vs MMKG "dark brown" (miss).

## Why n_rare came in at 19, not the proxy's 42

The pre-registered feasibility pre-check used kg.json **build** attributes as a proxy for labels and
estimated 42 rare cases; it cleared the ≥30 gate, so the VLM pass ran (as designed). The **actual**
held-out labels are sparser: for 20 of 25 concepts the label rests on a **single** disjoint held-out
image (accepted in the spec), and single-image reads collapse to "not visible" or to a common value
more often than the multi-image build attributes — shrinking included cases (96→75) and rare cases
(42→19). This is a measurement/data limit, disclosed here; per pre-registration the observed n governs
the decision.

## Interpretation (careful, not over-claimed)

- With 19 cases and 2 discordant pairs, this is **not** a statistically decisive claim that taxonomy is
  *worse*. What it **is**: the pilot produced **no hint of the required gain** — the point estimate is
  negative, and taxonomy uniquely helped on **zero** rare cases. There is no positive signal to justify
  spending GPU on the Gate-1 generation pilot.
- Consistent with the OracleHub result: a taxonomic sibling's attribute value is a family-typical value,
  which for a *specific* target's *rare* attribute is systematically the wrong (common/off) value —
  retrieval's visually-nearest concept is at least appearance-matched.
- **Limitation:** the label and the judge share one VLM; residual VLM-dependent bias is a limitation,
  not assumed absent (spec §5). The Haldu example shows judge noise (calling "elliptical with pointed
  tip" ≠ "elliptical" but = "oval"). Both arms are judged identically, so this adds noise, not a
  directional advantage to either arm.

## Consequence — project winds down

This was the one MMKG contribution left unrefuted by the visual-repair results
(`reports/2026-09-08-mmkg-hypothesis-verdict.md`). The premise gate says taxonomy attribute-transfer
does **not** beat retrieval on rare attributes, so:

- **Do not** run the Gate-1 generation pilot or the full Stage-B benchmark.
- **Standing conclusion holds across the board:** a relational MMKG (embedding-hub, taxonomy-hub, or
  taxonomy attribute-transfer) provides **no advantage over retrieval** — for visual repair (decisive)
  or for rare-attribute recovery (this gate, under-powered but with zero positive signal). The robust,
  supported finding remains that **borrowing a good *retrieved* reference helps** (Stage-A′
  borrow−isolated +0.12; rawnn−random +0.062); the value is in retrieval quality, not graph structure.
- If Stage-B is ever revisited, it needs a genuinely different, non-VLM-circular attribute ground truth
  and a denser per-concept image budget than Treevill provides — i.e. a new dataset (e.g. the scoped
  FewMedical-XJAU, `reports/2026-09-07-fewmedical-xjau-scope.md`), not a re-run here.

## Reproducibility

`graft/stageb_gate0.py`, `graft/stageb_vlm.py`; tests `tests/test_stageb_gate0.py` (9),
`tests/test_stageb_vlm.py` (2); 136 non-GPU tests pass. Config `configs/oraclehub.yaml`. Held-out store
`outputs/mmkg_oh/`. Every knob frozen pre-run (spec); no threshold or metric changed after seeing results.
