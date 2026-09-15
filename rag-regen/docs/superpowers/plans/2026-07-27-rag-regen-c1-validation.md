# rag-regen Plan 2 — C1 Verifier Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure claim C1 — does grounded + semantic fusion detect prompt-image mismatch better
than a single VLM judge — against the 22 hand-labelled drafts, and calibrate `tau` from data
instead of guesswork.

**Architecture:** Two short-lived GPU stages that never co-reside write per-case scores to JSON;
a third CPU-only stage fuses them, sweeps `tau`, and renders the report. Stream A caches
*continuous* similarities rather than thresholded verdicts, so every threshold is recovered by
arithmetic instead of re-running GroundingDINO.

**Tech Stack:** Python 3.11 (`kontext` conda env), transformers 5.14.1, Qwen2.5-VL-7B-Instruct,
GroundingDINO-base, SigLIP-SO400M-384, pytest. No diffusers — this plan runs no generation.

**Spec:** `docs/superpowers/specs/2026-07-27-rag-regen-c1-validation-design.md`
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md`
**Out of scope (Plan 3):** `query.py`, `mask.py`, `regen.py`, `schedule.py`, `metrics.py`, Stage 2.

## Global Constraints

- **Interpreter:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` (3.11.15). Never
  `ImageRAG_qwen` — it is 3.10.13 with transformers 4.44.2.
- **`HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache`** before any transformers import.
  `ragregen.env.setup()` does this; import `env` first.
- **Model weights live at `.cache/models--*`, not `.cache/hub/models--*`**, because every loader
  passes `cache_dir=str(env.HF_CACHE)`. The `hub/` entries are stubs — do not judge cache state
  from them.
- **Stream B model:** `Qwen/Qwen2.5-VL-7B-Instruct`, already cached (16 GB). Do **not** download
  Qwen3-VL: it is deliberately reserved as the held-out judge (parent spec §5).
- **One GPU:** RTX 4090, 24 GB, shared, routinely ~9 GB held by others. Qwen-7B (~16 GB) and the
  detector stack must **never** be loaded in the same process.
- **Dependency injection is mandatory.** No module-level model loading, no `from_pretrained` at
  import time.
- **Test markers:** GPU tests are `@pytest.mark.gpu`; the default run excludes them.
- **Fakes are built from `inspect.signature` of the real API, never from assumption.** Plan 1
  shipped a green suite that crashed on the first real case because `FakePipe` declared a signature
  `FluxKontextPipeline` did not have.
- **`state_at_tau` must use `>=`**, matching `ragregen/verify/grounded.py:83`, which Plan 1 pins
  with a dedicated boundary test.
- All paths are relative to `/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen/`.

---

## File Structure

| file | responsibility |
|---|---|
| `ragregen/c1.py` | Pure metrics and report rendering. No models, no I/O beyond reading JSON/CSV. |
| `ragregen/vlm.py` | `QwenVLM` implementing Plan 1's `VLM` protocol. The only new model adapter. |
| `scripts/score_a.py` | Stream A sweep over labelled drafts → `stream_a.json`. |
| `scripts/score_b.py` | Stream B sweep → `stream_b.json`. |
| `scripts/c1_report.py` | Fuse, sweep `tau`, render `c1.md` + `c1.json`. |
| `scripts/screen_premise.py` | **modify** — emit the `verdict_identity` column. |
| `scripts/run.sh` | **modify** — dispatch `score-a`, `score-b`, `c1`. |

---

## Task 1: Labels — the `verdict_identity` column and its loader

**Files:**
- Create: `ragregen/c1.py`, `tests/test_c1_labels.py`
- Modify: `scripts/screen_premise.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `c1.LabelRow(case_id, prompt, concept, draft_path, verdict, verdict_identity, notes)`,
  `c1.load_labels(path: Path) -> list[LabelRow]`,
  `c1.ground_truth(rows, column: str) -> tuple[list[LabelRow], list[bool], int]` returning
  (kept rows, `is_fail` flags, number excluded for a blank verdict).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c1_labels.py
import pytest

from ragregen import c1

HEADER = "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"


def _csv(tmp_path, body):
    p = tmp_path / "labels.csv"
    p.write_text(HEADER + body)
    return p


def test_load_labels_reads_both_verdict_columns(tmp_path):
    p = _csv(tmp_path, "fox,a fox,fox,/d/fox.png,pass,fail,note text\n")
    rows = c1.load_labels(p)
    assert len(rows) == 1
    assert rows[0].case_id == "fox"
    assert rows[0].verdict == "pass"
    assert rows[0].verdict_identity == "fail"
    assert rows[0].notes == "note text"


def test_ground_truth_maps_fail_to_true(tmp_path):
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n"
                       "b,p,c,/d/b.png,pass,pass,\n")
    rows = c1.load_labels(p)
    kept, flags, excluded = c1.ground_truth(rows, "verdict_identity")
    assert flags == [True, False]
    assert excluded == 0
    assert len(kept) == 2


def test_ground_truth_excludes_blank_verdicts_and_counts_them(tmp_path):
    """A blank verdict is never guessed at -- it is dropped and counted."""
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n"
                       "b,p,c,/d/b.png,,,\n"
                       "c,p,c,/d/c.png,pass,pass,\n")
    rows = c1.load_labels(p)
    kept, flags, excluded = c1.ground_truth(rows, "verdict_identity")
    assert [r.case_id for r in kept] == ["a", "c"]
    assert flags == [True, False]
    assert excluded == 1


def test_ground_truth_rejects_an_unknown_column(tmp_path):
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n")
    with pytest.raises(ValueError, match="unknown verdict column"):
        c1.ground_truth(c1.load_labels(p), "verdict_vibes")


def test_ground_truth_rejects_an_unparseable_verdict(tmp_path):
    """Anything that is neither pass, fail, nor blank is an operator typo and
    must be named, not silently coerced to one side."""
    p = _csv(tmp_path, "a,p,c,/d/a.png,maybe,maybe,\n")
    with pytest.raises(ValueError, match="case 'a'.*'maybe'"):
        c1.ground_truth(c1.load_labels(p), "verdict")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_c1_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.c1'`

- [ ] **Step 3: Implement**

```python
# ragregen/c1.py
"""Metrics for the C1 verifier validation.

Pure: no models, and no I/O beyond reading the labels CSV and the two stage
JSONs. That is what makes the arms comparison testable on CPU with fixtures,
the same split that made Plan 1's fusion.py verifiable without a GPU.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VERDICT_COLUMNS = ("verdict", "verdict_identity")


@dataclass(frozen=True)
class LabelRow:
    case_id: str
    prompt: str
    concept: str
    draft_path: str
    verdict: str
    verdict_identity: str
    notes: str


def load_labels(path: Path) -> list[LabelRow]:
    with Path(path).open(newline="") as fh:
        return [
            LabelRow(
                case_id=r["case_id"], prompt=r["prompt"], concept=r["concept"],
                draft_path=r["draft_path"], verdict=r.get("verdict", "").strip(),
                verdict_identity=r.get("verdict_identity", "").strip(),
                notes=r.get("notes", ""),
            )
            for r in csv.DictReader(fh)
        ]


def ground_truth(rows: list[LabelRow], column: str
                 ) -> tuple[list[LabelRow], list[bool], int]:
    """Rows carrying a verdict in `column`, plus their is_fail flags.

    Blank verdicts are excluded and counted rather than guessed at: an
    unlabelled case is not evidence either way.
    """
    if column not in VERDICT_COLUMNS:
        raise ValueError(
            f"unknown verdict column '{column}'. Known: {', '.join(VERDICT_COLUMNS)}")

    kept: list[LabelRow] = []
    flags: list[bool] = []
    excluded = 0
    for row in rows:
        value = getattr(row, column).strip().lower()
        if not value:
            excluded += 1
            continue
        if value not in ("pass", "fail"):
            raise ValueError(
                f"case '{row.case_id}': {column} is '{value}', expected "
                f"'pass', 'fail', or blank.")
        kept.append(row)
        flags.append(value == "fail")
    return kept, flags, excluded
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_c1_labels.py -v`
Expected: 5 passed

- [ ] **Step 5: Emit the new column from the gate**

In `scripts/screen_premise.py`, change the `FIELDNAMES` constant and the row written in the loop:

```python
FIELDNAMES = ["case_id", "prompt", "concept", "draft_path",
              "verdict", "verdict_identity", "notes"]
```

and in the `writer.writerow(...)` call add `"verdict_identity": "",` immediately after
`"verdict": "",`.

Update the closing operator message so it names both columns:

```python
    print(f"\n[gate] Open {labels}")
    print("[gate] Fill `verdict` (any visible defect) and `verdict_identity` "
          "(is it the right thing?) with pass|fail for every row.")
    print("[gate] If fewer than ~30% of `verdict` are `fail`, pick rarer concepts and re-run.")
```

- [ ] **Step 6: Backfill the column on the existing run**

The 22 labelled drafts predate the column. Their identity verdicts are recorded in `notes` as an
`[identity]` tag. Backfill once:

**Back the file up first — it holds irreplaceable hand-labelling, and `outputs/` is gitignored so
git cannot recover it.** Read and write with `newline=""` at both ends, and write to a temporary
file then rename: `open("w")` truncates immediately, so a mid-write exception destroys the labels.
That is not hypothetical — it happened during this plan's own execution.

```bash
cp outputs/screen_20260726_233320/labels.csv /tmp/labels.csv.backup

$PY - <<'EOF'
import csv
from pathlib import Path
p = Path("outputs/screen_20260726_233320/labels.csv")
fields = ["case_id","prompt","concept","draft_path","verdict","verdict_identity","notes"]
with p.open(newline="") as fh:                 # newline="" on read, too
    rows = list(csv.DictReader(fh))
assert len(rows) == 22, f"expected 22 rows, got {len(rows)} -- restore the backup"
for r in rows:
    r["verdict_identity"] = "fail" if "[identity]" in r["notes"] else "pass"
tmp = p.with_suffix(".csv.tmp")                # write-then-rename: a failed
with tmp.open("w", newline="") as fh:          # write cannot truncate the original
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
tmp.replace(p)
n = sum(r["verdict_identity"] == "fail" for r in rows)
print(f"verdict_identity: {n} fail / {len(rows)-n} pass")
EOF
```

Expected: `verdict_identity: 15 fail / 7 pass`

- [ ] **Step 7: Verify both ground truths load**

```bash
$PY -c "
from pathlib import Path
from ragregen import c1
rows = c1.load_labels(Path('outputs/screen_20260726_233320/labels.csv'))
for col in ('verdict', 'verdict_identity'):
    kept, flags, exc = c1.ground_truth(rows, col)
    print(f'{col:18} {sum(flags)} fail / {len(flags)-sum(flags)} pass, {exc} excluded')
"
```

Expected:
```
verdict            18 fail / 4 pass, 0 excluded
verdict_identity   15 fail / 7 pass, 0 excluded
```

- [ ] **Step 8: Commit**

```bash
git add ragregen/c1.py tests/test_c1_labels.py scripts/screen_premise.py
git commit -m "feat: verdict_identity column and label loading for C1"
```

---

## Task 2: Confusion matrix, Wilson intervals, and the metrics that survive a 68% base rate

**Files:**
- Modify: `ragregen/c1.py`
- Create: `tests/test_c1_metrics.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure numerics).
- Produces: `c1.Confusion(tp, fp, tn, fn)` with properties `precision`, `recall`, `specificity`,
  `f1`, `accuracy`, `balanced_accuracy`, `mcc`, `n`; `c1.confusion(y_true, y_pred) -> Confusion`;
  `c1.wilson_ci(k, n, z=1.96) -> tuple[float, float]`.

Positive = **the verifier says FAIL**. The ground truth is 15 fail / 7 pass, a **68% base rate**, so
an always-FAIL constant scores F1 0.81 while being useless. `balanced_accuracy` and `mcc` are the
metrics that put it at chance, and they are the headline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c1_metrics.py
import math

import pytest

from ragregen import c1


def test_confusion_counts_each_cell():
    y_true = [True, True, False, False]
    y_pred = [True, False, True, False]
    m = c1.confusion(y_true, y_pred)
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
    assert m.n == 4


def test_confusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        c1.confusion([True], [True, False])


def test_perfect_prediction():
    m = c1.confusion([True, False], [True, False])
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.balanced_accuracy == 1.0
    assert m.mcc == 1.0


def test_always_fail_on_the_real_base_rate_is_exposed_by_balanced_accuracy():
    """15 fail / 7 pass. An always-FAIL verifier must look good on F1 and
    useless on balanced accuracy and MCC -- that contrast is the whole reason
    those two are the headline metrics."""
    y_true = [True] * 15 + [False] * 7
    m = c1.confusion(y_true, [True] * 22)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(15 / 22)
    assert m.f1 == pytest.approx(2 * (15 / 22) / ((15 / 22) + 1))
    assert m.f1 > 0.8                      # flattering
    assert m.balanced_accuracy == 0.5      # chance
    assert m.mcc == 0.0                    # no information


def test_always_pass_is_also_at_chance():
    y_true = [True] * 15 + [False] * 7
    m = c1.confusion(y_true, [False] * 22)
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.balanced_accuracy == 0.5
    assert m.mcc == 0.0


def test_metrics_are_zero_not_nan_when_a_denominator_vanishes():
    m = c1.confusion([False, False], [False, False])
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
    assert m.mcc == 0.0
    assert not any(math.isnan(v) for v in (m.precision, m.recall, m.f1, m.mcc))


def test_mcc_is_negative_when_prediction_is_inverted():
    y_true = [True, True, False, False]
    m = c1.confusion(y_true, [False, False, True, True])
    assert m.mcc == -1.0


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = c1.wilson_ci(15, 22)
    assert lo < 15 / 22 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_ci_matches_a_known_value():
    """k=15, n=22, z=1.96 -> (0.4732, 0.8364), computed independently."""
    lo, hi = c1.wilson_ci(15, 22)
    assert lo == pytest.approx(0.4732, abs=1e-3)
    assert hi == pytest.approx(0.8364, abs=1e-3)


def test_wilson_ci_is_defined_at_the_boundaries():
    assert c1.wilson_ci(0, 10)[0] == 0.0
    assert c1.wilson_ci(10, 10)[1] == 1.0


def test_wilson_ci_of_an_empty_sample_is_the_whole_interval():
    assert c1.wilson_ci(0, 0) == (0.0, 1.0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_c1_metrics.py -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'confusion'`

- [ ] **Step 3: Implement**

Append to `ragregen/c1.py`:

```python
import math


@dataclass(frozen=True)
class Confusion:
    """Counts with positive = the verifier says FAIL.

    Every ratio returns 0.0 rather than NaN when its denominator vanishes, so
    a degenerate arm produces a comparable table row instead of blowing up the
    report.
    """
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def f1(self) -> float:
        d = self.precision + self.recall
        return 2 * self.precision * self.recall / d if d else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2

    @property
    def mcc(self) -> float:
        num = self.tp * self.tn - self.fp * self.fn
        den = math.sqrt((self.tp + self.fp) * (self.tp + self.fn)
                        * (self.tn + self.fp) * (self.tn + self.fn))
        return num / den if den else 0.0


def confusion(y_true: list[bool], y_pred: list[bool]) -> Confusion:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, "
            f"got {len(y_true)} and {len(y_pred)}")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Used instead of the normal approximation because at n=22 the latter is
    unreliable and can produce bounds outside [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
```

Move the `import math` to the top of the file with the other imports rather than leaving it inline.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_c1_metrics.py -v`
Expected: 11 passed

- [ ] **Step 5: Prove the base-rate test has teeth**

The always-FAIL test is the plan's central guard. Confirm it fails when `balanced_accuracy` is
replaced by plain accuracy — a real mutation, with real pasted output:

```bash
cp ragregen/c1.py /tmp/c1.bak
$PY - <<'EOF'
from pathlib import Path
p = Path("ragregen/c1.py"); s = p.read_text()
s = s.replace("        return (self.recall + self.specificity) / 2",
              "        return self.accuracy  # MUTATION")
p.write_text(s)
EOF
$PY -m pytest tests/test_c1_metrics.py -q
cp /tmp/c1.bak ragregen/c1.py
$PY -m pytest tests/test_c1_metrics.py -q
```

Expected: the mutated run fails `test_always_fail_on_the_real_base_rate_is_exposed_by_balanced_accuracy`
(accuracy would be 0.68, not 0.5); the restored run is green again. Paste both transcripts.

- [ ] **Step 6: Commit**

```bash
git add ragregen/c1.py tests/test_c1_metrics.py
git commit -m "feat: confusion matrix, Wilson intervals, MCC and balanced accuracy"
```

---

## Task 3: The three arms and the `tau` sweep

**Files:**
- Modify: `ragregen/c1.py`
- Create: `tests/test_c1_arms.py`

**Interfaces:**
- Consumes: `c1.confusion`, `c1.Confusion` (Task 2).
- Produces:
  - `c1.state_at_tau(sim: float | None, tau: float) -> str`
  - `c1.grounded_fails(case_scores: dict, tau: float) -> bool`
  - `c1.semantic_fails(case_b: dict) -> bool`
  - `c1.fused_fails(case_scores: dict, case_b: dict, tau: float) -> bool`
  - `c1.abstain_rate(stream_a: dict, tau: float) -> float`
  - `c1.sweep_tau(stream_a, stream_b, case_ids, y_true, grid) -> list[dict]`
  - `c1.TAU_GRID: tuple[float, ...]`

`stream_a` is `{case_id: {phrase: {"sim": float|None, "kind": str, ...}}}`.
`stream_b` is `{case_id: {"ok": bool, "degenerate": bool, ...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c1_arms.py
import pytest

from ragregen import c1


def A(**phrases):
    """Stream A record: phrase -> sim (None means ABSTAIN)."""
    return {p: {"sim": s, "kind": "subject"} for p, s in phrases.items()}


def B(ok, degenerate=False):
    return {"ok": ok, "degenerate": degenerate, "raw": "", "issues": []}


# --- state_at_tau ---------------------------------------------------------

def test_abstain_when_sim_is_none_regardless_of_tau():
    assert c1.state_at_tau(None, 0.01) == "ABSTAIN"
    assert c1.state_at_tau(None, 0.99) == "ABSTAIN"


def test_present_at_or_above_tau_missing_below():
    assert c1.state_at_tau(0.30, 0.25) == "PRESENT"
    assert c1.state_at_tau(0.20, 0.25) == "MISSING"


def test_sim_exactly_equal_to_tau_is_present():
    """Must match `best_sim >= self.tau` at grounded.py:83, which Plan 1 pins
    with its own boundary test. Flipping >= to > here would silently make this
    module disagree with the verifier it is grading."""
    assert c1.state_at_tau(0.25, 0.25) == "PRESENT"


# --- arms -----------------------------------------------------------------

def test_grounded_fails_when_any_concept_is_missing():
    assert c1.grounded_fails(A(fox=0.10, tree=0.90), tau=0.25) is True


def test_grounded_passes_when_all_present():
    assert c1.grounded_fails(A(fox=0.30, tree=0.90), tau=0.25) is False


def test_abstain_alone_never_fails_the_grounded_arm():
    """ABSTAIN != MISSING (parent spec §2). A detector that found nothing is
    not evidence the concept is absent."""
    assert c1.grounded_fails(A(fox=None, tree=None), tau=0.25) is False


def test_grounded_arm_on_an_empty_record_passes():
    assert c1.grounded_fails({}, tau=0.25) is False


def test_semantic_arm_reads_ok():
    assert c1.semantic_fails(B(ok=False)) is True
    assert c1.semantic_fails(B(ok=True)) is False


@pytest.mark.parametrize("g_sim,b_ok,expected", [
    (0.10, False, True),   # both fail
    (0.10, True, True),    # grounded only
    (0.90, False, True),   # semantic only
    (0.90, True, False),   # neither
])
def test_fused_is_the_or_of_both_arms(g_sim, b_ok, expected):
    """fuse() fails if EITHER stream fails (fusion.py). This mirrors it, and
    the OR is precisely why fused recall cannot be evidence for C1."""
    assert c1.fused_fails(A(fox=g_sim), B(ok=b_ok), tau=0.25) is expected


# --- rates ----------------------------------------------------------------

def test_abstain_rate_counts_concepts_not_cases():
    stream_a = {"c1": A(fox=None, tree=0.5), "c2": A(fox=0.4)}
    assert c1.abstain_rate(stream_a, tau=0.25) == pytest.approx(1 / 3)


def test_abstain_rate_of_nothing_is_zero():
    assert c1.abstain_rate({}, tau=0.25) == 0.0


# --- sweep ----------------------------------------------------------------

def test_sweep_returns_one_row_per_tau_with_all_three_arms():
    stream_a = {"a": A(fox=0.10), "b": A(fox=0.90)}
    stream_b = {"a": B(ok=False), "b": B(ok=True)}
    rows = c1.sweep_tau(stream_a, stream_b, ["a", "b"], [True, False],
                        grid=(0.05, 0.5))
    assert [r["tau"] for r in rows] == [0.05, 0.5]
    for r in rows:
        assert set(r["arms"]) == {"grounded", "semantic", "fused"}


def test_sweep_tracks_tau_changing_the_grounded_arm():
    """At tau=0.05 nothing is MISSING; at tau=0.5 case 'a' is. The grounded arm
    must therefore go from useless to perfect across the sweep."""
    stream_a = {"a": A(fox=0.10), "b": A(fox=0.90)}
    stream_b = {"a": B(ok=True), "b": B(ok=True)}
    lo, hi = c1.sweep_tau(stream_a, stream_b, ["a", "b"], [True, False],
                          grid=(0.05, 0.5))
    assert lo["arms"]["grounded"].recall == 0.0
    assert hi["arms"]["grounded"].recall == 1.0
    assert hi["arms"]["grounded"].balanced_accuracy == 1.0


def test_sweep_rejects_a_case_missing_from_a_stream():
    with pytest.raises(KeyError, match="case 'b' missing from stream_b"):
        c1.sweep_tau({"a": A(fox=0.1), "b": A(fox=0.1)}, {"a": B(ok=True)},
                     ["a", "b"], [True, True], grid=(0.25,))


def test_default_grid_spans_the_open_interval():
    assert min(c1.TAU_GRID) > 0.0
    assert max(c1.TAU_GRID) < 1.0
    assert len(c1.TAU_GRID) >= 20
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_c1_arms.py -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'state_at_tau'`

- [ ] **Step 3: Implement**

Append to `ragregen/c1.py`:

```python
#: tau values swept. Open interval: GroundedVerifier rejects tau outside (0, 1).
TAU_GRID: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 100))


def state_at_tau(sim: float | None, tau: float) -> str:
    """Recover a ConceptScore state from a cached similarity.

    ABSTAIN is tau-independent -- it means no box was found, or the concept
    kind is unboxable -- and is recorded as sim=None. The `>=` matches
    grounded.py:83 exactly; diverging would grade the verifier against a rule
    it does not use.
    """
    if sim is None:
        return "ABSTAIN"
    return "PRESENT" if sim >= tau else "MISSING"


def grounded_fails(case_scores: dict, tau: float) -> bool:
    """Stream A fails a case iff some concept is MISSING at this tau."""
    return any(state_at_tau(s.get("sim"), tau) == "MISSING"
               for s in case_scores.values())


def semantic_fails(case_b: dict) -> bool:
    return not case_b["ok"]


def fused_fails(case_scores: dict, case_b: dict, tau: float) -> bool:
    """Mirrors fusion.fuse: ok = not (grounded_failed or semantic_failed)."""
    return grounded_fails(case_scores, tau) or semantic_fails(case_b)


def abstain_rate(stream_a: dict, tau: float) -> float:
    """Fraction of scored concepts that abstained.

    Reported because a high rate means Stream A is inert and `fused` is
    silently just Stream B wearing a second name.
    """
    states = [state_at_tau(s.get("sim"), tau)
              for case in stream_a.values() for s in case.values()]
    if not states:
        return 0.0
    return sum(1 for s in states if s == "ABSTAIN") / len(states)


def sweep_tau(stream_a: dict, stream_b: dict, case_ids: list[str],
              y_true: list[bool], grid=TAU_GRID) -> list[dict]:
    """One row per tau: a Confusion for each arm, plus the abstain rate."""
    for cid in case_ids:
        if cid not in stream_a:
            raise KeyError(f"case '{cid}' missing from stream_a")
        if cid not in stream_b:
            raise KeyError(f"case '{cid}' missing from stream_b")

    rows = []
    for tau in grid:
        preds = {
            "grounded": [grounded_fails(stream_a[c], tau) for c in case_ids],
            "semantic": [semantic_fails(stream_b[c]) for c in case_ids],
            "fused": [fused_fails(stream_a[c], stream_b[c], tau)
                      for c in case_ids],
        }
        rows.append({
            "tau": tau,
            "arms": {name: confusion(y_true, p) for name, p in preds.items()},
            "abstain_rate": abstain_rate(
                {c: stream_a[c] for c in case_ids}, tau),
        })
    return rows
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_c1_arms.py -v`
Expected: 16 passed

- [ ] **Step 5: Prove the boundary test has teeth**

```bash
cp ragregen/c1.py /tmp/c1.bak
$PY - <<'EOF'
from pathlib import Path
p = Path("ragregen/c1.py"); s = p.read_text()
s = s.replace('return "PRESENT" if sim >= tau else "MISSING"',
              'return "PRESENT" if sim > tau else "MISSING"  # MUTATION')
p.write_text(s)
EOF
$PY -m pytest tests/test_c1_arms.py -q
cp /tmp/c1.bak ragregen/c1.py
$PY -m pytest tests/test_c1_arms.py -q
```

Expected: the mutated run fails exactly `test_sim_exactly_equal_to_tau_is_present`; the restored run
is green. Paste both transcripts.

- [ ] **Step 6: Commit**

```bash
git add ragregen/c1.py tests/test_c1_arms.py
git commit -m "feat: grounded/semantic/fused arms and the tau sweep"
```

---

## Task 4: Report rendering

**Files:**
- Modify: `ragregen/c1.py`
- Create: `tests/test_c1_render.py`

**Interfaces:**
- Consumes: `c1.Confusion`, `c1.wilson_ci`, `c1.sweep_tau`, `c1.TAU_GRID`.
- Produces:
  - `c1.baseline_rows(y_true) -> dict[str, Confusion]` — `always-FAIL` and `always-PASS`
  - `c1.best_tau(sweep, arm: str, metric: str = "balanced_accuracy") -> float`
  - `c1.render_report(...) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c1_render.py
from ragregen import c1


def A(**phrases):
    return {p: {"sim": s, "kind": "subject"} for p, s in phrases.items()}


def B(ok, degenerate=False):
    return {"ok": ok, "degenerate": degenerate, "raw": "", "issues": []}


Y_TRUE = [True] * 15 + [False] * 7


def test_baselines_include_both_constants():
    base = c1.baseline_rows(Y_TRUE)
    assert set(base) == {"always-FAIL", "always-PASS"}
    assert base["always-FAIL"].recall == 1.0
    assert base["always-PASS"].recall == 0.0
    assert base["always-FAIL"].balanced_accuracy == 0.5


def test_best_tau_picks_the_maximising_value():
    sweep = c1.sweep_tau({"a": A(fox=0.10), "b": A(fox=0.90)},
                         {"a": B(ok=True), "b": B(ok=True)},
                         ["a", "b"], [True, False], grid=(0.05, 0.5))
    assert c1.best_tau(sweep, "grounded") == 0.5


def test_report_names_every_arm_and_both_baselines():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=0, n_cases=1)
    for token in ("grounded", "semantic", "fused", "always-FAIL", "always-PASS"):
        assert token in md


def test_report_states_the_or_caveat_and_the_tau_fitting_caveat():
    """Both are the difference between a finding and a misleading number, so
    they are asserted rather than left to the writer's discretion."""
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=0, n_cases=1)
    low = md.lower()
    assert "or" in low and "recall" in low
    assert "fitted" in low or "same sample" in low


def test_report_surfaces_a_nonzero_degenerate_count():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False, degenerate=True)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=1, n_cases=1)
    assert "degenerate" in md.lower()
    assert "1" in md


def test_report_reports_exclusions():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict",
                          excluded=3, degenerate=0, n_cases=1)
    assert "3" in md and "exclud" in md.lower()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_c1_render.py -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'baseline_rows'`

- [ ] **Step 3: Implement**

Append to `ragregen/c1.py`:

```python
def baseline_rows(y_true: list[bool]) -> dict[str, Confusion]:
    """The two constant verifiers.

    Reported in every table so no arm is credited for clearing a bar that a
    constant already clears -- at a 68% base rate, always-FAIL scores F1 0.81.
    """
    return {
        "always-FAIL": confusion(y_true, [True] * len(y_true)),
        "always-PASS": confusion(y_true, [False] * len(y_true)),
    }


def best_tau(sweep: list[dict], arm: str,
             metric: str = "balanced_accuracy") -> float:
    """The tau maximising `metric` for `arm`. Ties resolve to the lowest tau."""
    best = max(sweep, key=lambda r: (getattr(r["arms"][arm], metric), -r["tau"]))
    return best["tau"]


def _row(name: str, m: Confusion) -> str:
    lo, hi = wilson_ci(m.tp + m.tn, m.n)          # accuracy interval
    return (f"| {name} | {m.tp} | {m.fp} | {m.tn} | {m.fn} "
            f"| {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} "
            f"| **{m.balanced_accuracy:.3f}** | **{m.mcc:+.3f}** "
            f"| {m.accuracy:.3f} [{lo:.2f}, {hi:.2f}] |")


_HEADER = ("| arm | TP | FP | TN | FN | prec | recall | F1 | bal-acc | MCC "
           "| acc [95% CI] |\n|---|---|---|---|---|---|---|---|---|---|---|")


def render_report(sweep: list[dict], y_true: list[bool], column: str,
                  excluded: int, degenerate: int, n_cases: int) -> str:
    """The operator-facing C1 report. Pure, so it is testable without a GPU."""
    n_fail = sum(y_true)
    n_pass = len(y_true) - n_fail
    base_rate = n_fail / len(y_true) if y_true else 0.0
    tau_star = best_tau(sweep, "fused")
    at_star = next(r for r in sweep if r["tau"] == tau_star)

    L = [f"# C1 verifier validation — `{column}`", "",
         f"{n_cases} cases scored. Ground truth **{n_fail} fail / {n_pass} pass** "
         f"(base rate {base_rate:.0%}). {excluded} excluded for a blank verdict.",
         f"Stream B degenerate replies: **{degenerate}**."
         + (" A non-zero count inflates semantic recall, because Plan 1 fails"
            " closed by design." if degenerate else ""),
         f"Stream A abstain rate at tau\\*: **{at_star['abstain_rate']:.0%}**"
         " (a high rate means the grounded arm is inert and `fused` is just"
         " Stream B).", "",
         f"## Arms at tau\\* = {tau_star}", "", _HEADER]

    for name in ("grounded", "semantic", "fused"):
        L.append(_row(name, at_star["arms"][name]))
    for name, m in baseline_rows(y_true).items():
        L.append(_row(name, m))

    L += ["", "## How to read this", "",
          "- **`semantic` is the single-VLM-judge baseline** C1 is defined against.",
          "- **`fused` is an OR of the two arms**, so its recall is mathematically"
          " >= both. A recall gain is an arithmetic identity, not evidence."
          " Judge C1 on **bal-acc** and **MCC**, which can go down.",
          "- At this base rate a constant `always-FAIL` scores F1"
          f" {baseline_rows(y_true)['always-FAIL'].f1:.2f} while carrying no"
          " information, which is why F1 is not the headline.",
          f"- **tau\\* = {tau_star} was fitted on the same {len(y_true)} cases it"
          " is scored on.** With this sample there is no held-out split worth"
          " making, so treat it as tau\\* *on this sample*, not as a calibrated"
          " value.", "",
          "## tau sweep (fused, balanced accuracy)", "",
          "| tau | grounded | semantic | fused | abstain |",
          "|---|---|---|---|---|"]

    for r in sweep:
        if round(r["tau"] * 100) % 5:      # every 0.05 keeps the table readable
            continue
        L.append(f"| {r['tau']:.2f} | {r['arms']['grounded'].balanced_accuracy:.3f} "
                 f"| {r['arms']['semantic'].balanced_accuracy:.3f} "
                 f"| {r['arms']['fused'].balanced_accuracy:.3f} "
                 f"| {r['abstain_rate']:.0%} |")

    return "\n".join(L)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_c1_render.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole CPU suite**

Run: `$PY -m pytest -q`
Expected: all previous tests still pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add ragregen/c1.py tests/test_c1_render.py
git commit -m "feat: C1 report rendering with constant baselines and caveats"
```

---

## Task 5: `QwenVLM` — the Stream B adapter

**Files:**
- Create: `ragregen/vlm.py`, `tests/test_vlm.py`

**Interfaces:**
- Consumes: `env.HF_CACHE`, Plan 1's `verify.semantic.VLM` protocol (`ask(image, prompt) -> str`).
- Produces: `vlm.QwenVLM(model_id, device, max_new_tokens)` with `.ask(image, prompt) -> str`,
  `vlm.VLM_ID: str`.

**Do not write this from memory.** Plan 1 shipped a green suite that crashed on the first real case
because a fake declared a signature the real pipeline did not have. Step 1 inspects the real API
first, and the fake in Step 3 is built from what Step 1 prints.

- [ ] **Step 1: Inspect the real API before writing anything**

```bash
$PY - <<'EOF'
import inspect
from ragregen import env
env.setup()
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as M
print("processor.__call__:", list(inspect.signature(AutoProcessor.__call__).parameters)[:8])
print("has apply_chat_template:", hasattr(AutoProcessor, "apply_chat_template"))
print("generate params:", list(inspect.signature(M.generate).parameters)[:8])
EOF
```

Record the output in the task report. If `apply_chat_template` is absent, stop and report — the
prompt construction below assumes it.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vlm.py
import pytest
from PIL import Image

from ragregen import vlm

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeProcessor:
    """Mirrors the real processor: apply_chat_template then __call__."""

    def __init__(self):
        self.templated = None
        self.called_with = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        self.templated = messages
        return "TEMPLATED"

    def __call__(self, text=None, images=None, padding=True, return_tensors=None):
        self.called_with = {"text": text, "images": images}
        return FakeInputs()

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["  the reply  "]


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeIds())

    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class FakeIds:
    shape = (1, 7)


class FakeModel:
    def __init__(self):
        self.gen_kwargs = None

    def generate(self, **kw):
        self.gen_kwargs = kw
        return [[0] * 12]


def _vlm(model=None, processor=None, max_new_tokens=64):
    v = object.__new__(vlm.QwenVLM)
    v.device = "cpu"
    v.model = model or FakeModel()
    v.processor = processor or FakeProcessor()
    v.max_new_tokens = max_new_tokens
    return v


def test_ask_returns_the_stripped_reply():
    assert _vlm().ask(IMG, "a fox") == "the reply"


def test_ask_passes_the_image_and_prompt_through_the_chat_template():
    proc = FakeProcessor()
    _vlm(processor=proc).ask(IMG, "an Amur leopard")
    content = proc.templated[0]["content"]
    assert {"type": "image"} in content
    assert any(c.get("text") == "an Amur leopard" for c in content)
    assert proc.called_with["images"] == [IMG]
    assert proc.called_with["text"] == ["TEMPLATED"]


def test_generation_is_deterministic():
    """A judge that samples would give a different verdict on a re-run, which
    would make the whole C1 measurement irreproducible."""
    model = FakeModel()
    _vlm(model=model).ask(IMG, "a fox")
    assert model.gen_kwargs["do_sample"] is False


def test_max_new_tokens_is_forwarded():
    model = FakeModel()
    _vlm(model=model, max_new_tokens=123).ask(IMG, "a fox")
    assert model.gen_kwargs["max_new_tokens"] == 123


def test_vlm_id_is_the_cached_qwen_2_5():
    """Qwen3-VL is deliberately NOT used: it is reserved as the held-out judge
    (parent spec §5), and this checkpoint is already cached."""
    assert vlm.VLM_ID == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_qwen_vlm_satisfies_the_semantic_vlm_protocol():
    from ragregen.verify.semantic import SemanticVerifier
    verdict = SemanticVerifier(_vlm()).judge(IMG, "a fox")
    assert verdict.raw == "the reply"


@pytest.mark.gpu
def test_real_qwen_answers_a_trivial_question():
    v = vlm.QwenVLM(device="cuda")
    reply = v.ask(Image.new("RGB", (64, 64), (255, 0, 0)),
                  "Reply with exactly one word: what colour is this image?")
    assert isinstance(reply, str) and reply.strip()
    assert "red" in reply.lower()
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_vlm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.vlm'`

- [ ] **Step 4: Implement**

```python
# ragregen/vlm.py
"""Qwen2.5-VL adapter for Stream B.

Qwen2.5-VL rather than Qwen3-VL on purpose: this checkpoint is already cached
(16 GB at .cache/models--Qwen--Qwen2.5-VL-7B-Instruct), and Qwen3-VL is held
back so it can serve as the held-out judge the design asks for (parent spec
§5). A judge used inside the loop cannot also grade it.

Generation is greedy. A sampling judge would return a different verdict on a
re-run, and the C1 numbers would not reproduce.
"""
from __future__ import annotations

from ragregen import env

env.setup()  # must precede transformers import

VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLM:
    """Implements verify.semantic.VLM: ask(image, prompt) -> str."""

    def __init__(self, model_id: str = VLM_ID, device: str = "cuda",
                 max_new_tokens: int = 512):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE),
            torch_dtype=torch.bfloat16).to(device).eval()
        self._torch = torch

    def ask(self, image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True,
                                return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=self.max_new_tokens)

        # Strip the prompt tokens; only the continuation is the reply.
        generated = [seq[inputs.input_ids.shape[1]:] for seq in out]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True)[0].strip()
```

Note the `self._torch.no_grad()` wrapper is required — `generate` without it holds the graph and
wastes GPU memory that this card does not have spare.

- [ ] **Step 5: Run the CPU tests**

Run: `$PY -m pytest tests/test_vlm.py -v`
Expected: 6 passed, 1 deselected

- [ ] **Step 6: Run the GPU test when the card is free**

```bash
nvidia-smi
$PY -m pytest tests/test_vlm.py -v -m gpu
```

Expected: 1 passed. Qwen-7B at bf16 needs ~16 GB; if `nvidia-smi` shows more than ~8 GB in use by
others, wait rather than crashing the card. If the real reply's shape differs from the fake's (for
example a list rather than a string), **fix the adapter and the fake together** — the fake exists to
mirror reality.

- [ ] **Step 7: Commit**

```bash
git add ragregen/vlm.py tests/test_vlm.py
git commit -m "feat: Qwen2.5-VL adapter for Stream B, greedy for reproducibility"
```

---

## Task 6: `score-a` — the Stream A sweep

**Files:**
- Create: `scripts/score_a.py`, `tests/test_score_a.py`

**Interfaces:**
- Consumes: `c1.load_labels`, `concepts.parse`, `models.DinoDetector`, `models.build_crop_scorer`,
  `verify.grounded.GroundedVerifier`, `trace.open_run`.
- Produces: `stream_a.json` — `{case_id: {phrase: {sim, box, dino_conf, kind, state}}}`;
  `score_a.score_case(verifier, image, prompt, concept) -> dict`.

`sim` is written unthresholded. `tau` at scoring time is irrelevant because every threshold is
recovered by `c1.state_at_tau`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_a.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.verify.grounded import ConceptScore  # noqa: E402
from scripts import score_a  # noqa: E402

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVerifier:
    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def score(self, image, concepts):
        self.calls.append([c.phrase for c in concepts])
        return self._scores


def test_score_case_serialises_every_field_needed_for_the_tau_sweep():
    scores = {"fox": ConceptScore("fox", "subject", "PRESENT",
                                  box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.8, sim=0.42)}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox")
    assert out["fox"]["sim"] == 0.42
    assert out["fox"]["kind"] == "subject"
    assert out["fox"]["state"] == "PRESENT"
    assert out["fox"]["box"] == [1.0, 2.0, 3.0, 4.0]
    assert out["fox"]["dino_conf"] == 0.8


def test_abstain_is_serialised_with_a_null_sim():
    """sim=None is what makes ABSTAIN recoverable offline at any tau."""
    scores = {"fox": ConceptScore("fox", "subject", "ABSTAIN")}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox")
    assert out["fox"]["sim"] is None
    assert out["fox"]["box"] is None
    assert out["fox"]["state"] == "ABSTAIN"


def test_the_case_concept_is_parsed_as_the_target():
    v = FakeVerifier({})
    score_a.score_case(v, IMG, "an Amur leopard on snow", "Amur leopard")
    assert v.calls[0][0] == "Amur leopard"


def test_output_is_json_serialisable():
    import json
    scores = {"fox": ConceptScore("fox", "subject", "MISSING",
                                  box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.5, sim=0.1)}
    out = score_a.score_case(FakeVerifier(scores), IMG, "a fox", "fox")
    assert json.loads(json.dumps(out)) == out
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_score_a.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_a' from 'scripts'`

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Stream A over every labelled draft (spec §3).

Writes raw similarities, never thresholded verdicts: tau is recovered offline
by c1.state_at_tau, so calibration is arithmetic instead of 99 GPU passes.

Loads GroundingDINO + the crop scorer and nothing else. Qwen must not be in
this process -- see scripts/score_b.py.

Usage: ./scripts/run.sh score-a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, concepts, config, env, models, trace  # noqa: E402
from ragregen.verify.grounded import GroundedVerifier  # noqa: E402

#: Any tau in (0, 1) works: only `sim` is persisted, and state is recomputed.
SCORING_TAU = 0.5


def score_case(verifier, image, prompt: str, concept: str) -> dict:
    parsed = concepts.parse(prompt, target=concept)
    scores = verifier.score(image, parsed)
    return {
        phrase: {
            "sim": s.sim,
            "box": list(s.box) if s.box is not None else None,
            "dino_conf": s.dino_conf,
            "kind": s.kind,
            "state": s.state,
        }
        for phrase, s in scores.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True,
                    help="labels.csv from a screen run")
    ap.add_argument("--out", type=Path, default=None,
                    help="stream_a.json (default: alongside labels.csv)")
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="rescore cases already present in the output")
    args = ap.parse_args()

    rows = c1.load_labels(args.labels)
    out_path = args.out or args.labels.parent / "stream_a.json"
    existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
    if args.force:
        existing = {}

    todo = [r for r in rows if r.case_id not in existing]
    print(f"[score-a] {len(todo)} to score, {len(existing)} cached -> {out_path}")
    if not todo:
        return 0

    for r in todo:
        if not Path(r.draft_path).is_file():
            print(f"[error] case '{r.case_id}': draft not found: {r.draft_path}")
            return 2

    pipe_cfg = config.load_pipeline(args.pipeline)
    run = trace.open_run("score_a", argv=sys.argv, args=vars(args))

    from PIL import Image
    detector = models.DinoDetector(device=args.device)
    scorer = models.build_crop_scorer(pipe_cfg.crop_scorer, device=args.device)
    verifier = GroundedVerifier(detector, scorer, tau=SCORING_TAU)

    for i, r in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {r.case_id}", flush=True)
        image = Image.open(r.draft_path).convert("RGB")
        existing[r.case_id] = score_case(verifier, image, r.prompt, r.concept)
        out_path.write_text(json.dumps(existing, indent=2))

    run.finish("ok", {"n_scored": len(todo), "out": str(out_path)})
    env.reclaim_gpu()
    print(f"[score-a] wrote {len(existing)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The JSON is rewritten after every case, for the same reason `screen_premise.py` writes `labels.csv`
row by row: a contended GPU must cost one case, not the sweep.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_score_a.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/score_a.py
git add scripts/score_a.py tests/test_score_a.py
git commit -m "feat: Stream A sweep persisting raw similarities"
```

---

## Task 7: `score-b` — the Stream B sweep

**Files:**
- Create: `scripts/score_b.py`, `tests/test_score_b.py`

**Interfaces:**
- Consumes: `c1.load_labels`, `vlm.QwenVLM`, `verify.semantic.SemanticVerifier`, `trace.open_run`.
- Produces: `stream_b.json` — `{case_id: {ok, degenerate, raw, issues[]}}`;
  `score_b.score_case(verifier, image, prompt) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_b.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.verify.semantic import Issue, SemanticVerdict  # noqa: E402
from scripts import score_b  # noqa: E402

IMG = Image.new("RGB", (32, 32), (0, 0, 0))


class FakeVerifier:
    def __init__(self, verdict):
        self._verdict = verdict
        self.prompts = []

    def judge(self, image, prompt):
        self.prompts.append(prompt)
        return self._verdict


def test_score_case_serialises_a_clean_pass():
    v = SemanticVerdict(ok=True, raw='{"verdict": "PASS"}')
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert out["ok"] is True
    assert out["degenerate"] is False
    assert out["issues"] == []
    assert out["raw"] == '{"verdict": "PASS"}'


def test_score_case_serialises_issues():
    v = SemanticVerdict(ok=False, raw="{}",
                        issues=[Issue("Amur leopard", "rosettes wrong")])
    out = score_b.score_case(FakeVerifier(v), IMG, "an Amur leopard")
    assert out["issues"] == [{"concept": "Amur leopard",
                              "problem": "rosettes wrong"}]


def test_degenerate_is_preserved_not_flattened_into_ok():
    """A degenerate reply fails closed, so it looks identical to a real FAIL
    unless the flag survives. The C1 report needs the count to explain an
    inflated semantic recall."""
    v = SemanticVerdict(ok=False, raw="!!!!", degenerate=True)
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert out["ok"] is False
    assert out["degenerate"] is True
    assert out["raw"] == "!!!!"


def test_the_case_prompt_is_what_gets_judged():
    v = SemanticVerdict(ok=True, raw="{}")
    fake = FakeVerifier(v)
    score_b.score_case(fake, IMG, "a durian on a market table")
    assert fake.prompts == ["a durian on a market table"]


def test_output_is_json_serialisable():
    import json
    v = SemanticVerdict(ok=False, raw="x", issues=[Issue("a", "b")])
    out = score_b.score_case(FakeVerifier(v), IMG, "a fox")
    assert json.loads(json.dumps(out)) == out
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_score_b.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_b' from 'scripts'`

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Stream B over every labelled draft (spec §3).

Loads Qwen2.5-VL and nothing else. GroundingDINO and the crop scorer must not
be in this process: Qwen-7B is ~16 GB, the card is 24 GB, and other researchers
routinely hold ~9 GB of it.

Usage: ./scripts/run.sh score-b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1, env, trace, vlm  # noqa: E402
from ragregen.verify.semantic import SemanticVerifier  # noqa: E402


def score_case(verifier, image, prompt: str) -> dict:
    v = verifier.judge(image, prompt)
    return {
        "ok": v.ok,
        "degenerate": v.degenerate,
        "raw": v.raw,
        "issues": [{"concept": i.concept, "problem": i.problem}
                   for i in v.issues],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = c1.load_labels(args.labels)
    out_path = args.out or args.labels.parent / "stream_b.json"
    existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
    if args.force:
        existing = {}

    todo = [r for r in rows if r.case_id not in existing]
    print(f"[score-b] {len(todo)} to score, {len(existing)} cached -> {out_path}")
    if not todo:
        return 0

    for r in todo:
        if not Path(r.draft_path).is_file():
            print(f"[error] case '{r.case_id}': draft not found: {r.draft_path}")
            return 2

    run = trace.open_run("score_b", argv=sys.argv, args=vars(args))

    from PIL import Image
    verifier = SemanticVerifier(vlm.QwenVLM(device=args.device))

    n_degenerate = 0
    for i, r in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {r.case_id}", flush=True)
        image = Image.open(r.draft_path).convert("RGB")
        rec = score_case(verifier, image, r.prompt)
        n_degenerate += bool(rec["degenerate"])
        existing[r.case_id] = rec
        out_path.write_text(json.dumps(existing, indent=2))

    run.finish("ok", {"n_scored": len(todo), "n_degenerate": n_degenerate})
    env.reclaim_gpu()
    print(f"[score-b] wrote {len(existing)} cases, "
          f"{n_degenerate} degenerate -> {out_path}")
    if n_degenerate:
        print("[score-b] WARNING: degenerate replies fail closed, which "
              "inflates semantic recall. The C1 report states the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_score_b.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/score_b.py
git add scripts/score_b.py tests/test_score_b.py
git commit -m "feat: Stream B sweep preserving degenerate replies"
```

---

## Task 8: `c1` — the report stage and `run.sh` wiring

**Files:**
- Create: `scripts/c1_report.py`, `tests/test_c1_report_cli.py`
- Modify: `scripts/run.sh`

**Interfaces:**
- Consumes: everything above.
- Produces: `./scripts/run.sh {score-a|score-b|c1}`; `c1.md` and `c1.json` in the run directory.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_c1_report_cli.py
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import c1_report  # noqa: E402

HEADER = "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "labels.csv").write_text(
        HEADER
        + "a,a prompt,fox,/d/a.png,fail,fail,\n"
        + "b,b prompt,fox,/d/b.png,pass,pass,\n")
    (tmp_path / "stream_a.json").write_text(json.dumps({
        "a": {"fox": {"sim": 0.10, "kind": "subject", "box": None,
                      "dino_conf": None, "state": "MISSING"}},
        "b": {"fox": {"sim": 0.90, "kind": "subject", "box": None,
                      "dino_conf": None, "state": "PRESENT"}}}))
    (tmp_path / "stream_b.json").write_text(json.dumps({
        "a": {"ok": False, "degenerate": False, "raw": "", "issues": []},
        "b": {"ok": True, "degenerate": False, "raw": "", "issues": []}}))
    return tmp_path


def test_report_runs_and_writes_both_artifacts(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 0
    assert (run_dir / "c1.md").is_file()
    assert (run_dir / "c1.json").is_file()


def test_report_covers_both_ground_truth_columns(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    c1_report.main()
    md = (run_dir / "c1.md").read_text()
    assert "verdict_identity" in md and "verdict`" in md


def test_missing_stream_a_exits_two_with_an_actionable_message(
        run_dir, monkeypatch, capsys):
    (run_dir / "stream_a.json").unlink()
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 2
    out = capsys.readouterr().out
    assert "stream_a.json" in out and "score-a" in out


def test_missing_stream_b_exits_two(run_dir, monkeypatch, capsys):
    (run_dir / "stream_b.json").unlink()
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 2
    assert "score-b" in capsys.readouterr().out


def test_json_artifact_carries_the_arms_at_tau_star(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    c1_report.main()
    data = json.loads((run_dir / "c1.json").read_text())
    assert "verdict_identity" in data
    arms = data["verdict_identity"]["arms_at_tau_star"]
    assert set(arms) == {"grounded", "semantic", "fused"}
    assert "balanced_accuracy" in arms["fused"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY -m pytest tests/test_c1_report_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'c1_report' from 'scripts'`

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Fuse the two stages, sweep tau, render the C1 report (spec §5).

No models: this reads the JSON the two GPU stages wrote. Re-running after a
label is revised costs seconds, which is the whole reason Stream A persists
raw similarities.

Usage: ./scripts/run.sh c1 --run outputs/screen_<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1  # noqa: E402

COLUMNS = ("verdict_identity", "verdict")


def _confusion_dict(m) -> dict:
    return {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "balanced_accuracy": m.balanced_accuracy, "mcc": m.mcc,
            "accuracy": m.accuracy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="a screen run directory holding labels.csv")
    args = ap.parse_args()

    labels = args.run / "labels.csv"
    a_path = args.run / "stream_a.json"
    b_path = args.run / "stream_b.json"

    for path, stage in ((labels, None), (a_path, "score-a"), (b_path, "score-b")):
        if not path.is_file():
            hint = f" Run `./scripts/run.sh {stage}` first." if stage else ""
            print(f"[error] missing {path.name} in {args.run}.{hint}")
            return 2

    rows = c1.load_labels(labels)
    stream_a = json.loads(a_path.read_text())
    stream_b = json.loads(b_path.read_text())
    degenerate = sum(1 for v in stream_b.values() if v.get("degenerate"))

    sections, artifact = [], {}
    for column in COLUMNS:
        kept, y_true, excluded = c1.ground_truth(rows, column)
        case_ids = [r.case_id for r in kept]
        sweep = c1.sweep_tau(stream_a, stream_b, case_ids, y_true)
        sections.append(c1.render_report(
            sweep=sweep, y_true=y_true, column=column, excluded=excluded,
            degenerate=degenerate, n_cases=len(case_ids)))

        tau_star = c1.best_tau(sweep, "fused")
        at_star = next(r for r in sweep if r["tau"] == tau_star)
        artifact[column] = {
            "tau_star": tau_star,
            "n_fail": sum(y_true),
            "n_pass": len(y_true) - sum(y_true),
            "excluded": excluded,
            "degenerate": degenerate,
            "abstain_rate": at_star["abstain_rate"],
            "arms_at_tau_star": {k: _confusion_dict(v)
                                 for k, v in at_star["arms"].items()},
            "baselines": {k: _confusion_dict(v)
                          for k, v in c1.baseline_rows(y_true).items()},
        }

    md = "\n\n---\n\n".join(sections)
    (args.run / "c1.md").write_text(md)
    (args.run / "c1.json").write_text(json.dumps(artifact, indent=2))
    print(md)
    print(f"\n[c1] wrote {args.run / 'c1.md'} and {args.run / 'c1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `$PY -m pytest tests/test_c1_report_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Wire the three stages into `run.sh`**

In the `usage()` heredoc, add these lines after the `bakeoff` line:

```
  score-a      Stream A over labelled drafts  (GPU, ~5 min)
  score-b      Stream B over labelled drafts  (GPU, ~10 min)
  c1           fuse, sweep tau, report        (no GPU, seconds)
```

and in the `case` statement, add these three branches before `-h|--help`:

```bash
  score-a)     exec "$PY" scripts/score_a.py "$@" ;;
  score-b)     exec "$PY" scripts/score_b.py "$@" ;;
  c1)          exec "$PY" scripts/c1_report.py "$@" ;;
```

- [ ] **Step 6: Check dispatch**

```bash
./scripts/run.sh --help
./scripts/run.sh c1 --run /nonexistent; echo "exit=$?"
```

Expected: help lists the three new stages; the second prints `[error] missing labels.csv` and
`exit=2`.

- [ ] **Step 7: Run the whole CPU suite**

Run: `$PY -m pytest -q`
Expected: green, no regressions.

- [ ] **Step 8: Commit**

```bash
git add scripts/c1_report.py scripts/run.sh tests/test_c1_report_cli.py
git commit -m "feat: C1 report stage and run.sh wiring"
```

---

## Task 9: Run it and report the finding

**Files:** none created. This task produces the measurement.

- [ ] **Step 1: Confirm the card is free enough**

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Stream A needs ~3 GB; Stream B needs ~16 GB. If others hold more than ~8 GB, wait before `score-b`.

- [ ] **Step 2: Stream A**

```bash
./scripts/run.sh score-a --labels outputs/screen_20260726_233320/labels.csv
```

Expected: 22 cases scored, `stream_a.json` written. Roughly 5 minutes.

- [ ] **Step 3: Stream B**

```bash
nvidia-smi
./scripts/run.sh score-b --labels outputs/screen_20260726_233320/labels.csv
```

Expected: 22 cases scored, `stream_b.json` written, degenerate count printed. Roughly 10 minutes.
If the degenerate count is not 0, inspect a raw reply before continuing:

```bash
$PY -c "
import json
d = json.load(open('outputs/screen_20260726_233320/stream_b.json'))
for k, v in d.items():
    if v['degenerate']: print(k, repr(v['raw'][:300])); break
"
```

A high degenerate count means the judge template needs work, and the C1 numbers are not meaningful
until it is fixed. Stop and report rather than reporting inflated semantic recall.

- [ ] **Step 4: The report**

```bash
./scripts/run.sh c1 --run outputs/screen_20260726_233320
```

- [ ] **Step 5: Sanity-check the numbers before believing them**

Confirm each of these, and report any that fail:

- `fused` recall >= both `grounded` and `semantic` recall at every tau. If not, `fused_fails`
  disagrees with `fusion.fuse` and one of them is wrong.
- The abstain rate is below ~50%. Above that, the grounded arm is inert and `fused` is just
  Stream B — say so plainly in the finding.
- `always-FAIL` shows balanced accuracy 0.500 and MCC 0.000 exactly.
- Every arm's `n` equals 22 for `verdict_identity` and 22 for `verdict`.

- [ ] **Step 6: Report the finding**

Write the answer to the question this plan exists to ask, in plain terms:

1. Does `fused` beat `semantic` on **balanced accuracy and MCC** — and do their Wilson intervals
   overlap? At n=22 they very likely do, and if so the honest finding is **"not separated at this
   sample size,"** not a winner.
2. Does any arm beat `always-FAIL` on balanced accuracy? An arm that does not is worthless
   regardless of its F1.
3. What is tau\*, and how flat is the curve around it? A flat curve means tau barely matters; a
   sharp peak fitted on 22 cases means it is fitted to noise.
4. What does the `agreement` split say about how often the streams disagree — the parent spec's
   stated evidence (§2) that decomposition buys anything?

- [ ] **Step 7: Commit the finding**

```bash
git add -A docs
git commit -m "docs: C1 validation result on the 22 labelled drafts"
```

Note `outputs/` is gitignored, so `c1.md` stays with the run. Quote its headline numbers in the
commit message so the result is in git history even though the artifact is not.

---

## Self-Review

**Spec coverage.** §2 ground truths → Task 1. §3 architecture and the three stages → Tasks 6, 7, 8.
§3 "why scores are cached" → Task 6 Step 3 (`SCORING_TAU`, raw `sim`). §4 components → Tasks 1–8
(`c1.py` across 1–4, `vlm.py` in 5, the three scripts in 6–8). §5 arms, base-rate baselines, Wilson,
MCC, degenerate and abstain rates, tau honesty → Tasks 2, 3, 4. §6 error handling → Task 1
(blank/typo verdicts), 6 and 7 (missing drafts, per-case resume), 8 (missing stage files). §7 testing
→ every task's TDD cycle, with explicit mutation steps in Tasks 2 and 3 and the
`inspect.signature`-first rule in Task 5. §8 environment → Global Constraints. §9 risks R1–R8 →
Task 4's report text and Task 9 Steps 5–6. §10 success criteria → Task 9.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step
carries runnable code; every test step carries real assertions.

**Type consistency.** `ConceptScore(phrase, kind, state, box, dino_conf, sim)` is consumed in Task 6
exactly as Plan 1 defines it. `SemanticVerdict(ok, raw, degenerate, issues)` field order matches
Plan 1's dataclass (defaults last). `Confusion` is produced in Task 2 and consumed by name in 3 and
4. `c1.ground_truth` returns `(kept, flags, excluded)` in Task 1 and is unpacked that way in Task 8.
`stream_a` / `stream_b` dict shapes are identical in Tasks 3, 6, 7 and 8. `state_at_tau` uses `>=`
in Task 3, matching `grounded.py:83` as the Global Constraints require.

**Known soft spot, flagged rather than hidden.** Task 5's `QwenVLM.ask` is the one place written
against an API this plan has not executed. Step 1 inspects the real signature before implementing,
and Step 6 runs it on the GPU; if reality differs, the instruction is to fix the adapter *and* the
fake together. That is the exact failure mode that cost Plan 1 a crash on its first real case.
