"""Metrics for the C1 verifier validation.

Pure: no models, and no I/O beyond reading the labels CSV and the two stage
JSONs. That is what makes the arms comparison testable on CPU with fixtures,
the same split that made Plan 1's fusion.py verifiable without a GPU.
"""
from __future__ import annotations

import csv
import math
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


def resolve_draft(row: LabelRow, labels_path: Path) -> Path:
    """Resolve legacy labels whose absolute worktree path no longer exists."""
    recorded = Path(row.draft_path)
    if recorded.is_file():
        return recorded
    local = Path(labels_path).parent / row.case_id / "draft.png"
    return local if local.is_file() else recorded


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


#: tau values swept. Open interval: GroundedVerifier rejects tau outside (0, 1).
TAU_GRID: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 100))

#: tau is pinned here for all fine-grained reporting. C1 showed tau* was fitted
#: to noise -- a two-cell peak at the edge of the grid -- so delta is the only
#: knob selected on this data (fine-grained spec §5).
BASELINE_TAU = 0.25

#: delta values swept: -0.20 to +0.20 in steps of 0.005. Negative delta fails
#: only when the coarse term strictly beats the fine term by a margin; zero
#: fails when coarse >= fine; positive delta is more aggressive.
DELTA_GRID: tuple[float, ...] = tuple(round(0.005 * i, 3) for i in range(-40, 41))

#: delta value that disables the contrastive test entirely. Similarities are in
#: [0, 1], so the margin is >= -1 and a strict `<` can never fire. NOT 0.0 --
#: delta = 0.0 fails whenever the coarse term merely ties or beats the fine one,
#: which is an aggressive point on the grid, not the mechanism switched off.
DELTA_OFF = -1.0

#: Grounded states that fail a case. Mirrors fusion.FAILING_STATES; diverging
#: would grade the verifier against a rule it does not use.
FAILING_STATES = ("MISSING", "FINE_MISMATCH")


def state_at_margin(sim: float | None, margin: float | None,
                    tau: float, delta: float) -> str:
    """Recover a state from a similarity and an explicit contrastive margin.

    Which margin -- text or prototype -- is the caller's choice. MISSING is
    always decided by `sim`, the text similarity, and always takes precedence:
    that is what keeps the pinned tau = 0.25 baseline comparable across both
    mechanisms.
    """
    if sim is None:
        return "ABSTAIN"
    if sim < tau:
        return "MISSING"
    # round() mirrors grounded.py exactly. Without it, 0.60 - 0.50 is
    # 0.09999999999999998 and a margin that should sit exactly on delta falls
    # below it. The two implementations must agree bit for bit or the report
    # grades the verifier against a rule it does not use.
    if margin is not None and round(margin, 9) < delta:
        return "FINE_MISMATCH"
    return "PRESENT"


def state_at(sim: float | None, sim_coarse: float | None,
             tau: float, delta: float) -> str:
    """The text mechanism. Mirrors GroundedVerifier._score_one exactly."""
    margin = None if (sim is None or sim_coarse is None) else sim - sim_coarse
    return state_at_margin(sim, margin, tau, delta)


def text_margin(concept: dict) -> float | None:
    """sim - sim_coarse, or None when either side is absent."""
    sim, coarse = concept.get("sim"), concept.get("sim_coarse")
    return None if (sim is None or coarse is None) else sim - coarse


def proto_margin(concept: dict) -> float | None:
    """sim_proto - sim_proto_coarse at the HIGHEST-CONFIDENCE candidate box.

    Deliberately not the top-level sim_proto. That belongs to the box chosen by
    argmax of the fine-phrase similarity, so the selection has already favoured
    the fine side before the margin is taken -- asymmetry (b) in
    findings/2026-07-27-finegrained-result.md §3b, which prototype scoring does
    NOT remove on its own. Detector confidence is neutral between the fine and
    coarse terms, so both sides are selected the same way and then measured the
    same way.

    This is the pre-registered graded rule. Other selection rules may be
    reported as diagnostics; none may replace this one.
    """
    scored = [c for c in (concept.get("candidates") or [])
              if c.get("sim_proto") is not None
              and c.get("sim_proto_coarse") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda c: c["dino_conf"])
    return best["sim_proto"] - best["sim_proto_coarse"]


def grounded_fails(case_scores: dict, tau: float, delta: float = DELTA_OFF,
                   margin_of=text_margin) -> bool:
    """Stream A fails a case iff some concept is MISSING or FINE_MISMATCH."""
    return any(
        state_at_margin(s.get("sim"), margin_of(s), tau, delta) in FAILING_STATES
        for s in case_scores.values()
    )


def semantic_fails(case_b: dict) -> bool:
    return not case_b["ok"]


def fused_fails(case_scores: dict, case_b: dict, tau: float,
                delta: float = DELTA_OFF, margin_of=text_margin) -> bool:
    """Mirrors fusion.fuse: ok = not (grounded_failed or semantic_failed)."""
    return (grounded_fails(case_scores, tau, delta, margin_of)
            or semantic_fails(case_b))


def abstain_rate(stream_a: dict, tau: float, delta: float = DELTA_OFF) -> float:
    """Fraction of scored concepts that abstained.

    Reported because a high rate means Stream A is inert and `fused` is
    silently just Stream B wearing a second name. Independent of delta and of
    the margin source; the parameter exists so callers can pass a sweep row
    through uniformly.
    """
    states = [state_at_margin(s.get("sim"), text_margin(s), tau, delta)
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


def sweep_delta(stream_a: dict, stream_b: dict, case_ids: list[str],
                y_true: list[bool], tau: float = BASELINE_TAU,
                grid=DELTA_GRID, margin_of=text_margin) -> list[dict]:
    """One row per delta at a pinned tau. Row shape mirrors sweep_tau."""
    for cid in case_ids:
        if cid not in stream_a:
            raise KeyError(f"case '{cid}' missing from stream_a")
        if cid not in stream_b:
            raise KeyError(f"case '{cid}' missing from stream_b")

    rows = []
    for delta in grid:
        preds = {
            "grounded": [grounded_fails(stream_a[c], tau, delta, margin_of)
                         for c in case_ids],
            "semantic": [semantic_fails(stream_b[c]) for c in case_ids],
            "fused": [fused_fails(stream_a[c], stream_b[c], tau, delta, margin_of)
                      for c in case_ids],
        }
        rows.append({
            "delta": delta,
            "arms": {name: confusion(y_true, p) for name, p in preds.items()},
            "abstain_rate": abstain_rate(
                {c: stream_a[c] for c in case_ids}, tau, delta),
        })
    return rows


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
          "## tau sweep (balanced accuracy)", "",
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


def baseline_split(stream_a: dict, stream_b: dict, case_ids: list[str],
                   y_true: list[bool], tau: float = BASELINE_TAU,
                   margin_of=text_margin) -> tuple[list[str], list[str]]:
    """Which true failures the fused arm catches with delta disabled.

    This is the pre-registered baseline the decision rule is graded against.
    It is recomputed rather than taken from the published C1 numbers, which
    were reported at tau* = 0.02 and may not have the same membership here
    (fine-grained spec §5).
    """
    caught, missed = [], []
    for cid, is_fail in zip(case_ids, y_true):
        if not is_fail:
            continue
        if fused_fails(stream_a[cid], stream_b[cid], tau, DELTA_OFF, margin_of):
            caught.append(cid)
        else:
            missed.append(cid)
    return caught, missed


def baseline_false_positives(stream_a: dict, stream_b: dict,
                             case_ids: list[str], y_true: list[bool],
                             tau: float = BASELINE_TAU,
                             margin_of=text_margin) -> list[str]:
    """Correct images the delta-disabled baseline already fails.

    tau = 0.25 imports two of these on the identity column (stack_of_books,
    sushi), both flagged MISSING by the threshold rather than by any
    contrastive rule. Requiring zero false positives absolutely made the
    pre-registered criterion unsatisfiable before delta was ever swept, so the
    rule is graded against *new* false positives instead.
    """
    return [cid for cid, is_fail in zip(case_ids, y_true)
            if not is_fail
            and fused_fails(stream_a[cid], stream_b[cid], tau, DELTA_OFF,
                            margin_of)]


@dataclass(frozen=True)
class RuleResult:
    delta: float
    caught: list[str]
    false_positives: list[str]
    gross_retained: bool
    passes: bool


def evaluate_rule(stream_a: dict, stream_b: dict, case_ids: list[str],
                  y_true: list[bool], baseline_caught: list[str],
                  baseline_missed: list[str], delta: float,
                  tau: float = BASELINE_TAU, min_catch: int = 4,
                  baseline_false_positives=(),
                  margin_of=text_margin) -> RuleResult:
    """The three pre-registered criteria at one delta (fine-grained spec §5).

    `baseline_false_positives` defaults to empty, which reproduces the original
    zero-FP criterion exactly.
    """
    fails = {cid: fused_fails(stream_a[cid], stream_b[cid], tau, delta, margin_of)
             for cid in case_ids}
    inherited = set(baseline_false_positives)

    caught = [cid for cid in baseline_missed if fails[cid]]
    false_positives = [cid for cid, is_fail in zip(case_ids, y_true)
                       if not is_fail and fails[cid] and cid not in inherited]
    gross_retained = all(fails[cid] for cid in baseline_caught)

    passes = (len(caught) >= min_catch
              and not false_positives
              and gross_retained)
    return RuleResult(delta=delta, caught=caught,
                      false_positives=false_positives,
                      gross_retained=gross_retained, passes=passes)


def robust_band(results: list[RuleResult],
                min_consecutive: int = 3) -> list[RuleResult]:
    """The longest run of consecutive passing deltas, if it is long enough.

    A result surviving at exactly one grid point is the tau* mistake wearing a
    different letter, so a lone pass is reported as no band at all.
    """
    best: list[RuleResult] = []
    run: list[RuleResult] = []
    for r in results:
        run = run + [r] if r.passes else []
        if len(run) > len(best):
            best = run
    return best if len(best) >= min_consecutive else []


def render_finegrained(stream_a: dict, stream_b: dict, case_ids: list[str],
                       y_true: list[bool], column: str,
                       tau: float = BASELINE_TAU) -> tuple[str, dict]:
    """The fine-grained section: baseline, mechanism split, rule verdict."""
    caught0, missed0 = baseline_split(stream_a, stream_b, case_ids, y_true, tau)
    results = [evaluate_rule(stream_a, stream_b, case_ids, y_true,
                             caught0, missed0, delta=d, tau=tau)
               for d in DELTA_GRID]
    band = robust_band(results)
    sweep = sweep_delta(stream_a, stream_b, case_ids, y_true, tau=tau)

    met = bool(band)
    chosen = band[len(band) // 2] if met else None

    L = [f"## Fine-grained verification — `{column}`", "",
         f"tau pinned at **{tau}** (not fitted). delta swept over "
         f"{len(DELTA_GRID)} values from {DELTA_GRID[0]} to {DELTA_GRID[-1]}.", "",
         f"**Baseline at tau = {tau}, delta disabled:** "
         f"{len(caught0)} caught, {len(missed0)} missed.", "",
         f"- caught: {', '.join(caught0) if caught0 else '(none)'}",
         f"- missed: {', '.join(missed0) if missed0 else '(none)'}", ""]

    if met:
        L += [f"### RULE MET across delta {band[0].delta} to {band[-1].delta} "
              f"({len(band)} consecutive values)", "",
              f"At delta = {chosen.delta}: **{len(chosen.caught)} of "
              f"{len(missed0)}** previously-missed cases now caught "
              f"({', '.join(chosen.caught) if chosen.caught else 'none'}), "
              f"**{len(chosen.false_positives)} false positives**, "
              f"gross-category catches retained: "
              f"**{'yes' if chosen.gross_retained else 'NO'}**.", ""]
    else:
        best = max(results, key=lambda r: len(r.caught))
        # When every delta catches nothing, max() returns the first grid point
        # and naming it would dress a tie-break artifact up as a finding.
        if best.caught:
            detail = (f"The best single delta was {best.delta}, catching "
                      f"{len(best.caught)} of {len(missed0)} with "
                      f"{len(best.false_positives)} false positives.")
        else:
            detail = (f"No delta caught any of the {len(missed0)} "
                      "previously-missed cases.")
        L += ["### RULE NOT MET", "",
              "No band of 3 or more consecutive delta values satisfies all "
              f"three criteria. {detail} "
              "Per spec §5 the conclusion is that phrase-level contrast is "
              "insufficient, not that delta needs more tuning.", ""]

    L += ["### Fused arm across delta", "", "| delta | prec | recall | bal-acc | MCC |",
          "|---|---|---|---|---|"]
    for r in sweep:
        # Every 0.02, via integer arithmetic. Float modulo on this grid drops
        # -0.2, -0.14, -0.1 and three others to representation error, silently
        # printing 15 rows where 21 are meant -- verified, not hypothetical.
        if round(r["delta"] * 1000) % 20 != 0:
            continue
        m = r["arms"]["fused"]
        L.append(f"| {r['delta']} | {m.precision:.3f} | {m.recall:.3f} | "
                 f"{m.balanced_accuracy:.3f} | {m.mcc:+.3f} |")

    artifact = {
        "tau": tau,
        "baseline_caught": caught0,
        "baseline_missed": missed0,
        "rule_met": met,
        "band": [r.delta for r in band],
        "chosen_delta": chosen.delta if met else None,
        "caught_at_chosen": chosen.caught if met else [],
        "false_positives_at_chosen": chosen.false_positives if met else [],
    }
    return "\n".join(L), artifact


def _has_prototypes(stream_a: dict) -> bool:
    return any(proto_margin(s) is not None
               for case in stream_a.values() for s in case.values())


def render_prototype(stream_a: dict, stream_b: dict, case_ids: list[str],
                     y_true: list[bool], column: str,
                     tau: float = BASELINE_TAU) -> tuple[str, dict]:
    """The prototype section: same rule machinery, image-side margin.

    Returns empty output when nothing carries prototype scores, so a run
    without --proto-arm is byte-for-byte what it was before.
    """
    if not _has_prototypes(stream_a):
        return "", {}

    caught0, missed0 = baseline_split(stream_a, stream_b, case_ids, y_true, tau)
    inherited = baseline_false_positives(stream_a, stream_b, case_ids, y_true, tau)
    results = [evaluate_rule(stream_a, stream_b, case_ids, y_true,
                             caught0, missed0, delta=d, tau=tau,
                             baseline_false_positives=inherited,
                             margin_of=proto_margin)
               for d in DELTA_GRID]
    band = robust_band(results)
    met = bool(band)
    chosen = band[len(band) // 2] if met else None

    L = [f"## Prototype verification — `{column}`", "",
         f"tau pinned at **{tau}** (not fitted). delta_proto swept over "
         f"{len(DELTA_GRID)} values from {DELTA_GRID[0]} to {DELTA_GRID[-1]}.", "",
         f"**Baseline at tau = {tau}, delta disabled:** "
         f"{len(caught0)} caught, {len(missed0)} missed.", "",
         f"- caught: {', '.join(caught0) if caught0 else '(none)'}",
         f"- missed: {', '.join(missed0) if missed0 else '(none)'}",
         f"- false positives inherited from the threshold: "
         f"{', '.join(inherited) if inherited else '(none)'}", ""]

    if met:
        L += [f"### RULE MET across delta_proto {band[0].delta} to "
              f"{band[-1].delta} ({len(band)} consecutive values)", "",
              f"At delta_proto = {chosen.delta}: **{len(chosen.caught)} of "
              f"{len(missed0)}** previously-missed cases now caught "
              f"({', '.join(chosen.caught) if chosen.caught else 'none'}), "
              f"**{len(chosen.false_positives)} new false positives**, "
              f"gross-category catches retained: "
              f"**{'yes' if chosen.gross_retained else 'NO'}**.", ""]
    else:
        best = max(results, key=lambda r: len(r.caught))
        L += ["### RULE NOT MET", "",
              "No band of 3 or more consecutive delta_proto values satisfies "
              "all three criteria. The best single delta_proto was "
              f"{best.delta}, catching {len(best.caught)} of {len(missed0)} "
              f"with {len(best.false_positives)} new false positives. "
              "Per the spec the conclusion is that image-side contrast is "
              "insufficient, not that delta_proto needs more tuning.", ""]

    artifact = {
        "mechanism": "prototype",
        "tau": tau,
        "baseline_caught": caught0,
        "baseline_missed": missed0,
        "inherited_false_positives": inherited,
        "rule_met": met,
        "band": [r.delta for r in band],
        "chosen_delta": chosen.delta if met else None,
        "caught_at_chosen": chosen.caught if met else [],
        "new_false_positives_at_chosen": chosen.false_positives if met else [],
    }
    return "\n".join(L), artifact
