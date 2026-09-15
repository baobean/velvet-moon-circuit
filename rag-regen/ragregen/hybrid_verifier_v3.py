"""v3 selector: gate repair-attempt eligibility on candidate semantic
confirmation (design Option A).

The v2 selector routed on two reranker margins (text, reference relevance) whose
agreement with ground-truth DINO was a coin flip (sign accuracy 0.60). v3 drops
the weak reference margin and instead requires an *independent* identity
judgement -- the plain semantic verifier affirming the concept on the attempt --
keeping only a light text-relevance floor to reject regressions:

    eligible(attempt) = candidate_semantic_ok(attempt)
                        AND attempt_text - draft_text >= text_margin

Among eligible attempts, select by (text_relevance, -attempt_number); else keep
the draft. The detector is unchanged from v2 and is reused directly. DINO remains
a fit/evaluation signal only, never a live input.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragregen import hybrid_v2_develop as h2
from ragregen import hybrid_verifier_v2 as v2


@dataclass(frozen=True)
class SelectorFitV3:
    text_margin: float
    feasible: bool
    sign_accuracy: float
    harmful: int
    mean_dino_delta: float
    exact_best_rate: float


def _eligible(scores: Mapping, label: str, semantic_ok: Mapping,
              text_margin: float) -> bool:
    draft = scores["draft"]
    text_gain = (v2._number(scores[label], label, "text_relevance")
                 - v2._number(draft, "draft", "text_relevance"))
    return bool(semantic_ok.get(label)) and text_gain + v2._ULP >= text_margin


def select_repair(scores: Mapping, semantic_ok: Mapping,
                  text_margin: float) -> str:
    if "draft" not in scores:
        raise ValueError("scores are missing draft")
    v2._number(scores["draft"], "draft", "text_relevance")
    eligible = []
    for label, record in scores.items():
        if label == "draft":
            continue
        attempt = v2._attempt_number(label)
        text = v2._number(record, label, "text_relevance")
        if _eligible(scores, label, semantic_ok, text_margin):
            eligible.append((text, -attempt, label))
    return max(eligible)[-1] if eligible else "draft"


def _metrics(rows: Sequence[Mapping], text_margin: float):
    pair_correct, deltas, exact = [], [], []
    for row in rows:
        scores, dino, sem = row["scores"], row["dino"], row["semantic"]
        selected = select_repair(scores, sem, text_margin)
        draft_dino = v2._finite(dino["draft"], "dino draft")
        deltas.append(v2._finite(dino[selected], "dino") - draft_dino)
        for label in scores:
            if label == "draft":
                continue
            predicted = _eligible(scores, label, sem, text_margin)
            actual = v2._finite(dino[label], "dino") > draft_dino
            pair_correct.append(predicted == actual)
        available = [label for label in scores if label in dino]
        oracle = max(available, key=lambda label: v2._finite(dino[label], "dino"))
        exact.append(selected == oracle)
    sign = sum(pair_correct) / len(pair_correct) if pair_correct else 0.0
    mean = sum(deltas) / len(deltas) if deltas else 0.0
    harmful = sum(1 for d in deltas if d < v2.HARM_MARGIN)
    exact_rate = sum(exact) / len(exact) if exact else 0.0
    return sign, harmful, mean, exact_rate, len(pair_correct)


def fit_selector(rows: Sequence[Mapping]) -> SelectorFitV3:
    """Grid-search the single text-margin floor on eligible ``rows``.

    Only one threshold now -- the semantic gate is binary -- so overfitting room
    is far smaller than v2. Feasible fits meet sign-accuracy and harmful bars and
    earn a strictly positive mean selected DINO delta.
    """
    rows = list(rows)
    margins = sorted(v2._nonneg_deltas(rows, "text_relevance"))
    best, best_fit = None, None
    for tm in margins:
        sign, harmful, mean, exact, _ = _metrics(rows, tm)
        feasible = (sign >= v2.MIN_SIGN_ACCURACY and harmful <= v2.MAX_HARMFUL
                    and mean > 0.0)
        key = (not feasible, -mean, -exact, -tm, tm)
        if best is None or key < best:
            best, best_fit = key, SelectorFitV3(
                text_margin=tm, feasible=feasible, sign_accuracy=sign,
                harmful=harmful, mean_dino_delta=mean, exact_best_rate=exact)
    assert best_fit is not None
    return best_fit


def _selector_rows(manifest, reranker, dino, candidate_semantic) -> dict:
    rows = h2._selector_rows(manifest, reranker, dino)
    for cid, row in rows.items():
        verdicts = candidate_semantic.get(cid, {})
        sem = {label: bool(v.get("ok")) for label, v in verdicts.items()}
        for label in row["scores"]:
            if label == "draft":
                continue
            if label not in sem:
                raise ValueError(f"{cid}: no candidate semantic verdict for {label}")
        row["semantic"] = sem
    return rows


def develop(manifest, semantic, reranker, dino, candidate_semantic) -> dict:
    """Leave-one-concept-out development study with the v3 semantic gate.

    Reuses v2's (already validated) detector folds and baselines verbatim; only
    the selector changes.
    """
    result = h2.develop(manifest, semantic, reranker, dino)

    sel_rows = _selector_rows(manifest, reranker, dino, candidate_semantic)
    ids = sorted(sel_rows)

    deltas, pair_correct, exact_best = [], [], []
    cohort_deltas: dict[str, list[float]] = {}
    per_case: dict[str, dict] = {}
    fold_fits: dict[str, dict] = {}
    for cid in ids:
        train = [sel_rows[c] for c in ids if c != cid]
        fit = fit_selector(train)
        row = sel_rows[cid]
        selected = select_repair(row["scores"], row["semantic"], fit.text_margin)
        d = row["dino"]
        draft_dino = v2._finite(d["draft"], "dino draft")
        delta = v2._finite(d[selected], "dino") - draft_dino
        deltas.append(delta)
        cohort_deltas.setdefault(row["cohort"], []).append(delta)
        for label in row["scores"]:
            if label == "draft":
                continue
            predicted = _eligible(row["scores"], label, row["semantic"],
                                  fit.text_margin)
            actual = v2._finite(d[label], "dino") > draft_dino
            pair_correct.append(predicted == actual)
        available = [label for label in row["scores"] if label in d]
        oracle = max(available, key=lambda label: v2._finite(d[label], "dino"))
        exact_best.append(selected == oracle)
        per_case[cid] = {"selected": selected, "dino_delta": delta,
                         "cohort": row["cohort"],
                         "text_margin": fit.text_margin}
        fold_fits[cid] = {"text_margin": fit.text_margin,
                          "feasible": fit.feasible}

    from ragregen import metrics
    sign = sum(pair_correct) / len(pair_correct) if pair_correct else 0.0
    harmful = sum(1 for d in deltas if d < v2.HARM_MARGIN)
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    exact = sum(exact_best) / len(exact_best) if exact_best else 0.0
    cohorts = {}
    for cohort, ds in sorted(cohort_deltas.items()):
        interval = metrics.paired_bootstrap_ci(ds)
        cohorts[cohort] = {
            "n": len(ds), "mean_dino_delta": sum(ds) / len(ds),
            "dino_delta_95ci": list(interval) if interval else None,
            "improved": sum(1 for x in ds if x > 0),
            "unchanged": sum(1 for x in ds if x == 0),
            "worsened": sum(1 for x in ds if x < 0)}

    result["selector"] = {
        "denominator": result["selector"]["denominator"],
        "comparisons": len(pair_correct),
        "sign_accuracy": sign,
        "harmful": harmful,
        "mean_dino_delta": mean_delta,
        "exact_best_rate": exact,
        "cohorts": cohorts,
        "signal": "candidate_semantic + text floor (v3, Option A)",
    }
    result["selector_folds"] = fold_fits
    for cid, rec in per_case.items():
        result["cases"][cid].update(rec)

    # Rebuild the per-fold table so it reports v3 selections, not the v2 ones
    # inherited from h2.develop. Detector fields on each fold are unchanged.
    for fold in result["folds"]:
        rec = per_case.get(fold["case_id"])
        fold.pop("selector_fit", None)
        if rec is None:  # not selector-eligible: no v3 selection at all
            fold["selected"] = None
            fold["dino_delta"] = None
        else:
            fold["selected"] = rec["selected"]
            fold["dino_delta"] = rec["dino_delta"]
            fold["cohort"] = rec["cohort"]
            fold["selector_text_margin"] = rec["text_margin"]

    v2_conf = result["detector"]["v2_guarded"]
    result["gates"] = h2._readiness_gates(v2_conf, sign, harmful, cohorts)
    result["readiness"] = h2._readiness(result["gates"])
    result["policy_version"] = "v3"
    return result
