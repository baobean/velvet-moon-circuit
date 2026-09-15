"""Paired, per-species GRAFT-vs-baseline comparison (spec section 11) -- the
central read now that exemplar selection and the name are no longer confounds.

Everything here is pure: it consumes the `rows` list that `run_eval.evaluate`
produces (each row tagged with `method`, `species`, `name_mode`, `ip_scale`,
`n_heldout` plus the metric fields) and returns plain dicts, so the sprint
readout is unit-testable without running the GPU matrix.
"""
from __future__ import annotations

from typing import List, Optional

#: Primary fidelity metrics -- the paired GRAFT-vs-B1 and prune readouts (spec 11).
PAIRED_METRICS = ("dino", "siglip2", "clip_i")
#: Everything worth a named-minus-neutral delta (CLIP-T is a diagnostic, not a win metric).
DELTA_METRICS = ("dino", "siglip2", "clip_i", "clip_t", "attribute_accuracy")


def paired_analysis(rows: List[dict], metric: str, a: str = "ours", b: str = "b1") -> dict:
    """Per-species `a` minus `b` on `metric`, over the species where BOTH arms
    produced a row.

    `mean_delta`/`win_rate` are None when `n == 0`: with no pair to compare
    there is no delta, and reporting 0.0/0.0 would read as "`a` tied `b` and
    never won" -- exactly the wrong conclusion when the real cause is that the
    `a` rows are missing (e.g. a stale kg.json dropped every GRAFT cell)."""
    by = {}
    for r in rows:
        if metric in r:
            by.setdefault(r["species"], {})[r["method"]] = r[metric]
    per = []
    for sp, m in sorted(by.items()):
        if a in m and b in m:
            per.append({"species": sp, a: m[a], b: m[b], "delta": m[a] - m[b]})
    n = len(per)
    return {
        "per_species": per,
        "n": n,
        "mean_delta": (sum(p["delta"] for p in per) / n) if n else None,
        "win_rate": (sum(1 for p in per if p["delta"] >= 0) / n) if n else None,
    }


def _mean(rows: List[dict], metric: str) -> Optional[float]:
    """Mean of `metric` over the rows that carry it; None if none do."""
    vals = [r[metric] for r in rows if metric in r]
    return (sum(vals) / len(vals)) if vals else None


def best_ip_scale(rows: List[dict]) -> Optional[float]:
    """The `ip_scale` maximizing mean DINO over the neutral GRAFT rows (spec 9).

    Falls back to the smallest `ip_scale` present in `rows` when there are no
    such rows or when the maximum is tied, and to None when no row is tagged
    with an `ip_scale` at all.
    """
    all_ips = sorted({r["ip_scale"] for r in rows if r.get("ip_scale") is not None})
    if not all_ips:
        return None
    fallback = all_ips[0]

    by_ip: dict = {}
    for r in rows:
        if r.get("method") == "ours" and r.get("name_mode") == "neutral" and "dino" in r:
            if r.get("ip_scale") is not None:
                by_ip.setdefault(r["ip_scale"], []).append(r["dino"])
    if not by_ip:
        return fallback

    scored = [(sum(v) / len(v), ip) for ip, v in by_ip.items()]
    best = max(s for s, _ in scored)
    winners = sorted(ip for s, ip in scored if s == best)
    if len(winners) != 1:
        return fallback
    return winners[0]


def build_analysis(rows: List[dict]) -> dict:
    """Assemble the sprint readout (spec 11) from the matrix rows.

    Produces:
      - `best_ip`: see `best_ip_scale`.
      - `graft_vs_b1` / `graft_vs_notree`: paired per-species deltas over the
        *neutral* rows at `best_ip` -- the confound-free comparison (spec 11)
        and the section-8.3 prune read respectively.
      - `named_minus_neutral`: per method, mean(named) - mean(neutral) per
        metric. Restricted to `best_ip` for the swept methods; methods that
        were run at a single `ip_scale` (b0/b2) have no rows there, so they
        fall back to all of their rows. None when either side is empty or the
        metric is absent from it.
    """
    best_ip = best_ip_scale(rows)

    neutral_at_best = [
        r for r in rows
        if r.get("name_mode") == "neutral" and r.get("ip_scale") == best_ip
    ]
    graft_vs_b1 = {
        m: paired_analysis(neutral_at_best, m, a="ours", b="b1") for m in PAIRED_METRICS
    }
    graft_vs_notree = {
        m: paired_analysis(neutral_at_best, m, a="ours", b="ours_notree")
        for m in PAIRED_METRICS
    }

    named_minus_neutral: dict = {}
    for method in sorted({r["method"] for r in rows if "method" in r}):
        method_rows = [r for r in rows if r.get("method") == method]
        at_best = [r for r in method_rows if r.get("ip_scale") == best_ip]
        scope = at_best or method_rows
        named = [r for r in scope if r.get("name_mode") == "named"]
        neutral = [r for r in scope if r.get("name_mode") == "neutral"]
        deltas = {}
        for metric in DELTA_METRICS:
            mn, mu = _mean(named, metric), _mean(neutral, metric)
            deltas[metric] = None if (mn is None or mu is None) else mn - mu
        named_minus_neutral[method] = deltas

    return {
        "best_ip": best_ip,
        "graft_vs_b1": graft_vs_b1,
        "graft_vs_notree": graft_vs_notree,
        "named_minus_neutral": named_minus_neutral,
    }
