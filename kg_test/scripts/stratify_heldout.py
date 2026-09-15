#!/usr/bin/env python3
"""Reliability-stratified re-aggregation of the full-corpus GRAFT-vs-B1 eval.

Reads the raw fast-driver artifacts and recomputes the GRAFT-vs-B1 fidelity
deltas stratified by per-species held-out count (a proxy for how reliably a
species' fidelity can be measured at all). No figure in the sprint report is
hand-edited; every number in the "reliability-stratified re-analysis" section
comes from this script.

Usage:
    python3 scripts/stratify_heldout.py [--eval-dir outputs/eval_fast_full]
"""
import argparse
import json
import statistics as st
from pathlib import Path

PILOT = ["Bamboo", "Ashok", "Egyptian lotus", "Nageshore",
         "Avocado", "Camphor Tree", "Hijol", "Ashore"]


def wilcoxon_greater_p(deltas):
    """One-sided Wilcoxon signed-rank p (H1: median > 0). scipy if available."""
    try:
        from scipy.stats import wilcoxon
        nz = [d for d in deltas if d != 0]
        return float(wilcoxon(nz, alternative="greater").pvalue)
    except Exception:
        return None


def summarize(rows):
    dl = [r["delta"] for r in rows]
    wins = sum(1 for d in dl if d > 0)
    return {
        "n": len(dl),
        "mean": st.mean(dl),
        "median": st.median(dl),
        "sd": st.pstdev(dl),
        "wins": wins,
        "win_rate": wins / len(dl),
        "wilcoxon_p": wilcoxon_greater_p(dl),
    }


def weighted_mean(rows, weight):
    num = sum(weight[r["species"]] * r["delta"] for r in rows)
    den = sum(weight[r["species"]] for r in rows)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval_fast_full")
    args = ap.parse_args()
    d = Path(args.eval_dir)

    analysis = json.loads((d / "analysis.json").read_text())
    heldout = json.loads((d / "_work" / "heldout.json").read_text())
    hc = {s: (len(v) if isinstance(v, list) else 0) for s, v in heldout.items()}

    print(f"# Reliability-stratified re-aggregation  ({args.eval_dir})\n")

    # held-out distribution
    from collections import Counter
    dist = Counter(hc.values())
    print("Held-out count distribution:")
    for k in sorted(dist):
        print(f"  held-out={k:<3} : {dist[k]:>2} species")
    allc = sorted(hc.values())
    print(f"  corpus median={st.median(allc)}  mean={st.mean(allc):.2f}  "
          f"#(>=3)={sum(1 for c in allc if c>=3)}/{len(allc)}\n")

    for metric in ("dino", "siglip2", "clip_i"):
        per = analysis["graft_vs_b1"][metric]["per_species"]
        print(f"## {metric}")
        for thr in (1, 2, 3, 5):
            sub = [r for r in per if hc.get(r["species"], 0) >= thr]
            s = summarize(sub)
            p = f"{s['wilcoxon_p']:.4f}" if s["wilcoxon_p"] is not None else "n/a"
            print(f"  held-out>={thr}: n={s['n']:>2}  mean={s['mean']:+.4f}  "
                  f"median={s['median']:+.4f}  SD={s['sd']:.3f}  "
                  f"win={s['wins']}/{s['n']} ({100*s['win_rate']:.0f}%)  "
                  f"Wilcoxon_p={p}")
        wm = weighted_mean(per, hc)
        print(f"  held-out-weighted mean delta (weight = #held-out photos) = {wm:+.4f}\n")

    # Mango cross-metric outlier check
    print("## Mango outlier cross-check (build pool = 5 refs, held-out = 1)")
    for metric in ("dino", "siglip2", "clip_i"):
        per = {r["species"]: r for r in analysis["graft_vs_b1"][metric]["per_species"]}
        m = per["Mango"]
        print(f"  {metric:<8} ours={m['ours']:.3f}  b1={m['b1']:.3f}  delta={m['delta']:+.3f}")

    # pilot species reliability
    print("\n## Pilot (n=8) species held-out counts")
    for s in PILOT:
        print(f"  {s:<16} held-out={hc.get(s, '?')}")
    pc = [hc[s] for s in PILOT]
    print(f"  pilot median={st.median(pc)}  mean={st.mean(pc):.1f}  "
          f"all >= 4: {all(c >= 4 for c in pc)}")


if __name__ == "__main__":
    main()
