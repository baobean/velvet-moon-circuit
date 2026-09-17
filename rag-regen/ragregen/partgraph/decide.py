"""The pre-registered Phase-0 GO/NO-GO rule. Frozen before data (spec §8)."""
from __future__ import annotations

import numpy as np

from ragregen import metrics

#: GAIN margin in DINO cosine, fixed before any scoring. Do not tune.
PHASE0_GAIN_MARGIN = 0.02


def gain_verdict(deltas, *, seed: int = 0) -> dict:
    vals = [d for d in deltas if d is not None]
    mean = float(np.mean(vals)) if vals else float("nan")
    ci = metrics.paired_bootstrap_ci(vals, seed=seed)
    if ci is None:
        return {"mean": mean, "ci": None, "n": len(vals), "verdict": "UNDERPOWERED"}
    lo, hi = ci
    if mean >= PHASE0_GAIN_MARGIN and lo > 0:
        verdict = "GAIN"
    elif mean <= -PHASE0_GAIN_MARGIN and hi < 0:
        verdict = "LOSS"
    else:
        verdict = "NO_LARGE_EFFECT"
    return {"mean": mean, "ci": [lo, hi], "n": len(vals), "verdict": verdict}
