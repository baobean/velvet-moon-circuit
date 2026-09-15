"""Flag low-information ("generic") attribute values so M1 can re-ask for
discriminative ones. Pure and unit-tested; see spec section 7 (folds Phase 2
design section 5.1)."""
from __future__ import annotations

from typing import Sequence

_GENERIC = {
    "", "green", "full", "tree", "plant", "leaf", "leaves", "normal", "typical",
    "standard", "not visible", "none", "n/a", "na", "unknown", "brown", "grey", "gray",
}


def is_generic(value: str) -> bool:
    v = " ".join(value.strip().lower().split())
    if v in _GENERIC:
        return True
    return len(v.split()) < 2  # a single bare word is not discriminative enough


def count_generic(values: Sequence[str]) -> int:
    return sum(1 for v in values if is_generic(v))
