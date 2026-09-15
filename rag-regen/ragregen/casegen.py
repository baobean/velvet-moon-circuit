"""Turn ImageNet classes into cases. Doc 5 §5.

"Common" is measured, not asserted: classes are ranked by how often they
appear in the captions of the corpus we actually downloaded. That makes
commonness a property of the retrieval database rather than an opinion, and it
is honest about what the benchmark tests -- concepts the database knows.
"""
from __future__ import annotations

#: Prompts follow configs/dataset.yaml's house style: a subject doing
#: something somewhere. Varied so the verifier keys on the concept rather than
#: on a single repeated sentence shape.
SCENE_FRAMES: tuple[str, ...] = (
    "a {c} in soft afternoon light",
    "a {c} photographed against a plain background",
    "a {c} resting on a wooden table",
    "a {c} outdoors on an overcast day",
    "a {c} in sharp focus, close up",
    "a {c} standing in an open field",
    "a {c} lit from one side in a quiet room",
    "a {c} seen from a low angle",
    "a {c} on a bright summer morning",
    "a {c} against a dark neutral backdrop",
)


def rank_classes(
    counts: dict[str, int],
    coarse_map: dict[str, str],
    *,
    exclude: set[str],
    n: int,
    min_count: int = 1,
) -> list[str]:
    """The `n` most common classes that can actually become cases.

    Restricted to `coarse_map`: a class with no groundable superordinate
    cannot pass config.load_dataset, so letting it win a slot would just
    shrink the case set. Restricted to classes the corpus mentions at least
    `min_count` times: a class the corpus never mentions is not a common one,
    and -- since cmd_select seeds per-concept rows for the DATASET's concepts
    only -- a common concept found in the random bulk can sit at a single
    digit. A concept the corpus cannot serve `k` references for would become a
    case and then fail to retrieve, so callers pass the reference requirement
    here rather than accepting the default floor of 1. Ties break
    alphabetically so the same manifest yields the same selection on a re-run.
    """
    pool = [c for c in coarse_map
            if c not in exclude and counts.get(c, 0) >= min_count]
    pool.sort(key=lambda c: (-int(counts.get(c, 0)), c))
    return pool[:n]


def frame_prompt(concept: str, index: int) -> str:
    """A prompt for `concept`, deterministic in `index`."""
    return SCENE_FRAMES[index % len(SCENE_FRAMES)].format(c=concept)
