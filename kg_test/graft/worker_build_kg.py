#!/usr/bin/env python
"""Subprocess worker: build one concept's MMKG, then exit.

Runs `build_kg` in a fresh, single-purpose process so it never shares a
Python heap/CUDA context with any other phase. This process isolation is a
deliberate robustness measure: sustained heavy model load/unload churn
within one long-lived process (as the in-process orchestration originally
did) was observed to segfault intermittently, with a different specific
crash signature each time (see refine.py's and env.py's docstrings for the
two signatures diagnosed before this rewrite -- a third, unrelated-looking
signature after both fixes is what motivated giving up on chasing the exact
native-library interaction and isolating processes instead). A fresh
process starting clean and exiting unconditionally releases everything
(CPU heap and GPU memory alike) regardless of any internal reference cycle
or thread-safety issue in whatever went wrong.

Usage:
    python -m graft.worker_build_kg <concept> <outputs_dir> <cfg_yaml> <build_ref>...

On success, writes <outputs_dir>/<concept>/kg.json (via build_kg itself)
and exits 0. On failure, prints the traceback to stderr and exits 1.
"""
from __future__ import annotations

import sys

from graft import env

env.setup()


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(
            "usage: worker_build_kg.py <concept> <outputs_dir> <cfg_yaml> <build_ref>...",
            file=sys.stderr,
        )
        return 2
    concept, outputs_dir, cfg_yaml, *build_refs = argv

    from graft.config import GraftConfig
    from graft.kg_build import build_kg
    from graft.models import Models

    cfg = GraftConfig.from_yaml(cfg_yaml)
    models = Models(cfg)
    build_kg(concept, build_refs, models, cfg, outputs_dir=outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
