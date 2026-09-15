from __future__ import annotations
import numpy as np

STARVE_CONCEPTS = ["Bamboo","Ashok","Egyptian lotus","Nageshore",
                   "Avocado","Camphor Tree","Hijol","Ashore"]
STARVE_LEVELS = [1, 2, 3, 5]
N_DRAWS = 3
CONDITIONS = ["isolated","random","rawnn","hub"]   # +OracleHub-k deferred (needs taxonomy map)

def keep_draw(build_refs, keep_n, seed):
    refs = list(build_refs)
    if keep_n >= len(refs):
        return refs
    rng = np.random.default_rng(seed)
    idx = sorted(rng.permutation(len(refs))[:keep_n].tolist())
    return [refs[i] for i in idx]

def iter_cells():
    for c in STARVE_CONCEPTS:
        for lvl in STARVE_LEVELS:
            for d in range(N_DRAWS):
                yield (c, lvl, d)
