"""Pure helpers shared by the fast eval driver (spec 2026-09-03). No torch at
import time -- the GPU workers import their model code lazily inside main()."""
from __future__ import annotations

import os
from typing import List, Optional


def image_id(species: str, method: str, name_mode: str, ip_scale: Optional[float], seed: int) -> str:
    return f"{species}|{method}|{name_mode}|ip{ip_scale}|seed{seed}"


def id_to_filename(iid: str) -> str:
    return iid.replace("|", "_").replace("/", "_") + ".png"


def kg_is_usable(kg_path: str) -> bool:
    """True iff kg_path exists and holds ref_embeddings aligned to ref_paths.
    A Phase-1 kg.json (no ref_embeddings) is NOT usable -- it must be rebuilt
    (spec 2026-08-30 C1)."""
    if not os.path.exists(kg_path):
        return False
    try:
        from graft.schema import ConceptKG

        kg = ConceptKG.from_json(kg_path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(kg.ref_embeddings) and len(kg.ref_embeddings) == len(kg.ref_paths)


SWEPT_METHODS = ("ours", "ours_notree", "b1")


def build_manifest(species_names, methods, name_modes, ip_scales, seeds) -> List[dict]:
    """Flat (species,method,name_mode,ip_scale,seed) rows. Swept methods span
    every ip_scale; b0/b2 ignore ip (one cell per name_mode, ip_scale=None)."""
    rows: List[dict] = []
    for species in species_names:
        for method in methods:
            ips = list(ip_scales) if method in SWEPT_METHODS else [None]
            for name_mode in name_modes:
                for ip in ips:
                    for seed in range(seeds):
                        rows.append({
                            "image_id": image_id(species, method, name_mode, ip, seed),
                            "species": species, "method": method,
                            "name_mode": name_mode, "ip_scale": ip, "seed": seed,
                        })
    return rows


def prompt_for(method: str, concept: str, kg, b2_attrs, neutralize: bool) -> str:
    """Reproduce the four methods' prompt heads exactly as graft/baselines.py +
    graft/generate.py do. Lazy import to keep module torch-free."""
    from graft.prompt import compose_prompt, compose_ravel_prompt, concept_token

    if method in ("ours", "ours_notree"):
        return compose_prompt(kg, neutralize=neutralize)
    if method == "b2":
        return compose_ravel_prompt(concept, b2_attrs, neutralize=neutralize)
    # b0, b1: bare neutralized head (matches graft/baselines.py)
    return f"a photo of a {concept_token(concept, neutralize)}"


def exemplar_for(method: str, kg, pool_paths=None, pool_embed_fn=None) -> Optional[str]:
    """Exemplar ref path for the IP-Adapter-conditioned methods. ours/ours_notree/b1
    all use the SigLIP2 medoid of the concept's stored ref_embeddings -> exemplar
    parity by construction (b1's pool IS the KG's build refs). b0/b2 use no exemplar."""
    if method in ("ours", "ours_notree", "b1"):
        from graft.generate import select_exemplar
        return select_exemplar(kg)
    return None


_METRIC_KEYS = ("dino", "siglip2", "clip_i", "clip_t", "attribute_accuracy")


def _verify_score(m: dict, use_part_tree: bool) -> float:
    """Compute verify score from a metrics dict. Without part_tree, it's just
    attribute_accuracy. With part_tree, blend 0.5·mean_part_sim + 0.5·attr."""
    attr = m.get("attribute_accuracy", 0.0)
    if not use_part_tree:
        return attr
    sims = [m[k] for k in ("part_sim_dino", "part_sim_siglip") if k in m]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return 0.5 * mean_sim + 0.5 * attr


def rows_from_tables(manifest, tables, n_heldout_by_species, use_part_tree=False):
    """Aggregate manifest seed-rows by cell, pick best seed per cell by verify score.
    Ties broken by lowest seed. Return (rows, missing_cells)."""
    # group seed-rows by cell
    cells: dict = {}
    for row in manifest:
        key = (row["species"], row["method"], row["name_mode"], row["ip_scale"])
        cells.setdefault(key, []).append(row)

    rows, missing = [], []
    for (species, method, name_mode, ip_scale), seed_rows in cells.items():
        scored = [
            (r["seed"], tables[r["image_id"]])
            for r in seed_rows
            if r["image_id"] in tables
            and all(k in tables[r["image_id"]] for k in _METRIC_KEYS)
        ]
        if not scored:
            missing.append({"species": species, "method": method,
                            "name_mode": name_mode, "ip_scale": ip_scale})
            continue
        # best verify score, lowest seed on tie
        seed, best = min(scored, key=lambda sm: (-_verify_score(sm[1], use_part_tree), sm[0]))
        rows.append({
            "method": method, "species": species, "name_mode": name_mode,
            "ip_scale": ip_scale, "n_heldout": n_heldout_by_species[species],
            **{k: best[k] for k in _METRIC_KEYS},
        })
    return rows, missing
