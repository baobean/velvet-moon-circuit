"""Best-effort taxonomy oracle for the Treevill concepts: common name -> (scientific
name, family). This is the ORTHOGONAL relational signal for the OracleHub experiment --
derived from species identity, NOT from any image embedding, so it can group crops that
are taxonomically related yet embedding-distant (the case RawNN provably misses). Family
assignments are hand-curated from the common/scientific names and should be verified
against dataset provenance before publication; the structural conclusions are robust to a
few reassignments. Concepts absent here are treated as taxonomy-unknown (family None):
valid RawNN candidates, never OracleHub siblings."""
from __future__ import annotations

# concept -> (scientific name, family)
TAXONOMY: dict[str, tuple[str, str]] = {
    # Fabaceae (legumes) -- the dense family
    "Ashok": ("Saraca asoca", "Fabaceae"),
    "Akashmoni": ("Acacia auriculiformis", "Fabaceae"),
    "Karanja": ("Pongamia pinnata", "Fabaceae"),
    "Sisso": ("Dalbergia sissoo", "Fabaceae"),
    "Golden Shower Tree": ("Cassia fistula", "Fabaceae"),
    "Koinar": ("Bauhinia purpurea", "Fabaceae"),
    "Piliostigma": ("Piliostigma malabaricum", "Fabaceae"),
    "Holudkrishnachura": ("Peltophorum pterocarpum", "Fabaceae"),
    # Lauraceae
    "Avocado": ("Persea americana", "Lauraceae"),
    "Camphor Tree": ("Cinnamomum camphora", "Lauraceae"),
    # Combretaceae (Terminalia)
    "Bahera": ("Terminalia bellirica", "Combretaceae"),
    "Haritaki": ("Terminalia chebula", "Combretaceae"),
    # Moraceae (Artocarpus)
    "Jack Fruit": ("Artocarpus heterophyllus", "Moraceae"),
    "Chaplash": ("Artocarpus chama", "Moraceae"),
    # Myrtaceae
    "Guava": ("Psidium guajava", "Myrtaceae"),
    "Baro bottle brush": ("Callistemon viminalis", "Myrtaceae"),
    # Anacardiaceae
    "Mango": ("Mangifera indica", "Anacardiaceae"),
    "Marking Nut tree": ("Semecarpus anacardium", "Anacardiaceae"),
    # Calophyllaceae
    "Nageshore": ("Mesua ferrea", "Calophyllaceae"),
    "Mastwood": ("Calophyllum inophyllum", "Calophyllaceae"),
    # Lecythidaceae
    "Cannonball Tree": ("Couroupita guianensis", "Lecythidaceae"),
    "Hijol": ("Barringtonia acutangula", "Lecythidaceae"),
    # Arecaceae (palms, monocot)
    "Palm": ("Arecaceae sp.", "Arecaceae"),
    "Khejur": ("Phoenix sylvestris", "Arecaceae"),
    # Rubiaceae
    "Haldu": ("Haldina cordifolia", "Rubiaceae"),
    "Crown Gardenia": ("Gardenia sp.", "Rubiaceae"),
    # confident singletons -- NN distractors, never siblings
    "Bamboo": ("Bambusoideae sp.", "Poaceae"),
    "Banana": ("Musa sp.", "Musaceae"),
    "Egyptian lotus": ("Nymphaea sp.", "Nymphaeaceae"),
    "Carambola": ("Averrhoa carambola", "Oxalidaceae"),
    "Teak": ("Tectona grandis", "Lamiaceae"),
    "Devil Tree": ("Alstonia scholaris", "Apocynaceae"),
    "Champaca": ("Magnolia champaca", "Magnoliaceae"),
    "Mahogany": ("Swietenia mahagoni", "Meliaceae"),
    "Australian Pine": ("Casuarina equisetifolia", "Casuarinaceae"),
    "Kamala Tree": ("Mallotus philippensis", "Euphorbiaceae"),
}


def family_of(concept: str) -> str | None:
    rec = TAXONOMY.get(concept)
    return rec[1] if rec else None


def family_id(concept: str) -> int:
    """Stable integer id for a concept's family (-1 if unknown), for use as
    select_borrowed pool_labels. Sorted family names -> contiguous ids."""
    fam = family_of(concept)
    if fam is None:
        return -1
    return _FAMILY_IDS[fam]


def multi_member_families() -> dict[str, list[str]]:
    """Families with >=2 mapped concepts -- the only ones that can furnish a sibling."""
    out: dict[str, list[str]] = {}
    for c, (_, fam) in TAXONOMY.items():
        out.setdefault(fam, []).append(c)
    return {f: cs for f, cs in out.items() if len(cs) >= 2}


_FAMILY_IDS = {fam: i for i, fam in enumerate(sorted({f for _, f in TAXONOMY.values()}))}
