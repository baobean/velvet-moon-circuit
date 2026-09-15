import pytest

pytestmark = pytest.mark.gpu

ROOT = "data/treevill/rawdata2"
SPECIES_NAME = "Ashok"  # 15 unique images, hand-checked to include a canopy/leaf shot


def test_build_kg_from_real_refs(tmp_path):
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.dataset import load_species, split_refs
    from graft.kg_build import PART_NAMES, build_kg
    from graft.models import Models
    from graft.schema import ConceptKG

    sp = load_species(ROOT, SPECIES_NAME)
    build_refs, _heldout = split_refs(sp, k_build=5, seed=0)

    cfg = GraftConfig(outputs_dir=str(tmp_path))
    models = Models(cfg)

    kg = build_kg(SPECIES_NAME, build_refs, models, cfg)

    kg_path = tmp_path / SPECIES_NAME / "kg.json"
    assert kg_path.exists()

    reloaded = ConceptKG.from_json(str(kg_path))
    assert reloaded.concept == SPECIES_NAME
    assert len(reloaded.parts) == len(PART_NAMES)
    assert len(reloaded.global_attrs) > 0
    assert reloaded.anchor == ""  # anchor/delta pass dropped (spec section 5)
    assert len(reloaded.ref_embeddings) == len(reloaded.ref_paths)
    for part in reloaded.parts:
        # A part with no reliable box is unscoreable: empty embeddings, no
        # whole-image fallback (spec section 8.1(b)).
        assert part.embeddings == {} or (
            "siglip2" in part.embeddings and "dino" in part.embeddings
        )
        if part.embeddings:
            assert len(part.embeddings["siglip2"]) > 0
