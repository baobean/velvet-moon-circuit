import pytest

pytestmark = pytest.mark.gpu

ROOT = "data/treevill/rawdata2"
SPECIES_NAME = "Mango"  # 6 unique images, hand-checked as real distinct trunk/canopy photos


def test_run_concept_produces_final_and_report(tmp_path):
    from graft import env

    env.setup()
    from graft.config import GraftConfig
    from graft.dataset import load_species, split_refs
    from graft.models import Models
    from graft.pipeline import run_concept

    sp = load_species(ROOT, SPECIES_NAME)
    build_refs, _heldout = split_refs(sp, k_build=5, seed=0)

    cfg = GraftConfig(outputs_dir=str(tmp_path), n_refine=1)
    models = Models(cfg)

    result = run_concept(SPECIES_NAME, build_refs, models, cfg)

    final_path = tmp_path / SPECIES_NAME / "final.png"
    report_path = tmp_path / SPECIES_NAME / "report.json"
    assert final_path.exists()
    assert report_path.exists()
    assert result["report"].part_scores
