import csv
import json

import yaml
from PIL import Image

from scripts.hybrid_verifier_report import main


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


def test_report_materializes_three_arms_and_frozen_decisions(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    cases = []
    for case_id, kind, cohort in (
            ("rare_bad", "target", "bridge"),
            ("common_good", "control", "common")):
        paths = []
        for index in range(3):
            path = refs / f"{case_id}_{index}.png"
            Image.new("RGB", (8, 8), "white").save(path)
            paths.append(str(path))
        cases.append({
            "id": case_id, "prompt": f"a {case_id}",
            "concept": case_id, "coarse": "thing", "kind": kind,
            "cohort": cohort, "gt_refs": paths,
        })
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({
        "name": "fixture", "images_root": "/", "cases": cases}))

    screen = tmp_path / "screen"
    candidates = tmp_path / "candidates"
    for case_id in ("rare_bad", "common_good"):
        (screen / case_id).mkdir(parents=True)
        (candidates / case_id).mkdir(parents=True)
        Image.new("RGB", (8, 8), "blue").save(
            screen / case_id / "draft.png")
        Image.new("RGB", (8, 8), "green").save(
            candidates / case_id / "attempt_1.png")

    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case_id", "prompt", "concept", "draft_path", "verdict",
            "verdict_identity", "notes"])
        writer.writeheader()
        writer.writerow({"case_id": "rare_bad", "prompt": "a rare_bad",
                         "concept": "rare_bad", "draft_path": "",
                         "verdict": "fail", "verdict_identity": "fail",
                         "notes": ""})
        writer.writerow({"case_id": "common_good",
                         "prompt": "a common_good",
                         "concept": "common_good", "draft_path": "",
                         "verdict": "pass", "verdict_identity": "pass",
                         "notes": ""})

    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({
        "rare_bad": {"ok": True}, "common_good": {"ok": True}}))
    reranker = tmp_path / "reranker.json"
    reranker.write_text(json.dumps({"cases": {
        "rare_bad": {"scores": {
            "draft": _score(0.3, 0.2),
            "attempt_1": _score(0.5, 0.4)}},
        "common_good": {"scores": {
            "draft": _score(0.7, 0.6),
            "attempt_1": _score(0.6, 0.5)}},
    }}))
    dino = tmp_path / "dino.json"
    dino.write_text(json.dumps({"cases": {
        "rare_bad": {"draft": 0.2, "attempt_1": 0.5},
        "common_good": {"draft": 0.7, "attempt_1": 0.6},
    }}))
    output = tmp_path / "report"

    rc = main([
        "--dataset", str(dataset), "--labels", str(labels),
        "--semantic", str(semantic), "--reranker", str(reranker),
        "--dino", str(dino), "--screen-run", str(screen),
        "--candidate-run", str(candidates), "--output-dir", str(output),
    ])

    assert rc == 0
    decisions = json.loads((output / "decisions.json").read_text())
    assert decisions["rare_bad"]["hybrid"] == "attempt_1"
    assert decisions["common_good"]["hybrid"] == "draft"
    for arm in ("draft", "no_verifier", "hybrid"):
        assert (output / "arms" / arm / "rare_bad.png").is_file()
    report = json.loads((output / "hybrid_validation.json").read_text())
    assert report["gates"]["detector"]["passed"] is True
    assert report["status"] == "INCONCLUSIVE"
    assert "Frozen hybrid verifier holdout" in (
        output / "hybrid_validation.md").read_text()
