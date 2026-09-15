"""GPU worker: attach a human-readable label to each already-formed PartType hub
(spec P10). INTERPRETABILITY ONLY -- membership is fixed (SigLIP2+HDBSCAN, Task 6)
and no label is ever read by the transfer consumer. Deterministic Qwen2.5-VL."""
from __future__ import annotations
import argparse
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft.build_graph import top_centroid_members
from graft.models import Models

INSTRUCTION = ("These are cropped photos of the same plant part from different plants. "
               "In 3-8 words, name the part type by its visible shape/texture/arrangement. "
               "Reply with only the phrase.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True); ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    by_id = graph.instances_by_id()
    models = Models(cfg)
    for hub in graph.part_types:
        crops = [Image.open(by_id[m].crop_path).convert("RGB")
                 for m in top_centroid_members(graph, hub, k=4)]
        hub.label = models.vlm.describe(              # decoding passed EXPLICITLY from config (P10)
            crops, INSTRUCTION,
            do_sample=cfg.hub_label_do_sample,
            max_new_tokens=cfg.hub_label_max_new_tokens,
        ).strip()
    models.unload("vlm")
    graph.to_json(a.graph)                         # rewrite in place; membership untouched

if __name__ == "__main__":
    main()
