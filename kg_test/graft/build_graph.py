from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph
from graft.parttype_graph import induce_parttypes, hub_coherence, valid_hubs

def assemble_graph(instances: list[PartInstanceRec]) -> PartTypeGraph:
    part_types: list[PartTypeRec] = []
    next_id = 0
    for part in sorted({i.part for i in instances}):
        members = [i for i in instances if i.part == part]
        if len(members) < 3:
            continue
        embeds = np.array([m.siglip2 for m in members], dtype=float)
        concepts = [m.concept for m in members]
        labels = induce_parttypes(embeds)
        coh = hub_coherence(embeds, labels)
        keep = valid_hubs(labels, concepts, coh)
        for h in sorted(keep):
            idx = np.where(labels == h)[0]
            centroid = embeds[idx].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            part_types.append(PartTypeRec(
                id=next_id, part=part, member_ids=[members[i].id for i in idx],
                centroid=list(centroid), coherence=float(coh[h])))
            next_id += 1
    return PartTypeGraph(part_instances=list(instances), part_types=part_types)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    insts = []
    for p in sorted(glob.glob(os.path.join(a.crops_dir, "*.json"))):
        with open(p) as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            continue
        insts += [PartInstanceRec(**r) for r in raw]
    os.makedirs(a.out, exist_ok=True)
    assemble_graph(insts).to_json(os.path.join(a.out, "graph.json"))

def top_centroid_members(graph, hub, k: int = 4) -> list[str]:
    by_id = graph.instances_by_id()
    c = np.asarray(hub.centroid, dtype=float)
    scored = [(mid, float(np.asarray(by_id[mid].siglip2, dtype=float) @ c))
              for mid in hub.member_ids if mid in by_id]
    scored.sort(key=lambda t: -t[1])
    return [mid for mid, _ in scored[:k]]

if __name__ == "__main__":
    main()
