from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from graft.starvation import iter_cells, CONDITIONS
from graft.starvation_analysis import recovery_curve, paired_delta

def plan_commands(graph, kg_dir, root, out, config):
    cmds = []
    for (concept, level, draw) in iter_cells():
        for cond in CONDITIONS:
            cmds.append([sys.executable, "-m", "graft.worker_starve_cell",
                "--concept", concept, "--level", str(level), "--draw", str(draw),
                "--condition", cond, "--graph", graph,
                "--kg", os.path.join(kg_dir, concept, "kg.json"),
                "--root", root, "--out", out, "--config", config])
    return cmds

def _row_path(out, argv):                 # row filename mirrors the worker's output name
    c = argv[argv.index("--concept")+1]; l = argv[argv.index("--level")+1]
    dr = argv[argv.index("--draw")+1]; k = argv[argv.index("--condition")+1]
    return os.path.join(out, f"{c}_{l}_{dr}_{k}.json")

def collect(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*_*_*_*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows

def main():
    ap = argparse.ArgumentParser()
    for f in ("graph","kg-dir","root","out","config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for argv in plan_commands(a.graph, a.kg_dir, a.root, a.out, a.config):
        if os.path.exists(_row_path(a.out, argv)):     # resume: skip done cells
            continue
        subprocess.run(["bash", "scripts/gpu_queue.sh", *argv], check=True)
    rows = collect(a.out)
    with open(os.path.join(a.out, "recovery.json"), "w") as f:
        json.dump(recovery_curve(rows, "dino"), f, indent=2)
    with open(os.path.join(a.out, "hub_vs_rawnn.json"), "w") as f:
        json.dump(paired_delta(rows, "hub", "rawnn", level=1), f, indent=2)

if __name__ == "__main__":
    main()
