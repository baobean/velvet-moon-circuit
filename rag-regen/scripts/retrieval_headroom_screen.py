#!/usr/bin/env python
"""Wk1 headroom half of the fail-fast gate (spec section 8).

The rarity/headroom screen has two independent halves: does text-only FLUX
*fail* (screen_premise.py, hand-labelled) AND is single-image retrieval
*weak*? A dataset only survives if BOTH hold -- if retrieval is already
near-ceiling, an MMKG mechanism has nothing to add over "just retrieve"
(the prior trap this project already fell into once; see
docs/superpowers/mmkg-negative-results.md equivalents in the handoff doc).

This script measures the retrieval half: leave-one-out top-1 species
identification accuracy over a dataset.yaml's gt_refs, using the SAME
retriever checkpoint pipeline.yaml declares (siglip_so400m_384 by default) --
never the DINO eval encoder (that would be self-marking in the opposite
direction from the usual concern: it would tell us how the EVAL metric sees
the images, not how a retrieval baseline would).

Usage:
  ./scripts/retrieval_headroom_screen.py --dataset configs/partgraph_cub_wk1_screen.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, env  # noqa: E402


def leave_one_out_top1(vecs, labels):
    """For each row, is its nearest OTHER row's label the same as its own?

    Returns (per_case_correct: dict[label, (n_correct, n_total)], overall_acc).
    """
    import numpy as np

    n = vecs.shape[0]
    norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    sims = norms @ norms.T
    per_label: dict[str, list[int]] = {}
    n_correct = 0
    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0  # exclude self
        j = int(row.argmax())
        correct = int(labels[j] == labels[i])
        n_correct += correct
        per_label.setdefault(labels[i], [0, 0])
        per_label[labels[i]][0] += correct
        per_label[labels[i]][1] += 1
    return per_label, n_correct / n if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default -- this script is meant to run "
                         "alongside a GPU-resident FLUX drafting job "
                         "(screen_premise.py) on the same shared card.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)

    from ragregen import encoders

    print(f"[headroom] retriever={pipe_cfg.retriever} device={args.device} "
          f"cases={len(ds.cases)}", flush=True)
    encoder = encoders.build_encoder(pipe_cfg.retriever, device=args.device)

    paths: list[Path] = []
    labels: list[str] = []
    for case in ds.cases:
        for ref in case.gt_refs:
            paths.append(ref)
            labels.append(case.concept)

    print(f"[headroom] embedding {len(paths)} images...", flush=True)
    vecs = encoder.encode_images(paths)
    encoder.free()

    per_label, overall_acc = leave_one_out_top1(vecs, labels)

    print(f"\n[headroom] overall leave-one-out top-1 accuracy: "
          f"{overall_acc:.3f} ({sum(v[0] for v in per_label.values())}/"
          f"{sum(v[1] for v in per_label.values())})")
    print(f"[headroom] gate reading: retrieval is "
          f"{'STRONG (near-ceiling -- bad for headroom)' if overall_acc >= 0.8 else 'WEAK/MODERATE (good -- there is headroom)' }")
    print("\nper-species accuracy:")
    for label, (nc, nt) in sorted(per_label.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {nc}/{nt}  {label}")

    result = {
        "dataset": ds.name,
        "retriever": pipe_cfg.retriever,
        "n_images": len(paths),
        "n_species": len(per_label),
        "overall_top1_accuracy": overall_acc,
        "per_species": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1]}
                        for k, v in per_label.items()},
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    out = args.out or (env.PROJECT_ROOT / "outputs" /
                       f"headroom_{ds.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[headroom] wrote {out}")
    return 0


if __name__ == "__main__":
    env.exit_now(main())
