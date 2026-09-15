"""GraftConfig: every model id, path, and tunable used across GRAFT."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Optional


@dataclass(frozen=True)
class GraftConfig:
    hf_cache: str = "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache"
    device: str = "cuda"

    # Cached model ids -- do not substitute (see plan Global Constraints).
    sdxl_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    vlm_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reranker_id: str = "Qwen/Qwen3-VL-Reranker-2B"
    siglip_id: str = "google/siglip2-base-patch16-384"
    dino_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    clip_id: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    dino_detector_id: str = "IDEA-Research/grounding-dino-base"
    sam_id: str = "facebook/sam-vit-huge"

    # Tunables.
    ip_scale: float = 0.6
    n_seeds: int = 4
    n_refine: int = 2
    part_sim_threshold: float = 0.5
    attr_pass_threshold: float = 0.6
    neutralize_name: bool = True          # replace the concept name with NEUTRAL_TOKEN everywhere
    exemplar_selection: str = "medoid"    # "medoid" (name-free) | "text" (Phase-1 reranker/text)
    use_part_tree: bool = True            # score part crops in verify (False = attribute-gate only)
    k_build_refs: int = 5                 # CAP on build refs; effective build = min(cap, n_unique - 1)
    hub_label_do_sample: bool = False        # P10: locked, serialized into cfg.yaml
    hub_label_max_new_tokens: int = 512       # P10: locked

    # Stage-A' (per-part inpaint restore) tunables.
    restore_k: int = 4                           # borrowed budget cap (k_eff <= this)
    # list (not tuple) so to_yaml/from_yaml roundtrips to an equal value (yaml has no tuple).
    restore_build_levels: list = field(default_factory=lambda: [0, 1, 2, 4])  # own-P-crop starvation levels
    restore_n_draws: int = 3
    restore_min_mask_area_frac: float = 0.01     # SAM mask area / box area, smoke lower bound
    restore_max_mask_area_frac: float = 0.9      # ... upper bound
    restore_inpaint_steps: int = 50
    restore_inpaint_strength: float = 0.99       # near-full repaint of the masked box
    restore_guidance: float = 7.5

    outputs_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "GraftConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown GraftConfig keys in {path}: {sorted(unknown)}")
        return replace(cls(), **raw)

    def to_yaml(self, path: str) -> None:
        """Dump every field (not just diffs from default) so a worker
        subprocess can reconstruct this exact config via from_yaml()."""
        import yaml
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f)
