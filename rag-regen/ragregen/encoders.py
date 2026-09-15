"""Whole-image/text encoders for retrieval.

Separate from models.py on purpose: models.py scores CROPS against phrases
(Stream A), this scores WHOLE IMAGES against queries (step 4). The bake-off
compares FG-CLIP and SigLIP in both roles independently, because a region-text
model can win one and lose the other (spec §6 Stage 1).

Three deviations from the plan text, all forced by what the checkpoints and
the pinned transformers actually do:

- `timm/ViT-SO400M-14-SigLIP-384` is an open_clip-format repo (no config.json)
  and cannot be loaded through AutoModel/AutoProcessor. The native
  transformers checkpoint of the same weights is
  `google/siglip-so400m-patch14-384` (Task 7).
- `SiglipConfig` has no `projection_dim`; reading it raises AttributeError.
  Embedding width therefore resolves through `_resolve_dim`, which falls back
  to the tower hidden size (1152, matching validate.ENCODER_DIMS).
- Under transformers 5.x, `get_image_features`/`get_text_features` return the
  tower's `BaseModelOutputWithPooling`, not a tensor as they did under 4.x.
  The embedding is unwrapped by `_pooled`.
"""
from __future__ import annotations

import numpy as np

from ragregen import env

env.setup()  # must precede transformers import

ENCODER_IDS: dict[str, str] = {
    "siglip_so400m_384": "google/siglip-so400m-patch14-384",
    "siglip_base_224": "google/siglip-base-patch16-224",
    # Kept for the bake-off env, which runs an older transformers where
    # FG-CLIP's remote code loads. build_encoder refuses it here.
    "fgclip": "qihoo360/fg-clip-base",
    "openclip_l14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "clip_b32": "openai/clip-vit-base-patch32",
}


def _resolve_dim(config) -> int:
    """Embedding width of a dual-encoder config.

    CLIP-family configs expose `projection_dim`; SigLIP does not project, so
    its embedding width is the tower hidden size. Guessing wrong here silently
    builds an index that retrieval later rejects, so an unrecognised config
    raises rather than defaulting.
    """
    projection = getattr(config, "projection_dim", None)
    if projection is not None:
        return int(projection)

    for tower in ("text_config", "vision_config"):
        sub = getattr(config, tower, None)
        hidden = getattr(sub, "hidden_size", None)
        if hidden is not None:
            return int(hidden)

    raise AttributeError(
        f"cannot determine the embedding width of {type(config).__name__}: "
        f"no projection_dim and no text_config/vision_config hidden_size. "
        f"Add an explicit case here and a matching entry in "
        f"validate.ENCODER_DIMS.")


def _pooled(features):
    """The embedding tensor from whatever `get_*_features` returned.

    transformers 5.x returns the tower's `BaseModelOutputWithPooling` from
    both `SiglipModel` and `CLIPModel`; CLIP applies its projection *into*
    `pooler_output` and returns the output object rather than the tensor. 4.x
    returned a bare tensor. Both are accepted; anything else raises, because a
    wrong unwrap here builds a silently garbage index rather than failing.
    """
    pooled = getattr(features, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(features, "detach"):  # already a tensor (the 4.x contract)
        return features
    raise TypeError(
        f"cannot extract a pooled embedding from {type(features).__name__}: "
        f"no `pooler_output` and not a tensor. Check what this model's "
        f"get_image_features/get_text_features returns.")


class HFEncoder:
    """AutoModel-based dual encoder. Covers SigLIP and CLIP."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    @property
    def dim(self) -> int:
        return _resolve_dim(self.model.config)

    def encode_images(self, paths, batch_size: int = 32) -> np.ndarray:
        """Embed images from disk, decoding one batch at a time.

        Decoding is streamed rather than materialised up front: the LAION
        corpus is ~57k images averaging 0.61 MB decoded, so holding every
        frame before batching wants ~35 GB and the OOM killer takes the
        process before a single batch is encoded. Peak RAM is O(batch_size).
        """
        from PIL import Image

        paths = list(paths)
        if not paths:
            return self.encode_pil([], batch_size=batch_size)

        chunks = []
        for start in range(0, len(paths), batch_size):
            batch = []
            try:
                for p in paths[start:start + batch_size]:
                    # `with` closes the file handle; convert() already
                    # returned an independent in-memory copy.
                    with Image.open(p) as src:
                        batch.append(src.convert("RGB"))
                chunks.append(self.encode_pil(batch, batch_size=batch_size))
                if start % (batch_size * 20) == 0:
                    print(f"  [encode] {start + len(batch)}/{len(paths)}",
                          flush=True)
            finally:
                for im in batch:
                    im.close()
        return np.concatenate(chunks, axis=0)

    def encode_pil(self, images, batch_size: int = 32) -> np.ndarray:
        """Embed already-open PIL images.

        Crops never reach disk, so the prototype path needs an entry point
        that does not go through `Image.open`.
        """
        chunks = []
        images = list(images)
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[start:start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                feats = _pooled(self.model.get_image_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
            # Only when this call is doing the batching. encode_images hands
            # us exactly one batch at a time and reports corpus-level
            # progress itself; printing here too would emit "32/32" once per
            # batch and bury the real counter.
            if len(images) > batch_size and start % (batch_size * 20) == 0:
                print(f"  [encode] {start + len(batch)}/{len(images)}", flush=True)
        return np.concatenate(chunks, axis=0)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), padding="max_length",
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = _pooled(self.model.get_text_features(**inputs))
        return feats.float().cpu().numpy()

    def free(self) -> None:
        """Drop the model and return its VRAM.

        Without this, `stage_with_model`'s `getattr(model, "free", None)`
        cleanup silently no-ops (no AttributeError, just nothing released),
        so a retrieval-stage encoder stays GPU-resident straight through the
        next stage's VRAM preflight -- discovered when the "retrieve" stage
        left ~4 GB stranded before "mask"'s check on a 16 GB card.
        """
        self.model = None
        self.processor = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def build_encoder(name: str, device: str = "cuda") -> HFEncoder:
    """Build a retrieval encoder by short name.

    "fgclip" is a known key in ENCODER_IDS (the bake-off env resolves the repo
    id from it) but is refused here, mirroring models.build_crop_scorer: its
    remote code cannot load under this environment's transformers.
    """
    if name not in ENCODER_IDS:
        raise ValueError(
            f"unknown encoder '{name}'. Known: {', '.join(sorted(ENCODER_IDS))}")
    if name == "fgclip":
        raise RuntimeError(
            "fgclip cannot be used as a live retrieval encoder in this "
            "environment: its remote code targets transformers ~4.12 and "
            "FGCLIPConfig fails to build under 5.14.1. It is a bake-off-only "
            "candidate; run it from the separate fgclip env. "
            "See scripts/bakeoff.py.")
    return HFEncoder(ENCODER_IDS[name], device=device)
