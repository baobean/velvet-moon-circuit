"""The only module that loads weights.

Everything else receives models as constructor arguments, so the logic layer
tests on CPU with fakes. Weights resolve from env.HF_CACHE.

Two crop scorers exist because they load from genuinely different HF repos:

- SigLIP (`google/siglip-so400m-patch14-384`) is a native transformers
  ``SiglipModel``: a single ``AutoProcessor`` handles both text and image,
  and ``logits_per_image`` is designed to be squashed with a plain sigmoid
  (that's the point of SigLIP's sigmoid loss). This is the live crop scorer
  in this environment.
- FG-CLIP (`qihoo360/fg-clip-base`) ships a custom `modeling_fgclip.py`
  vendoring a transformers-~4.12-era CLIP implementation, which does not
  load under the transformers==5.14.1 this environment pins for Qwen3-VL and
  FLUX-Kontext (confirmed: even past its `AutoModelForCausalLM` +
  `trust_remote_code=True` loading contract and its `einops` dependency,
  `FGCLIPConfig` construction fails -- `config.text_config` arrives as a
  plain `dict` instead of `CLIPTextConfig`). FG-CLIP is therefore a
  bake-off-only candidate, loaded from a separate conda env (see Task 13).
  `build_crop_scorer("fgclip")` refuses to attempt the load here rather than
  fail deep inside model construction -- see its docstring.
  `FGCLIPScorer` and `SCORER_IDS["fgclip"]` are kept as the reference
  implementation and the correct repo id for that separate env to use.

`_SigmoidTextImageScorer` factors out the shared last step (raw logit ->
sigmoid -> float in [0, 1]) while leaving model loading and logit
computation to the subclasses, since those genuinely differ.
"""
from __future__ import annotations

from ragregen import env

env.setup()  # must precede transformers import

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-huge"

SCORER_IDS: dict[str, str] = {
    # NOTE: the brief's `timm/ViT-SO400M-14-SigLIP-384` is an open_clip-format
    # repo (open_clip_config.json / open_clip_pytorch_model.bin, no
    # config.json) -- it cannot be loaded through transformers'
    # AutoModel/AutoProcessor at all. google/siglip-so400m-patch14-384 is the
    # native transformers checkpoint of the same weights (SiglipModel,
    # hidden_size 1152, matching validate.ENCODER_DIMS["siglip_so400m_384"]).
    "siglip_so400m_384": "google/siglip-so400m-patch14-384",
    "fgclip": "qihoo360/fg-clip-base",
}


class DinoDetector:
    """GroundingDINO box proposals for a text phrase."""

    def __init__(self, model_id: str = DINO_ID, device: str = "cuda",
                 box_threshold: float = 0.25, text_threshold: float = 0.25):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    def all_boxes(self, image, phrase: str, max_boxes: int = 8):
        text = phrase if phrase.endswith(".") else phrase + "."
        inputs = self.processor(images=image, text=text.lower(),
                                 return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        pairs = [(tuple(float(v) for v in box), float(score))
                 for box, score in zip(results["boxes"], results["scores"])]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:max_boxes]


class _SigmoidTextImageScorer:
    """Shared tail: turn a raw text/image logit into a score in [0, 1].

    Subclasses own model loading (`_load`) and logit computation (`_logit`)
    since those differ between SigLIP and FG-CLIP; this base class only
    guarantees the CropScorer contract is met regardless of the raw logit's
    scale or sign.
    """

    def __init__(self, model_id: str, device: str):
        import torch

        self.device = device
        self._torch = torch
        self._load(model_id, device)

    def _load(self, model_id: str, device: str) -> None:
        raise NotImplementedError

    def _logit(self, crop, phrase: str) -> float:
        raise NotImplementedError

    def score(self, crop, phrase: str) -> float:
        logit = self._logit(crop, phrase)
        return float(self._torch.sigmoid(self._torch.tensor(logit)))


class SigLIPScorer(_SigmoidTextImageScorer):
    def __init__(self, model_id: str = SCORER_IDS["siglip_so400m_384"],
                 device: str = "cuda"):
        super().__init__(model_id, device)

    def _load(self, model_id: str, device: str) -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()

    def _logit(self, crop, phrase: str) -> float:
        inputs = self.processor(text=[phrase], images=crop, padding="max_length",
                                 return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs)
        return float(out.logits_per_image[0][0])

    @property
    def dim(self) -> int:
        from ragregen.encoders import _resolve_dim

        return _resolve_dim(self.model.config)

    def encode_pil(self, images, batch_size: int = 32):
        """Reuse this resident SigLIP tower for image prototypes."""
        import numpy as np
        from ragregen.encoders import _pooled

        images = list(images)
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB")
                     for im in images[start:start + batch_size]]
            inputs = self.processor(images=batch,
                                    return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                feats = _pooled(self.model.get_image_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def encode_text(self, texts: list[str]):
        """Reuse this resident SigLIP tower for retrieval queries."""
        from ragregen.encoders import _pooled

        inputs = self.processor(text=list(texts), padding="max_length",
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = _pooled(self.model.get_text_features(**inputs))
        return feats.float().cpu().numpy()


class FGCLIPScorer(_SigmoidTextImageScorer):
    """FG-CLIP (`qihoo360/fg-clip-base`): custom remote code, its own
    AutoModelForCausalLM registration, and separate tokenizer/image
    processor -- see the module docstring.

    NOT constructible in this environment: `build_crop_scorer` refuses to
    dispatch to this class here (see its docstring). This class is kept as
    the reference implementation for the separate bake-off env (Task 13),
    where transformers is pinned to a version FG-CLIP's vendored code
    actually supports. It is not wired around the `transformers.onnx`
    import-time incompatibility documented in Task 7's report, since that
    workaround is unreachable -- and therefore dead -- as long as
    `build_crop_scorer` blocks "fgclip" here; the bake-off env's own
    transformers version may not need it at all.
    """

    _IMAGE_SIZE = 224

    def __init__(self, model_id: str = SCORER_IDS["fgclip"], device: str = "cuda"):
        super().__init__(model_id, device)

    def _load(self, model_id: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE), trust_remote_code=True
        ).to(device).eval()

    def _logit(self, crop, phrase: str) -> float:
        image = crop.convert("RGB").resize((self._IMAGE_SIZE, self._IMAGE_SIZE))
        image_input = self.image_processor.preprocess(
            image, return_tensors="pt")["pixel_values"].to(self.device)
        caption_input = self._torch.tensor(
            self.tokenizer([phrase], max_length=77, padding="max_length",
                            truncation=True).input_ids,
            dtype=self._torch.long, device=self.device)
        with self._torch.no_grad():
            image_feature = self.model.get_image_features(image_input)
            text_feature = self.model.get_text_features(
                caption_input, walk_short_pos=True)
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
            logit = self.model.logit_scale.exp() * (image_feature @ text_feature.T)
        return float(logit[0][0])


def build_crop_scorer(name: str, device: str = "cuda"):
    """Construct a CropScorer by name.

    "fgclip" is a known key in SCORER_IDS (the bake-off env resolves the repo
    id from it) but is refused here rather than attempted: its remote code
    cannot load under this environment's pinned transformers, and letting the
    attempt proceed would fail deep inside model construction with a message
    that doesn't explain why -- see the module docstring.
    """
    if name not in SCORER_IDS:
        raise ValueError(
            f"unknown crop scorer '{name}'. Known: {', '.join(sorted(SCORER_IDS))}")
    if name == "fgclip":
        raise RuntimeError(
            "fgclip cannot be used as the live crop scorer in this environment: "
            "its remote code requires transformers ~4.12, incompatible with the "
            "5.14.1 pinned here for Qwen3-VL and FLUX-Kontext. FG-CLIP is a "
            "bake-off-only candidate; run it from the separate fgclip env. "
            "Set crop_scorer: siglip_so400m_384 in configs/pipeline.yaml."
        )
    cls = {"siglip_so400m_384": SigLIPScorer, "fgclip": FGCLIPScorer}[name]
    return cls(device=device)


class SamSegmenter:
    """SAM box -> binary mask.

    Ported from ../rag-edit/ragedit/mask.py:74-87. SAM ships natively in
    transformers, so this is pure PyTorch with no CUDA extension build -- the
    usual reason a GroundingDINO/SAM install eats a day.
    """

    def __init__(self, model_id: str = SAM_ID, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, SamModel

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))
        self.model = SamModel.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    def mask_from_box(self, image, box):
        import numpy as np

        inputs = self.processor(image, input_boxes=[[list(box)]],
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        return np.asarray(masks[0][0][0], dtype=bool)

    def to_cpu(self) -> None:
        self.model.to("cpu")
        env.reclaim_gpu()

    def free(self) -> None:
        del self.model, self.processor
        env.reclaim_gpu()
