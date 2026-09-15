"""The only module that loads model weights.

Everything else (kg_build, generate, verify, refine, metrics, baselines)
receives models as constructor/function arguments typed against
graft.interfaces, so that logic is testable with in-memory fakes.

Idioms below are lifted verbatim from proven in-workspace code:
  - QwenVLM: rag-regen/ragregen/vlm.py (Qwen2.5-VL, nf4-quantized).
  - QwenReranker: rag-regen/scripts/qwen_reranker_score.py (CrossEncoder).
  - GroundingDinoDetector / SamSegmenter: rag-regen/ragregen/models.py.
  - SdxlIpGenerator: ImageRAG/imageRAG_SDXL.py (twin pipe_clean/pipe_ip).

One cache quirk this module works around: HF_HOME=env.HF_CACHE alone is not
enough here, because huggingface_hub's default lookup is $HF_HOME/hub, and
several models (siglip2, dinov3, the laion CLIP, grounding-dino, sam, SDXL)
are cached in the *flat* legacy layout directly under env.HF_CACHE, not under
its hub/ subdirectory. Every from_pretrained call below therefore also passes
cache_dir=str(env.HF_CACHE) explicitly (confirmed necessary by hand-testing
siglip2 and dinov3 loads against this exact cache).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from graft import env

env.setup()  # must precede transformers/diffusers import

VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
RERANKER_ID = "Qwen/Qwen3-VL-Reranker-2B"
SIGLIP_ID = "google/siglip2-base-patch16-384"
DINO_EMBED_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CLIP_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
DINO_DETECT_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-huge"
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"

RERANK_PROMPT = (
    "Score whether the document image depicts the exact fine-grained rare "
    "concept named in the query. Focus on identity-defining shape, texture, "
    "markings, and structure; ignore composition and background."
)

_DEFAULT_NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"


def _fit_image(image: Image.Image, max_pixels: int) -> Image.Image:
    """Bound multi-image vision tokens without changing aspect ratio."""
    image = image.convert("RGB")
    if image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (image.width * image.height))
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


class QwenVLM:
    """Implements interfaces.VLM. nf4-quantized: at bf16 the 7B weights need
    ~15.2GB on a card this session shares with other researchers."""

    def __init__(
        self,
        model_id: str = VLM_ID,
        device: str = "cuda",
        max_new_tokens: int = 512,
        quantize: str = "nf4",
        max_image_pixels: int | None = 1_000_000,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.do_sample = False
        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))

        kwargs = {"cache_dir": str(env.HF_CACHE), "torch_dtype": torch.bfloat16}
        if quantize == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = device

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model = (model if quantize == "nf4" else model.to(device)).eval()
        self._torch = torch

    def _chat(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        do_sample: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        images = list(images)
        if not images:
            raise ValueError("QwenVLM needs at least one image")
        if self.max_image_pixels is not None:
            images = [_fit_image(im, self.max_image_pixels) for im in images]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            images=images, text=[text], padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=self.do_sample if do_sample is None else do_sample,
                max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            )
        generated = [seq[inputs.input_ids.shape[1] :] for seq in out]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def describe(
        self,
        images: Sequence[Image.Image],
        instruction: str,
        *,
        do_sample: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        return self._chat(images, instruction, do_sample=do_sample, max_new_tokens=max_new_tokens)

    def ask(self, image: Image.Image, question: str) -> str:
        return self._chat([image], question)


class QwenReranker:
    """Implements interfaces.Reranker via Qwen3-VL-Reranker-2B as a
    CrossEncoder, mirroring rag-regen/scripts/qwen_reranker_score.py."""

    def __init__(self, model_id: str = RERANKER_ID, device: str = "cuda"):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_id,
            device=device,
            cache_folder=str(env.HF_CACHE),
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        )
        self._torch = torch

    def rank(self, query: str, images: Sequence[Image.Image]) -> List[float]:
        pairs = [(query, {"image": im}) for im in images]
        values = self.model.predict(
            pairs,
            prompt=RERANK_PROMPT,
            batch_size=1,
            activation_fn=self._torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        return [float(v) for v in values]


def _pooled(features):
    """Unwrap whatever get_image_features/get_text_features returned.

    transformers 5.x returns the tower's BaseModelOutputWithPooling from both
    SiglipModel and CLIPModel (CLIP applies its projection into
    pooler_output); older versions returned a bare tensor. Both are accepted;
    anything else raises rather than silently indexing garbage. Mirrors
    rag-regen/ragregen/encoders.py:_pooled.
    """
    pooled = getattr(features, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(features, "detach"):  # already a tensor
        return features
    raise TypeError(
        f"cannot extract a pooled embedding from {type(features).__name__}: "
        f"no pooler_output and not a tensor."
    )


class _HFDualEncoder:
    """Shared tail for transformers models exposing get_image_features and
    get_text_features (SigLIP2, laion CLIP)."""

    def __init__(self, model_cls, processor_cls, model_id: str, device: str):
        self.device = device
        self.processor = processor_cls.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))
        self.model = model_cls.from_pretrained(model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return x / norm

    def embed_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        images = [im.convert("RGB") for im in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = _pooled(self.model.get_image_features(**inputs))
        return self._normalize(feats.float().cpu().numpy())

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        inputs = self.processor(
            text=list(texts), padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            feats = _pooled(self.model.get_text_features(**inputs))
        return self._normalize(feats.float().cpu().numpy())


class SiglipEmbedder(_HFDualEncoder):
    def __init__(self, model_id: str = SIGLIP_ID, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor

        super().__init__(AutoModel, AutoProcessor, model_id, device)


class ClipEmbedder(_HFDualEncoder):
    def __init__(self, model_id: str = CLIP_ID, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        super().__init__(CLIPModel, CLIPProcessor, model_id, device)


class DinoEmbedder:
    """Implements interfaces.Embedder for DINOv3. Vision-only: embed_text
    raises, since DINOv3 has no text tower."""

    def __init__(self, model_id: str = DINO_EMBED_ID, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()

    def embed_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        images = [im.convert("RGB") for im in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        feats = out.pooler_output.float().cpu().numpy()
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return feats / norm

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("DINOv3 has no text tower")


class GroundingDinoDetector:
    """Implements interfaces.Detector."""

    def __init__(
        self,
        model_id: str = DINO_DETECT_ID,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))
        self.model = (
            GroundingDinoForObjectDetection.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))
            .to(device)
            .eval()
        )
        self._torch = torch

    def detect(
        self, image: Image.Image, phrase: str
    ) -> List[Tuple[float, float, float, float]]:
        text = phrase if phrase.endswith(".") else phrase + "."
        inputs = self.processor(images=image, text=text.lower(), return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        pairs = sorted(
            zip(results["boxes"], results["scores"]), key=lambda p: float(p[1]), reverse=True
        )
        return [tuple(float(v) for v in box) for box, _ in pairs]


class SamSegmenter:
    """Implements interfaces.Segmenter. Ported from
    rag-regen/ragregen/models.py:SamSegmenter."""

    def __init__(self, model_id: str = SAM_ID, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, SamModel

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(env.HF_CACHE))
        self.model = SamModel.from_pretrained(model_id, cache_dir=str(env.HF_CACHE)).to(device).eval()
        self._torch = torch

    def mask(self, image: Image.Image, box: Tuple[float, float, float, float]) -> np.ndarray:
        inputs = self.processor(image, input_boxes=[[list(box)]], return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return np.asarray(masks[0][0][0], dtype=bool)


class SdxlIpGenerator:
    """Implements interfaces.Generator. Mirrors ImageRAG/imageRAG_SDXL.py's
    twin-pipeline pattern: a text-only pipe and an IP-Adapter pipe, each
    lazily constructed the first time its mode is actually used."""

    def __init__(
        self,
        sdxl_id: str = SDXL_ID,
        ip_adapter_repo: str = IP_ADAPTER_REPO,
        ip_adapter_weight: str = IP_ADAPTER_WEIGHT,
        device: str = "cuda",
        ip_scale: float = 0.6,
    ):
        self.sdxl_id = sdxl_id
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_weight = ip_adapter_weight
        self.device = device
        self.ip_scale = ip_scale
        self._pipe_clean = None
        self._pipe_ip = None
        self._pipe_inpaint = None
        self._torch = None

    def _image_encoder(self):
        import torch
        from transformers import CLIPVisionModelWithProjection

        return CLIPVisionModelWithProjection.from_pretrained(
            self.ip_adapter_repo,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            cache_dir=str(env.HF_CACHE),
        )

    @property
    def pipe_clean(self):
        if self._pipe_clean is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            self._torch = torch
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.sdxl_id,
                image_encoder=self._image_encoder(),
                torch_dtype=torch.float16,
                cache_dir=str(env.HF_CACHE),
            ).to(self.device)
            # This card is shared and routinely down to ~9GB free under other
            # researchers' jobs (see scripts/run_remaining_phase1.sh's comment).
            # VAE slicing removes the decode-time memory spike so a generate()
            # still fits; output is unchanged. (Attention slicing is NOT used --
            # its SlicedAttnProcessor is incompatible with IP-Adapter's custom
            # attention processors on pipe_ip.)
            pipe.vae.enable_slicing()
            self._pipe_clean = pipe
        return self._pipe_clean

    @property
    def pipe_ip(self):
        if self._pipe_ip is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            self._torch = torch
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.sdxl_id,
                image_encoder=self._image_encoder(),
                torch_dtype=torch.float16,
                cache_dir=str(env.HF_CACHE),
            ).to(self.device)
            pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder="sdxl_models",
                weight_name=self.ip_adapter_weight,
                cache_dir=str(env.HF_CACHE),
            )
            pipe.set_ip_adapter_scale(self.ip_scale)
            # See pipe_clean: shared card, keep the generate() memory peak low.
            # VAE slicing only -- attention slicing breaks the IP-Adapter path.
            pipe.vae.enable_slicing()
            self._pipe_ip = pipe
        return self._pipe_ip

    def generate(
        self,
        prompt: str,
        ip_image: Image.Image | None,
        seed: int,
        negative: str = _DEFAULT_NEGATIVE,
        steps: int = 50,
    ) -> Image.Image:
        if ip_image is None:
            pipe = self.pipe_clean
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
            return pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        pipe = self.pipe_ip
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            ip_adapter_image=ip_image,
            negative_prompt=negative,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    def generate_multi(
        self,
        prompt: str,
        ref_images: list[Image.Image],
        weights: Sequence[float],
        seed: int,
        negative: str = _DEFAULT_NEGATIVE,
        steps: int = 50,
    ) -> Image.Image:
        """Condition on the weighted average of each ref image's IP-Adapter
        embed. weights align to ref_images and sum to 1 (see transfer.reference_weights)."""
        pipe = self.pipe_ip
        # no_grad: prepare_ip_adapter_image_embeds runs the CLIP image encoder OUTSIDE the
        # pipe's own @torch.no_grad(), so without this each call builds an autograd graph
        # whose activations accumulate across a resident generation loop -> ~0.5GB/gen VRAM
        # leak -> OOM after ~25 cells. Inference output is unchanged.
        with self._torch.no_grad():
            per_image = []
            for img in ref_images:
                e = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[[img]],    # outer len 1 == the single loaded adapter
                    ip_adapter_image_embeds=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                per_image.append(e[0])           # single adapter -> list of length 1
            # weighted average across reference images (weights align to ref_images, sum to 1).
            # Plain weighted sum of tensors is shape-agnostic (no manual broadcast/view needed).
            avg = sum(w * emb for w, emb in zip(weights, per_image))
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
            return pipe(
                prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
                ip_adapter_image_embeds=[avg], generator=generator,
            ).images[0]

    def set_scale(self, scale: float) -> None:
        """Change the IP-Adapter scale without rebuilding the pipe. Safe before
        pipe_ip is loaded (records it; pipe_ip's constructor applies it)."""
        self.ip_scale = scale
        if self._pipe_ip is not None:
            self._pipe_ip.set_ip_adapter_scale(scale)
        if getattr(self, "_pipe_inpaint", None) is not None:
            self._pipe_inpaint.set_ip_adapter_scale(scale)

    @property
    def pipe_inpaint(self):
        if self._pipe_inpaint is None:
            import torch
            from diffusers import AutoPipelineForInpainting

            self._torch = torch
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.sdxl_id,
                image_encoder=self._image_encoder(),
                torch_dtype=torch.float16,
                cache_dir=str(env.HF_CACHE),
            ).to(self.device)
            pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder="sdxl_models",
                weight_name=self.ip_adapter_weight,
                cache_dir=str(env.HF_CACHE),
            )
            pipe.set_ip_adapter_scale(self.ip_scale)
            pipe.vae.enable_slicing()               # shared card; keep decode peak low
            self._pipe_inpaint = pipe
        return self._pipe_inpaint

    def inpaint_multi(self, prompt, base_image, mask_image, ref_images, weights, seed,
                      negative=_DEFAULT_NEGATIVE, steps=50, strength=0.99, guidance=7.5):
        """Inpaint the masked region conditioned on the weighted average of each ref
        crop's IP-Adapter embed. Same no_grad discipline as generate_multi (avoids the
        ~0.5GB/gen CLIP-image-encoder VRAM leak under a resident loop)."""
        pipe = self.pipe_inpaint
        with self._torch.no_grad():
            per_image = []
            for img in ref_images:
                e = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[[img]], ip_adapter_image_embeds=None,
                    device=self.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                per_image.append(e[0])
            avg = sum(w * emb for w, emb in zip(weights, per_image))
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
            return pipe(
                prompt=prompt, image=base_image, mask_image=mask_image,
                negative_prompt=negative, num_inference_steps=steps,
                strength=strength, guidance_scale=guidance,
                ip_adapter_image_embeds=[avg], generator=generator,
            ).images[0]


class Models:
    """Lazy-singleton facade over every concrete loader, so pipeline code
    only ever says `models.vlm`, `models.siglip`, etc., and can `unload`
    a stage's models before the next stage claims the GPU (spec's
    single-GPU-sequential-load requirement)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cache: dict = {}

    def _get(self, name: str, factory):
        if name not in self._cache:
            self._cache[name] = factory()
        return self._cache[name]

    @property
    def vlm(self) -> QwenVLM:
        return self._get("vlm", lambda: QwenVLM(self.cfg.vlm_id, self.cfg.device))

    @property
    def reranker(self) -> QwenReranker:
        return self._get(
            "reranker", lambda: QwenReranker(self.cfg.reranker_id, self.cfg.device)
        )

    @property
    def siglip(self) -> SiglipEmbedder:
        return self._get(
            "siglip", lambda: SiglipEmbedder(self.cfg.siglip_id, self.cfg.device)
        )

    @property
    def dino(self) -> DinoEmbedder:
        return self._get("dino", lambda: DinoEmbedder(self.cfg.dino_id, self.cfg.device))

    @property
    def clip(self) -> ClipEmbedder:
        return self._get("clip", lambda: ClipEmbedder(self.cfg.clip_id, self.cfg.device))

    @property
    def detector(self) -> GroundingDinoDetector:
        return self._get(
            "detector",
            lambda: GroundingDinoDetector(self.cfg.dino_detector_id, self.cfg.device),
        )

    @property
    def segmenter(self) -> SamSegmenter:
        return self._get("segmenter", lambda: SamSegmenter(self.cfg.sam_id, self.cfg.device))

    @property
    def generator(self) -> SdxlIpGenerator:
        return self._get(
            "generator",
            lambda: SdxlIpGenerator(
                self.cfg.sdxl_id,
                self.cfg.ip_adapter_repo,
                self.cfg.ip_adapter_weight,
                self.cfg.device,
                self.cfg.ip_scale,
            ),
        )

    def unload(self, name: str) -> None:
        self._cache.pop(name, None)
        env.free_gpu()

    def unload_all(self) -> None:
        """Drop every cached model. The spec's single-GPU-sequential
        constraint (models load sequentially, released between stages) is
        not automatic: SDXL's twin pipelines alone run ~14GB fp16, and
        VLM+reranker+siglip+dino+detector together run ~13-14GB more --
        concurrently that overflows this card's 24GB (confirmed: caused a
        CUDA segfault in the M2/M3 generate<->verify alternation before this
        was called at each phase boundary). Callers that alternate between
        generation and verification/metrics phases must call this between
        phases rather than trust individual unload() calls to catch every
        stale model."""
        self._cache.clear()
        env.free_gpu()
