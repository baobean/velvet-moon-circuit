"""Qwen2.5-VL adapter for Stream B.

Qwen2.5-VL rather than Qwen3-VL on purpose: this checkpoint is already cached
(16 GB at .cache/models--Qwen--Qwen2.5-VL-7B-Instruct), and Qwen3-VL is held
back so it can serve as the held-out judge the design asks for (parent spec
§5). A judge used inside the loop cannot also grade it.

Generation is greedy. A sampling judge would return a different verdict on a
re-run, and the C1 numbers would not reproduce.

Every processor argument is passed by keyword. `Qwen2_5_VLProcessor.__call__`
is `(images, text, videos, audio, **kwargs)` -- images FIRST, the same shape
that made a positional call to FluxKontextPipeline fail in Plan 1.

Quantised to nf4 by default. At bf16 the 7B weights need ~15.2 GB, and this
card is shared -- a measured attempt died with OutOfMemoryError at 14.04 GB
while two other researchers' processes held 9.4 GB between them. nf4 brings it
to roughly 5 GB, so Stream B runs on a busy card instead of waiting for one.
This mirrors what draft.load_kontext_t2i already does for FLUX.
"""
from __future__ import annotations

import math

from ragregen import env

env.setup()  # must precede transformers import

VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLM:
    """Implements verify.semantic.VLM: ask(image, prompt) -> str."""

    def __init__(self, model_id: str = VLM_ID, device: str = "cuda",
                 max_new_tokens: int = 512, quantize: str = "nf4",
                 max_image_pixels: int | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(env.HF_CACHE))

        kwargs = {"cache_dir": str(env.HF_CACHE), "torch_dtype": torch.bfloat16}
        if quantize == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # bitsandbytes places the weights itself; .to(device) would move an
            # already-quantised model and raise.
            kwargs["device_map"] = device

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **kwargs)
        self.model = (model if quantize == "nf4" else model.to(device)).eval()
        self._torch = torch

    def ask(self, image, prompt: str) -> str:
        return self.ask_images([image], prompt)

    def ask_images(self, images, prompt: str) -> str:
        """Ask about an ordered set of images in one deterministic turn."""
        images = list(images)
        if not images:
            raise ValueError("QwenVLM.ask_images needs at least one image")
        if self.max_image_pixels is not None:
            images = [_fit_image(image, self.max_image_pixels)
                      for image in images]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=images, text=[text], padding=True,
                                return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=self.max_new_tokens)

        # Strip the prompt tokens; only the continuation is the reply.
        generated = [seq[inputs.input_ids.shape[1]:] for seq in out]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True)[0].strip()


def _fit_image(image, max_pixels: int):
    """Bound multi-image vision tokens without changing aspect ratio."""
    if max_pixels < 1:
        raise ValueError("max_image_pixels must be >= 1")
    image = image.convert("RGB")
    if image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (image.width * image.height))
    size = (max(1, int(image.width * scale)),
            max(1, int(image.height * scale)))
    from PIL import Image

    return image.resize(size, Image.Resampling.LANCZOS)
