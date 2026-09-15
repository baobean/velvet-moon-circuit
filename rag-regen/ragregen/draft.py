"""Text-only drafting with FLUX.1-Kontext.

Loading the pipeline takes ~5m21s and one generation ~2 min at nf4 on the 4090
(measured: ../ImageRAG/results/kontext_example_20260721_224056/run.log). Load
once per process and draft every case before releasing the card -- never load
per case.
"""
from __future__ import annotations

from ragregen import env

env.setup()

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


def load_kontext_t2i(device: str = "cuda", quantize: str = "nf4"):
    """Load FluxKontextPipeline, nf4-quantised so it fits beside nothing else."""
    import torch
    from diffusers import FluxKontextPipeline

    kwargs = {"cache_dir": str(env.HF_CACHE), "torch_dtype": torch.bfloat16}
    if quantize == "nf4":
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True,
                          "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
    pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.to(device)
    return pipe


class Drafter:
    """Text-to-image drafting at a fixed seed.

    The generator is rebuilt per call rather than held on the instance: a
    generator advances as it is consumed, so a shared one would make the same
    case draft differently on a re-run within one process, and the gate's fail
    rate would stop being reproducible.
    """

    def __init__(self, pipe, steps: int = 28, seed: int = 0,
                 guidance: float = 3.5, device: str = "cuda"):
        self.pipe = pipe
        self.steps = steps
        self.seed = seed
        self.guidance = guidance
        self.device = device

    def draft(self, prompt: str):
        import torch

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        # Every argument is keyword: FluxKontextPipeline.__call__ is
        # (image, prompt, ...), so a positional prompt binds to `image` and
        # the pipeline rejects the call for having no prompt at all. `image`
        # stays unset -- this is the text-only draft.
        out = self.pipe(prompt=prompt, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=gen)
        return out.images[0]
