"""The two repair mechanisms.

`stitch` pastes the reference's object into the hole and runs no model.
`Inpainter` conditions FLUX.1-Kontext on the reference and owns ~12 GB of
weights. The pilot picks one (parent spec §6); until it runs, neither is the
default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ragregen import composite

MECHANISMS = ("inpaint", "stitch")


@dataclass(frozen=True)
class RegenResult:
    image: Image.Image
    raw: Image.Image | None
    alpha: np.ndarray
    mechanism: str
    meta: dict = field(default_factory=dict)


def _mask_bounds(mask: Image.Image) -> tuple[np.ndarray, int, int, int, int]:
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(m)
    if xs.size == 0:
        raise ValueError("stitch: the mask is entirely black -- nothing to fill")
    return m, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def stitch(draft: Image.Image, mask: Image.Image, cutout: Image.Image, *,
           feather_px: int = 6, fill_residual: bool = True) -> RegenResult:
    """Paste the reference's object into the masked region. No diffusion.

    Fit-inside rather than fill: the object must never be cropped by its own
    hole. Placement is centred on the mask's centroid, which is where the
    object being replaced actually sat.
    """
    m, x0, x1, y0, y1 = _mask_bounds(mask)
    tw, th = x1 - x0 + 1, y1 - y0 + 1

    cw, ch = cutout.size
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cutout.convert("RGBA").resize((nw, nh), Image.LANCZOS)

    ys, xs = np.where(m)
    ox = int(round(xs.mean() - nw / 2))
    oy = int(round(ys.mean() - nh / 2))

    canvas = Image.new("RGBA", draft.size, (0, 0, 0, 0))
    canvas.paste(resized, (ox, oy), resized)

    pasted_alpha = Image.fromarray(np.asarray(canvas)[..., 3], "L")
    a = composite.feather_alpha(pasted_alpha, draft.size, feather_px)[..., None]

    d = np.array(draft.convert("RGB")).astype(np.float32)
    c = np.asarray(canvas.convert("RGB")).astype(np.float32)
    out = np.rint(d * (1 - a) + c * a).astype(np.uint8)

    filled = False
    if fill_residual:
        # mask minus what the paste actually covered. cv2.inpaint diffuses
        # surrounding colour inward: classical, milliseconds, and no diffusion
        # model -- which is the entire point of stitch being an arm.
        residual = m & (a[..., 0] <= 0.0)
        if residual.any():
            import cv2

            out = cv2.inpaint(out, residual.astype(np.uint8), 3,
                              cv2.INPAINT_TELEA)
            filled = True

    # Restore outside the MASK, not outside the paste. The residual sits
    # INSIDE the mask with alpha 0, so restoring by `a == 0` would overwrite
    # every pixel cv2.inpaint just filled and silently nullify the fill. `~m`
    # is also the guarantee as actually stated: pixels outside the mask are
    # bit-identical; inside it, we are free to paste and fill.
    out[~m] = np.array(draft.convert("RGB"))[~m]

    return RegenResult(image=Image.fromarray(out, "RGB"), raw=None,
                       alpha=a[..., 0], mechanism="stitch",
                       meta={"scale": round(scale, 4),
                             "placed_at": [ox, oy],
                             "cutout_size": [nw, nh],
                             "mask_box": [x0, y0, x1, y1],
                             "residual_filled": filled})


@dataclass(frozen=True)
class KontextConfig:
    """Everything tunable in one place, so the trace can record the whole state.

    Two defaults deviate from the port source, both deliberately:

    `dilate_px = 0` -- mask_draft already dilated by the operator's
    `mask_dilate_px`. Dilating again would grow the repaint region to ~24px of
    slack without saying so.

    `largest_cc = False` -- keeping only the biggest connected component
    discards most of a multi-piece mask. A plate of sushi is many pieces; that
    is a fact about the concept, not noise to clean up.
    """
    model_id: str = "black-forest-labs/FLUX.1-Kontext-dev"
    device: str = "cuda"
    quantize: str = "nf4"
    offload: str | None = None
    dilate_px: int = 0
    blur_factor: int = 12
    largest_cc: bool = False
    steps: int = 28
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    strength: float = 1.0
    max_area: int = 1024 ** 2
    seed: int = 0
    ref_prep: str = "crop"
    feather_px: int = 6


@dataclass
class PreparedInputs:
    prompt: str
    original: Image.Image
    image: Image.Image
    mask_binary: Image.Image
    mask_blurred: Image.Image
    reference: Image.Image | None
    notes: list = field(default_factory=list)


def _to_binary(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    m = mask.convert("L")
    if m.size != size:
        # NEAREST only: bilinear invents grey along the boundary.
        m = m.resize(size, Image.NEAREST)
    return np.array(m) > 127


def _dilate_binary(binary: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return binary
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(binary.astype(np.uint8), k, iterations=1) > 0


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2

    n, labels = cv2.connectedComponents(binary.astype(np.uint8))
    n_comp = n - 1                       # label 0 is background
    if n_comp <= 1:
        return binary, max(n_comp, 0)
    sizes = [int((labels == i).sum()) for i in range(1, n_comp + 1)]
    return labels == (int(np.argmax(sizes)) + 1), n_comp


def _blur(mask: Image.Image, blur_factor: int) -> Image.Image:
    from PIL import ImageFilter

    if blur_factor <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(blur_factor))


def validate_inputs(prompt: str, image, mask_image, image_reference=None,
                    cfg: KontextConfig | None = None) -> PreparedInputs:
    """Everything the pipeline call needs, validated.

    Order matters: binarise -> (largest component) -> dilate -> blur. The hard
    mask is kept alongside the blurred one because compositing and the
    preservation metric need a crisp boundary, not the soft one the model eats.
    """
    cfg = cfg or KontextConfig()
    notes: list = []

    if not (prompt or "").strip():
        raise ValueError("validate_inputs: empty prompt. Kontext is "
                         "instruction-driven and ignores an empty one.")

    image = image.convert("RGB")
    original = image.copy()

    binary = _to_binary(mask_image, image.size)
    frac = float(binary.mean())
    if frac == 0.0:
        raise ValueError("validate_inputs: mask is entirely black -- nothing "
                         "to edit. White = the region to edit.")
    if frac == 1.0:
        raise ValueError("validate_inputs: mask is entirely white -- the whole "
                         "frame would be regenerated. White = the region to edit.")
    if frac > 0.5:
        notes.append(f"WARNING: mask covers {frac:.0%} of the frame -- white "
                     f"must be the EDIT region; this looks inverted")

    if cfg.largest_cc:
        binary, n_comp = _largest_component(binary)
        if n_comp > 1:
            notes.append(f"mask had {n_comp} components; kept the largest")

    binary = _dilate_binary(binary, cfg.dilate_px)
    mask_binary = Image.fromarray((binary * 255).astype(np.uint8), "L")

    # Blur softens the seam the model renders. It preserves nothing -- that is
    # composite_back's job.
    mask_blurred = _blur(mask_binary, cfg.blur_factor)

    reference = (image_reference.convert("RGB")
                 if image_reference is not None else None)

    return PreparedInputs(prompt=prompt.strip(), original=original, image=image,
                          mask_binary=mask_binary, mask_blurred=mask_blurred,
                          reference=reference, notes=notes)


def prepare_reference(reference: Image.Image, concept: str, mode: str = "crop",
                      masker=None, pad_frac: float = 0.08
                      ) -> tuple[Image.Image, dict]:
    """Crop / matte the reference before it is encoded.

    The pipeline auto-resizes the reference to a ~1MP preferred resolution and
    encodes it to ~4k latent tokens, so subject scale in the reference frame
    buys token budget directly: a subject filling 10% of the frame gets 10% of
    the budget. Cropping to the subject is the cheap lever and is the default.

    `crop_matte` is offered but not default: the hard alpha edge it leaves is
    encoded by the VAE as a real object boundary. Which default is right is
    settled by the ref-prep ablation, not by assertion.

    `masker` is callable(image, phrase) -> (float mask HxW, info) -- exactly
    what doc 1's mask_reference produces.
    """
    info: dict = {"ref_prep": mode}
    if mode == "none" or masker is None:
        info["applied"] = "none"
        if mode != "none":
            info["reason"] = "no masker supplied"
        return reference, info

    found = masker(reference, concept)
    if found is None:
        info["applied"] = "none"
        info["reason"] = f"{concept!r} not grounded in the reference"
        return reference, info

    # ``mask.Masker`` returns MaskResult while the original ImageRAG helper
    # returned ``(mask, info)``.  Supporting both shapes makes this function
    # usable by the real pipeline instead of only by its synthetic unit tests.
    if hasattr(found, "mask"):
        m = found.mask
        minfo = {
            "box": list(found.box), "score": found.score,
            "n_candidates": found.n_candidates,
            "selected_by": found.selected_by,
        }
    else:
        m, minfo = found
    info["mask_info"] = {k: v for k, v in (minfo or {}).items() if k != "bbox"}
    ys, xs = np.where(np.asarray(m) > 0.5)
    if xs.size == 0:
        info["applied"] = "none"
        info["reason"] = f"{concept!r} not grounded in the reference"
        return reference, info

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pw = int((x1 - x0) * pad_frac) + 4
    ph = int((y1 - y0) * pad_frac) + 4
    box = (max(0, x0 - pw), max(0, y0 - ph),
           min(reference.width, x1 + pw + 1),
           min(reference.height, y1 + ph + 1))

    out = reference
    if mode == "crop_matte":
        # Neutral grey, not white or black: a white cut-out on a dark scene
        # reads to the model as a bright object, and it renders one.
        arr = np.array(reference.convert("RGB")).astype(np.float32)
        alpha = np.clip(np.asarray(m), 0, 1)[..., None]
        out = Image.fromarray(
            (arr * alpha + 127.0 * (1 - alpha)).astype(np.uint8), "RGB")

    out = out.crop(box)
    info["applied"] = mode
    info["bbox"] = list(box)
    info["subject_frac_before"] = round(
        float((np.asarray(m) > 0.5).mean()), 4)
    return out, info


def edit_prompt(original_prompt: str, coarse: str, concept: str) -> str:
    """Instruction that transfers identity without importing the ref scene."""
    return (
        f"Replace only the {coarse} with a {concept}. Preserve its pose, "
        f"position, scale, viewpoint, lighting and shadows, and preserve the "
        f"surrounding scene. Use the reference image only for the defining "
        f"visual identity of the {concept}; do not copy its background, text, "
        f"people, framing or artistic style. The result must still depict: "
        f"{original_prompt}."
    )


#: Below this, FLUX-nf4 does not fit beside another researcher's job and the
#: run thrashes or OOMs mid-generation. ../rag-edit died exactly that way.
MIN_VRAM_GB = 12.0


def free_vram_gb() -> float | None:
    """Free VRAM, or None when there is no CUDA device to ask."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def check_vram(free_gb: float | None = None) -> None:
    """Raise unless the card can hold the pipeline. Never lower this floor."""
    free_gb = free_vram_gb() if free_gb is None else free_gb
    if free_gb is None:
        return
    if free_gb < MIN_VRAM_GB:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free; FLUX-nf4 needs {MIN_VRAM_GB} GB. "
            f"Another job is holding the card -- wait rather than thrash. "
            f"`nvidia-smi` shows who.")


class Inpainter:
    """FLUX.1-Kontext, conditioned on a reference image.

    `pipe` is injectable so every path except the diffusion call itself is
    testable on CPU -- which matters because FLUX needs ~12 GB and the card is
    shared (design §7).
    """

    def __init__(self, cfg: KontextConfig | None = None, pipe=None):
        self.cfg = cfg or KontextConfig()
        self.pipe = pipe

    def load(self) -> "Inpainter":
        check_vram()

        import torch
        from diffusers import FluxKontextInpaintPipeline

        from ragregen import env

        cfg = self.cfg
        kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": str(env.HF_CACHE)}
        if cfg.quantize == "nf4":
            from diffusers import PipelineQuantizationConfig

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True,
                              "bnb_4bit_quant_type": "nf4",
                              "bnb_4bit_compute_dtype": torch.bfloat16,
                              "bnb_4bit_use_double_quant": True},
                # T5-XXL is 9.5GB in bf16 and is the difference between fitting
                # a 24GB card and not.
                components_to_quantize=["transformer", "text_encoder_2"])

        self.pipe = FluxKontextInpaintPipeline.from_pretrained(cfg.model_id,
                                                               **kwargs)
        self.pipe.set_progress_bar_config(disable=True)
        if cfg.offload == "on":
            self.pipe.enable_model_cpu_offload(device=cfg.device)
        else:
            self.pipe.to(cfg.device)
        # The VAE decode is the peak allocation on a 1MP frame and the first
        # thing to OOM on a shared card.
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()
        return self

    def regen(self, draft, mask, reference, prompt: str, *, seed=None,
              use_reference: bool = True, padding_mask_crop=None) -> RegenResult:
        if self.pipe is None:
            raise RuntimeError("Inpainter.load() has not been called")

        cfg = self.cfg
        ref = reference if use_reference else None
        if padding_mask_crop is not None and ref is not None:
            raise ValueError(
                "padding_mask_crop cannot be combined with a reference: the "
                "pipeline preprocesses the reference with the SOURCE's crop "
                "region, cropping it to an unrelated rectangle. Leave it None.")

        prepared = validate_inputs(prompt, draft, mask, ref, cfg)
        actual_seed = cfg.seed if seed is None else int(seed)
        import torch
        generator = torch.Generator(device="cpu").manual_seed(actual_seed)

        call = {"prompt": prepared.prompt, "image": prepared.image,
                "mask_image": prepared.mask_blurred, "image_reference": ref,
                "strength": cfg.strength, "num_inference_steps": cfg.steps,
                "guidance_scale": cfg.guidance_scale,
                "true_cfg_scale": cfg.true_cfg_scale, "max_area": cfg.max_area,
                "generator": generator}
        if padding_mask_crop is not None:
            call["padding_mask_crop"] = padding_mask_crop

        import time
        t0 = time.time()
        raw = self.pipe(**call).images[0]

        image, alpha = composite.composite_back(prepared.original, raw,
                                                prepared.mask_binary,
                                                cfg.feather_px)
        meta = {"seed": actual_seed,
                "steps": cfg.steps, "strength": cfg.strength,
                "guidance_scale": cfg.guidance_scale,
                "true_cfg_scale": cfg.true_cfg_scale,
                "prompt": prepared.prompt,
                "used_reference": ref is not None,
                "input_size": list(prepared.original.size),
                "raw_output_size": list(raw.size),
                "feather_px": cfg.feather_px,
                "notes": prepared.notes,
                "seconds": round(time.time() - t0, 1)}
        return RegenResult(image=image, raw=raw, alpha=alpha,
                           mechanism="inpaint", meta=meta)

    def free(self) -> None:
        from ragregen import env

        self.pipe = None
        env.reclaim_gpu()
