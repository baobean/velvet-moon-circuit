"""Evaluation metrics. Doc 4.

Pure functions plus the three eval encoders. Nothing here runs inside the
repair loop -- selection uses the verifier only, and selecting on these would
be selecting on the thing being measured (design §5).

Ported from ImageRAG/scripts/kontext_metrics.py so numbers stay comparable
between the two projects' runs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop_to_mask(image: Image.Image, mask, pad_frac: float = 0.15,
                 min_side: int = 64) -> Image.Image:
    """Crop to the mask bbox + padding. Full frame if degenerate.

    Identity is a property of the object, not the canvas: a rare parrot in a
    wide scene scores against its own background unless the crop is taken
    (findings/2026-07-27-finegrained-result.md §3a).
    """
    a = (np.array(mask.convert("L")) if isinstance(mask, Image.Image)
         else np.asarray(mask))
    thr = 127 if a.dtype == np.uint8 else 0.5
    ys, xs = np.where(a > thr)
    if xs.size == 0:
        return image
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pw = int((x1 - x0) * pad_frac) + 8
    ph = int((y1 - y0) * pad_frac) + 8
    crop = image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(image.width, x1 + pw), min(image.height, y1 + ph)))
    if min(crop.size) < min_side:
        return image
    return crop


def split_refs(gt_refs, heldout: int = 1):
    """(edit pool, held-out) -- the split that makes DINO reportable.

    The oracle arm regenerates using its references and DINO scores the output
    against them. Scoring an edit with the image it was handed is self-marking,
    the same objection RUNBOOK line 247 makes about --proto-arm ceiling. The
    last `heldout` references are reserved for scoring and never edited with.
    """
    refs = list(gt_refs)
    if len(refs) - heldout < 1:
        raise ValueError(
            f"{len(refs)} refs with heldout={heldout} leaves nothing to edit "
            f"with. Add a reference or lower eval.heldout_refs.")
    return refs[:-heldout], refs[-heldout:]


def preservation(draft: Image.Image, output: Image.Image,
                 alpha: np.ndarray) -> dict:
    """How much of the region outside the mask survived, 0-1, higher better.

    `score` is 1 - mean|diff|/255 over the alpha==0 zone. For a composite it
    must be EXACTLY 1.0 -- pixels outside the mask are bit-identical by
    construction, and anything less means the compositor changed behaviour.

    Always read next to DINO: a method that improves identity by repainting
    the whole canvas has not solved the problem (RUNBOOK §5.2).
    """
    if output.size != draft.size:
        raise ValueError(
            f"output size {output.size} != draft size {draft.size}. All "
            f"metrics compute at one pinned resolution; resizing here would "
            f"show resampling artifacts as real deltas.")

    o = np.array(draft.convert("RGB")).astype(np.float32)
    r = np.array(output.convert("RGB")).astype(np.float32)

    outside = alpha == 0.0
    if not outside.any():
        return {"score": float("nan"), "outside_l1": float("nan"),
                "outside_max": float("nan"), "outside_frac": 0.0}

    d = np.abs(o - r)[outside]
    return {
        "score": float(1.0 - d.mean() / 255.0),
        "outside_l1": float(d.mean()),
        "outside_max": float(d.max()),
        "outside_frac": float(outside.mean()),
    }


class DinoEncoder:
    """DINOv3. Instance-level identity -- the headline metric.

    Loaded lazily so importing this module costs nothing on a busy card.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.proc = None

    def _load(self):
        from transformers import AutoImageProcessor, AutoModel

        from ragregen import env
        self.proc = AutoImageProcessor.from_pretrained(
            self.model_id, cache_dir=str(env.HF_CACHE))
        self.model = AutoModel.from_pretrained(
            self.model_id, cache_dir=str(env.HF_CACHE)).to(self.device).eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        import torch

        if self.model is None:
            self._load()
        inputs = self.proc(images=image.convert("RGB"),
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # pooler_output is the CLS token; mean-pool the patches when a
        # checkpoint exposes no pooler.
        feats = (out.pooler_output if getattr(out, "pooler_output", None)
                 is not None else out.last_hidden_state.mean(dim=1))
        v = feats.float().cpu().numpy()[0]
        return v / (np.linalg.norm(v) + 1e-12)

    def free(self):
        import torch

        self.model = None
        self.proc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dino_identity(output_crop: Image.Image, ref_crops, encoder) -> float:
    """Mean cosine between the output crop and each HELD-OUT reference crop.

    The references here must be the ones the repair never saw. Passing the
    edit pool makes this self-marking (spec §2).
    """
    refs = list(ref_crops)
    if not refs:
        raise ValueError(
            "no held-out references to score against; check "
            "heldout_refs.json for this case")
    o = encoder.embed(output_crop)
    return float(np.mean([float(o @ encoder.embed(r)) for r in refs]))


def prompt_alignment(image: Image.Image, prompt: str, encoder) -> float:
    """Cosine between the image and its prompt, for CLIP and SigLIP alike.

    Context, never the claim: prompt alignment cannot tell an Amur leopard
    from a generic one, which is the entire premise of the project. A flat
    CLIP delta beside a clear DINO gain is a success (RUNBOOK §5.2).
    """
    iv = np.asarray(encoder.encode_pil([image]), dtype=np.float32)[0]
    tv = np.asarray(encoder.encode_text([prompt]), dtype=np.float32)[0]
    iv = iv / (np.linalg.norm(iv) + 1e-12)
    tv = tv / (np.linalg.norm(tv) + 1e-12)
    return float(iv @ tv)


class EvalEncoders:
    """The three eval encoders, loaded once per report run and freed together."""

    def __init__(self, dino=None, clip=None, siglip=None):
        self.dino = dino
        self.clip = clip
        self.siglip = siglip

    @classmethod
    def load(cls, pipe_cfg, device: str = "cuda") -> "EvalEncoders":
        from ragregen import encoders

        return cls(
            dino=DinoEncoder(pipe_cfg.eval_dino, device),
            clip=encoders.HFEncoder(pipe_cfg.eval_clip, device),
            siglip=encoders.HFEncoder(pipe_cfg.eval_siglip, device),
        )

    def free(self):
        for enc in (self.dino, self.clip, self.siglip):
            free = getattr(enc, "free", None)
            if callable(free):
                free()


#: The non-inferiority margin, in DINO cosine, fixed by doc 5 §2 before any
#: data existed. Doc 4's measured cropped DINO was 0.626, so 0.02 is ~3% of
#: the operating point: below it a drop is not distinguishable from crop and
#: reference noise. Choosing this after seeing deltas would make the test
#: meaningless -- do not tune it.
MARGIN = -0.02

#: The smallest paired sample that gets an interval at all. A bootstrap over
#: fewer than five paired observations reports precision it does not have:
#: every resample of a one-case stratum is that same case, so the interval
#: collapses to zero width and prints as "+0.010 [+0.010, +0.010] no_harm" --
#: extraordinary certainty that is an artifact of resampling a degenerate
#: sample, not a measurement. With 11 bridge controls mostly expected healthy
#: a 1-2 case repair stratum is a likely outcome, not a corner case, so the
#: floor is enforced here rather than left to whoever reads the table. Below
#: it there is no interval: the report renders "—" and no verdict.
MIN_CI_N = 5


def paired_bootstrap_ci(deltas, *, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of per-case deltas.

    Bootstrap rather than a t-test: n is well under 100 and the deltas are
    likely skewed, so no normality is assumed. Seeded, because a headline that
    moves between report runs is not a headline. None below MIN_CI_N paired
    observations -- see that constant.
    """
    vals = np.asarray([d for d in deltas if d is not None], dtype=np.float64)
    if vals.size < MIN_CI_N:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def delta_verdict(lo: float, hi: float) -> str:
    """Which of doc 5 §2's three readings this interval supports.

    The boundary lives here rather than in whoever reads the table.
    """
    if lo >= MARGIN:
        return "no_harm"
    if hi < 0:
        return "harm"
    return "inconclusive"
