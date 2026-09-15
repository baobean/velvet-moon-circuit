"""Pure pixel work shared by both regeneration mechanisms.

No models, no torch, no network. Everything here runs on CPU in a unit test,
which is what lets doc 2 be validated while FLUX cannot fit on the shared card
(specs/2026-07-28-regeneration-design.md §7).

Ported from ../ImageRAG/scripts/kontext_engine.py:473-518. The arithmetic is
load-bearing and reproduced exactly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def feather_alpha(mask_binary: Image.Image, size: tuple[int, int],
                  feather_px: int = 6) -> np.ndarray:
    """Hard mask -> float alpha in [0, 1] at `size`.

    Three zones fall out and the preservation metric depends on all three:
      outside  alpha == 0   must stay bit-identical to the original
      band     0 < alpha < 1
      inside   alpha == 1
    """
    m = mask_binary.convert("L")
    if m.size != size:
        # NEAREST only: bilinear invents grey values along the boundary, which
        # would put pixels in the feather band that the mask never marked.
        m = m.resize(size, Image.NEAREST)
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_px))
    return np.array(m).astype(np.float32) / 255.0


def composite_back(original: Image.Image, raw: Image.Image,
                   mask_binary: Image.Image, feather_px: int = 6
                   ) -> tuple[Image.Image, np.ndarray]:
    """Paste the edited region onto the untouched original, feathered.

    Mandatory, not cosmetic: FLUX blends in latent space at 1/8 resolution and
    then VAE-decodes the whole frame, so every unmasked pixel has already
    changed by the time we see `raw`. This is the collateral-damage guarantee.

    Also absorbs the resolution change -- `raw` comes back at a preferred
    Kontext resolution, which is almost never the original's size.
    """
    if raw.size != original.size:
        raw = raw.resize(original.size, Image.LANCZOS)

    o = np.array(original.convert("RGB")).astype(np.float32)
    r = np.array(raw.convert("RGB")).astype(np.float32)
    a = feather_alpha(mask_binary, original.size, feather_px)[..., None]

    out = np.rint(o * (1 - a) + r * a).astype(np.uint8)

    # Belt and braces: force exact equality outside the feather. The float
    # blend already gives this, but the preservation metric asserts == 0.0
    # exactly and should be testing the compositing rule, not float rounding.
    outside = a[..., 0] == 0.0
    out[outside] = np.array(original.convert("RGB"))[outside]

    return Image.fromarray(out, "RGB"), a[..., 0]


def cutout_from(reference: Image.Image, mask_result) -> Image.Image | None:
    """The reference's object as RGBA, cropped to its bounding box.

    `stitch` needs pixels plus a silhouette, not a MaskResult -- keeping the
    conversion here means stitch can be tested on synthetic arrays with no
    masker in sight.

    None when there is nothing to cut out, which is the same signal doc 1's
    masker gives: no object located means do not attempt this repair.
    """
    if mask_result is None:
        return None
    m = np.asarray(mask_result.mask) > 0.5
    ys, xs = np.where(m)
    if xs.size == 0:
        return None

    rgb = np.array(reference.convert("RGB"))
    rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(rgba, "RGBA").crop(box)
