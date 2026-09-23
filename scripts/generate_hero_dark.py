"""Deterministic dark-teal color grade of the hero photo (no generative editing).

Produces frontend/public/images/hero-dark.webp from hero.png using pure
per-pixel curve/tint math (numpy), so the result is fully reproducible and
pixel-aligned with the source — required because the scan edge extraction and
WebM overlay must be built from this exact graded image.

Grade, in order:
  1. Moderate exposure reduction (uniform multiply).
  2. Global channel scale to suppress the amber/brown cast (pull red down,
     push blue up) without touching hue *structure* (no per-object masking).
  3. Luminance-based shadow/midtone desaturation + tint toward a deep petrol
     teal, with strength falling off toward zero in the brightest highlights
     (the lamp glow) so a restrained pool of warm light survives there.

Usage:
    source scripts/.venv-masks/bin/activate
    python3 scripts/generate_hero_dark.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_DIR / "frontend/public/images/hero.png"
OUT_PATH = ROOT_DIR / "frontend/public/images/hero-dark.webp"

EXPOSURE = 0.80  # moderate overall reduction
CHANNEL_SCALE = np.array([0.80, 0.95, 1.20], dtype=np.float32)  # R, G, B — kills the amber cast
TEAL = np.array([0.018, 0.075, 0.085], dtype=np.float32)  # deep, near-black petrol teal
SHADOW_DESATURATION = 0.45  # how much shadow/mid saturation is pulled toward gray before tinting
TEAL_STRENGTH = 0.60  # max blend-in of the teal target in the deepest, non-lamp shadows
HIGHLIGHT_LOW = 0.42  # luminance where the "keep it warm" lamp mask starts to open
HIGHLIGHT_HIGH = 0.80  # luminance at which the lamp mask is fully open (fully preserved warmth)


def rec601_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(SRC_PATH)

    src = Image.open(SRC_PATH).convert("RGB")
    arr = np.asarray(src).astype(np.float32) / 255.0

    # 1. Exposure.
    arr = arr * EXPOSURE

    # 2. Global amber-cast suppression.
    arr = arr * CHANNEL_SCALE[None, None, :]

    lum = rec601_luminance(arr)

    # Lamp/highlight mask: smoothstep so only genuinely bright pixels (the
    # overhead lamp and its immediate glow) are allowed to stay warm.
    t = np.clip((lum - HIGHLIGHT_LOW) / (HIGHLIGHT_HIGH - HIGHLIGHT_LOW), 0.0, 1.0)
    highlight_mask = t * t * (3.0 - 2.0 * t)  # smoothstep

    # 3. Shadow/midtone desaturate + tint toward teal, gated off in highlights.
    shadow_gate = 1.0 - highlight_mask
    gray = lum[..., None]
    desat = SHADOW_DESATURATION * shadow_gate[..., None]
    arr_desat = arr * (1.0 - desat) + gray * desat

    # Darker pixels get pulled toward teal harder than mid pixels, still
    # gated to zero wherever the highlight mask is open.
    depth_weight = np.clip(1.0 - lum, 0.0, 1.0)
    tint_strength = (TEAL_STRENGTH * depth_weight * shadow_gate)[..., None]
    graded = arr_desat * (1.0 - tint_strength) + TEAL[None, None, :] * tint_strength

    # Recombine with the (untinted, exposure+cast-corrected) highlight color
    # in proportion to the lamp mask, so warm light survives under the lamp.
    result = graded * (1.0 - highlight_mask[..., None]) + arr * highlight_mask[..., None]

    result = np.clip(result, 0.0, 1.0)
    out_img = Image.fromarray((result * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(OUT_PATH, format="WEBP", quality=92, method=6)
    print(f"wrote {OUT_PATH} size={out_img.size}")


if __name__ == "__main__":
    main()
