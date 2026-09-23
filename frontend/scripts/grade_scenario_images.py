"""Deterministic grading pass for the 4 Scenario-carousel photos.

Crops each source to a 4:5 editorial portrait (with a per-image focal
region), then applies one shared cinematic treatment — reduced exposure,
desaturation, deeper blacks, a teal color cast, a cool highlight shift and a
soft vignette — so all four feel like one photographed set rather than four
different stock photos. Same deterministic-grading approach used earlier for
hero-dark.webp (no generative/AI editing).
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../public/images/scenarios"

# name -> (raw filename, output filename, focal crop box as fraction of width [x0,x1] within the vertical strip)
JOBS = [
    ("raw-system-design.jpg", "system-design.webp", (0.30, 0.86)),   # server racks + depth
    ("raw-algorithms.jpg", "algorithms.webp", (0.18, 0.74)),          # monitor / code
    ("raw-behavioral.jpg", "behavioral.webp", (0.28, 0.84)),          # two people facing each other
    ("raw-negotiation.jpg", "negotiation.webp", (0.22, 0.78)),        # documents + conversation
]

TARGET_RATIO = 4 / 5  # width / height

TEAL_CAST = np.array([5, 20, 17], dtype=np.float32)      # #050807-ish deep cast, added into shadows
TEAL_TINT = np.array([14, 42, 38], dtype=np.float32)      # #0E2A26 tint mixed toward midtones


def crop_to_ratio(im: Image.Image, x_frac: tuple[float, float]) -> Image.Image:
    w, h = im.size
    target_w = int(h * TARGET_RATIO)
    if target_w <= w:
        # crop horizontally within the requested focal band
        x0f, x1f = x_frac
        band_w = x1f * w - x0f * w
        cx = x0f * w + band_w / 2
        left = int(max(0, min(w - target_w, cx - target_w / 2)))
        return im.crop((left, 0, left + target_w, h))
    else:
        target_h = int(w / TARGET_RATIO)
        top = int(max(0, (h - target_h) / 2))
        return im.crop((0, top, w, top + target_h))


TARGET_MEAN_LUMA = 62.0  # out of 255 — pulls every source to the same overall darkness


def grade(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    luma_coef = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # 0. auto-exposure normalize: match every source's mean brightness to the
    # same target so a bright white-desk shot and a near-black server room
    # land at the same overall darkness, not just the same relative -24%.
    src_mean = float((arr @ luma_coef).mean())
    exposure_gain = np.clip(TARGET_MEAN_LUMA / max(src_mean, 1.0), 0.28, 1.0)
    arr *= exposure_gain

    # 1. saturation down ~40% (luminance-preserving desaturate) — stronger
    # than a plain -28% so warm/olive casts (e.g. tungsten office light)
    # don't survive as a competing hue against the teal grade.
    gray = arr @ luma_coef
    gray = gray[..., None]
    arr = gray + (arr - gray) * 0.6

    # 2. deepen blacks — lift the black point down via a gentle S-curve on shadows
    norm = arr / 255.0
    shadow_mask = np.clip(1.0 - norm, 0, 1) ** 1.6
    arr = arr - shadow_mask * 16.0

    # 3. teal color cast into shadows + midtones, stronger than before so it
    # reads as the dominant hue rather than a light wash over the source tone
    luma = (arr @ luma_coef)[..., None] / 255.0
    shadow_weight = np.clip(1.0 - luma * 1.3, 0, 1)
    mid_weight = np.clip(1.0 - np.abs(luma - 0.5) * 1.6, 0, 1)
    arr = arr + TEAL_CAST[None, None, :] * shadow_weight * 0.85
    arr = arr + TEAL_TINT[None, None, :] * mid_weight * 0.32

    # 3b. global channel balance toward teal — independent of local luma, so
    # a warm/olive source (tungsten light, skin, wood) can't out-vote the
    # shared grade the way a purely luma-weighted cast can.
    arr[..., 0] *= 0.90
    arr[..., 1] *= 0.97
    arr[..., 2] *= 1.10

    # 4. slight contrast increase
    arr = (arr - 127.5) * 1.1 + 127.5

    # 5. cool the highlights toward teal, and kill residual warm cast on the
    # brightest source (paper/desk whites) so they sit blue-teal, not cream
    high_weight = np.clip((luma - 0.5) * 2.0, 0, 1)
    arr[..., 0] -= high_weight[..., 0] * 14
    arr[..., 1] -= high_weight[..., 0] * 2
    arr[..., 2] += high_weight[..., 0] * 10

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr, "RGB")

    # 7. subtle vignette
    w, h = out.size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    vig = np.clip(1.0 - (dist - 0.55) * 0.55, 0.72, 1.0).astype(np.float32)
    out_arr = np.asarray(out).astype(np.float32) * vig[..., None]
    out = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), "RGB")

    # 8. restrained grain
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 6.5, size=(h, w, 1)).astype(np.float32)
    grained = np.asarray(out).astype(np.float32) + noise
    out = Image.fromarray(np.clip(grained, 0, 255).astype(np.uint8), "RGB")

    return out


def main():
    for raw_name, out_name, focal in JOBS:
        src_path = os.path.join(SRC_DIR, raw_name)
        im = Image.open(src_path)
        cropped = crop_to_ratio(im, focal)
        cropped = cropped.resize((960, 1200), Image.LANCZOS)
        graded = grade(cropped)
        out_path = os.path.join(SRC_DIR, out_name)
        graded.save(out_path, "WEBP", quality=82, method=6)
        print(f"{raw_name} -> {out_name} ({graded.size}, {os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
