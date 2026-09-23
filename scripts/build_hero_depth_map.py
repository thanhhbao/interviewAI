"""Build the lightweight depth texture used by the homepage WebGL scan.

The SAM masks provide the important depth discontinuities (foreground,
subjects, chairs and table). Gentle spatial and luminance gradients keep a
whole segmented object from flashing as one flat block.

Usage:
    scripts/.venv-masks/bin/python scripts/build_hero_depth_map.py
"""
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT_DIR / "frontend/public/images/hero.png"
MASKS_DIR = ROOT_DIR / "frontend/public/images/masks"
OUTPUT_PATH = ROOT_DIR / "frontend/public/images/hero-depth.png"


def load_mask(name: str) -> np.ndarray:
    mask = cv2.imread(str(MASKS_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(MASKS_DIR / f"{name}.png")
    return mask.astype(np.float32) / 255.0


def main() -> None:
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    luminance = cv2.GaussianBlur(luminance, (0, 0), 5.0)

    # The room is far away near the upper centre and gradually comes closer
    # towards the floor and side edges.
    side_distance = np.abs(x - 0.5) * 2.0
    depth = 0.07 + 0.16 * y + 0.08 * side_distance + 0.035 * luminance

    layers = [
        ("robot-chair", 0.47, 0.10, -0.025),
        ("human-chair", 0.49, 0.10, 0.025),
        ("table", 0.54, 0.055, 0.0),
        ("robot", 0.57, 0.13, -0.02),
        ("human", 0.59, 0.13, 0.02),
        ("foreground-left", 0.76, 0.13, -0.035),
        ("foreground-right", 0.78, 0.13, 0.035),
    ]

    for name, base, vertical_range, horizontal_tilt in layers:
        alpha = load_mask(name)
        # Keep mask borders smooth enough for a clean contour, while retaining
        # the original silhouette instead of turning it into a broad halo.
        alpha = cv2.GaussianBlur(alpha, (0, 0), 2.2)
        layer_depth = base + vertical_range * y + horizontal_tilt * (x - 0.5)
        layer_depth += 0.035 * (luminance - 0.5)
        depth = depth * (1.0 - alpha) + layer_depth * alpha

    # A final tiny blur removes 8-bit stair steps without erasing object edges.
    depth = cv2.GaussianBlur(depth, (0, 0), 1.15)
    depth = np.clip(depth, 0.0, 1.0)
    output = np.round(depth * 255.0).astype(np.uint8)

    if not cv2.imwrite(str(OUTPUT_PATH), output):
        raise RuntimeError(f"Could not write {OUTPUT_PATH}")
    print(f"wrote {OUTPUT_PATH} ({width}x{height})")


if __name__ == "__main__":
    main()
