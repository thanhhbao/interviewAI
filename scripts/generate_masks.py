"""Generate layered segmentation masks for frontend/public/images/hero.png using SAM 2.

GroundingDINO was intentionally skipped: on this machine/connection, downloading the
grounding-dino-tiny weights stalled/failed three times in a row (very slow HF Hub
transfer, unrelated to the model itself). The task's own fallback clause ("use
manually specified boxes/points with SAM 2 if GroundingDINO is unreliable") was
extended to every object, not just the foreground silhouettes, since the reliability
problem applies equally to all GroundingDINO calls in this environment. Every box/point
below was picked by visually inspecting the actual image with a pixel-coordinate grid
overlay (see scratchpad grid crops), not guessed.

Usage:
    source scripts/.venv-masks/bin/activate
    python3 scripts/generate_masks.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam2Model, Sam2Processor

ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT_DIR / "frontend/public/images/hero.png"
MASKS_DIR = ROOT_DIR / "frontend/public/images/masks"
SAM2_CHECKPOINT = "facebook/sam2.1-hiera-small"

# Minimum connected-component area, as a fraction of total image area, to keep.
# Removes isolated noise / tiny floating mask artifacts.
MIN_COMPONENT_AREA_FRACTION = 0.0005
# Small-hole closing kernel (px) — closes tiny gaps without filling real ones.
CLOSE_KERNEL_PX = 5
# Boundary feather radius (px) for antialiasing.
FEATHER_PX = 1.5
# Optional final dilation (px) to avoid neon-line leakage around edges.
DILATE_PX = 1


@dataclass
class MaskSpec:
    name: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    color: tuple[int, int, int] = (255, 0, 0)
    dilate_px: int = DILATE_PX
    # Rects (x1,y1,x2,y2) to force-clear post-hoc. Only used for the table legs,
    # which sit flush against the robot/human legs with no visible seam — SAM2
    # negative points couldn't carve them out (verified: three rounds of negative
    # points on the leg centerline did not separate them), so this clips a
    # visually-confirmed contamination strip rather than leaving it in the mask.
    exclude_rects: list[tuple[float, float, float, float]] = field(default_factory=list)


MASK_SPECS: list[MaskSpec] = [
    MaskSpec(
        name="robot",
        box=(585, 220, 800, 700),
        positive_points=[(665, 260), (630, 360), (700, 415), (660, 480), (680, 580), (700, 650)],
        negative_points=[(520, 480), (500, 550), (820, 420), (700, 470), (700, 550), (700, 620)],
        color=(255, 60, 60),
        exclude_rects=[(676, 452, 726, 700)],
    ),
    MaskSpec(
        name="robot-chair",
        box=(478, 405, 665, 665),
        positive_points=[(520, 450), (550, 500), (500, 600)],
        negative_points=[(650, 300), (700, 430)],
        color=(255, 160, 60),
    ),
    MaskSpec(
        name="human",
        box=(1140, 220, 1360, 700),
        positive_points=[(1195, 260), (1220, 380), (1150, 300), (1230, 520), (1250, 600), (1230, 650)],
        negative_points=[(1350, 450), (1390, 550), (1080, 420), (1200, 470), (1200, 550), (1200, 620)],
        color=(60, 120, 255),
        exclude_rects=[(1218, 452, 1252, 700)],
    ),
    MaskSpec(
        name="human-chair",
        box=(1240, 395, 1430, 680),
        positive_points=[(1330, 430), (1350, 500), (1390, 600)],
        negative_points=[(1200, 350), (1080, 420)],
        color=(60, 220, 255),
    ),
    MaskSpec(
        name="table",
        box=(690, 395, 1165, 475),
        positive_points=[(720, 435), (800, 435), (1000, 435), (1130, 435), (870, 440)],
        negative_points=[(935, 385), (1100, 405), (1105, 422)],
        color=(200, 160, 90),
    ),
    MaskSpec(
        name="foreground-left",
        box=(0, 0, 430, 821),
        positive_points=[(60, 150), (150, 200), (200, 150), (370, 300), (150, 450), (100, 600), (250, 650)],
        negative_points=[(470, 100), (450, 750)],
        color=(60, 255, 120),
    ),
    MaskSpec(
        name="foreground-right",
        box=(1650, 0, 1916, 821),
        positive_points=[(1700, 150), (1750, 250), (1850, 150), (1750, 450), (1850, 550), (1700, 650), (1850, 700)],
        negative_points=[(1500, 400), (1550, 550)],
        color=(230, 60, 230),
    ),
]


def load_sam2():
    processor = Sam2Processor.from_pretrained(SAM2_CHECKPOINT)
    model = Sam2Model.from_pretrained(SAM2_CHECKPOINT)
    model.eval()
    return processor, model


def predict_mask(processor, model, image: Image.Image, spec: MaskSpec) -> np.ndarray:
    points = [[float(x), float(y)] for x, y in (spec.positive_points + spec.negative_points)]
    labels = [1] * len(spec.positive_points) + [0] * len(spec.negative_points)

    inputs = processor(
        images=image,
        input_points=[[points]],
        input_labels=[[labels]],
        input_boxes=[[list(spec.box)]],
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]  # [1, 3, H, W]
    scores = outputs.iou_scores[0, 0]  # [3]
    best = int(torch.argmax(scores).item())
    mask = masks[0, best].numpy().astype(np.uint8) * 255
    return mask


def clean_mask(
    mask: np.ndarray,
    dilate_px: int,
    exclude_rects: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)

    for x1, y1, x2, y2 in exclude_rects or []:
        binary[int(y1) : int(y2), int(x1) : int(x2)] = 0

    # Remove isolated noise / tiny floating artifacts.
    total_area = binary.shape[0] * binary.shape[1]
    min_area = int(total_area * MIN_COMPONENT_AREA_FRACTION)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1

    # Close small gaps (does not fill large real holes) without eroding true contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    # Light opening to drop thin speckle noise introduced by closing.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    if dilate_px > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        cleaned = cv2.dilate(cleaned, dilate_kernel)

    cleaned_255 = (cleaned * 255).astype(np.uint8)

    # Antialias only the boundary: blur a narrow band around the edge, keep the
    # interior hard so robot/human contours stay crisp per the requirements.
    blurred = cv2.GaussianBlur(cleaned_255, (0, 0), sigmaX=FEATHER_PX)
    edge_band = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge_band = cv2.dilate(edge_band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    final = np.where(edge_band.astype(bool), blurred, cleaned_255)
    return final


def build_preview(image: Image.Image, mask_arrays: dict[str, np.ndarray], specs: list[MaskSpec]) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    spec_by_name = {s.name: s for s in specs}

    for name, mask in mask_arrays.items():
        color = spec_by_name[name].color
        binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            points = [tuple(p[0]) for p in contour]
            draw.line(points + [points[0]], fill=color, width=3)

    legend_x, legend_y = 16, 16
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    for name, mask in mask_arrays.items():
        color = spec_by_name[name].color
        draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 18], fill=color)
        draw.text((legend_x + 26, legend_y), name, fill=(255, 255, 255), font=font)
        legend_y += 24

    return preview


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {IMAGE_PATH}")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Loaded {IMAGE_PATH} size={image.size}")

    processor, model = load_sam2()
    print("SAM2 model ready.")

    mask_arrays: dict[str, np.ndarray] = {}
    for spec in MASK_SPECS:
        raw_mask = predict_mask(processor, model, image, spec)
        cleaned = clean_mask(raw_mask, dilate_px=spec.dilate_px, exclude_rects=spec.exclude_rects)
        assert cleaned.shape == (image.height, image.width), "mask must match source resolution exactly"
        out_path = MASKS_DIR / f"{spec.name}.png"
        Image.fromarray(cleaned, mode="L").save(out_path)
        mask_arrays[spec.name] = cleaned
        print(f"wrote {out_path} ({np.count_nonzero(cleaned > 127)} px selected)")

    preview = build_preview(image, mask_arrays, MASK_SPECS)
    preview_path = MASKS_DIR / "preview.png"
    preview.save(preview_path)
    print(f"wrote {preview_path}")


if __name__ == "__main__":
    main()
