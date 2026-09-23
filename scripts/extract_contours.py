"""Derive short SVG contour-fragment paths from the hero segmentation masks.

For each mask, this finds the external silhouette(s) with cv2.findContours,
simplifies them with approxPolyDP, and chops each simplified polyline into
several short open arcs (not the full closed outline) so the frontend can
animate an energy pulse that traces PIECES of object geometry rather than a
single line crossing the photo. Output is a generated TypeScript module
consumed directly by HeroEnergyStage.tsx — no masks or hero.png are modified.

Usage:
    source scripts/.venv-masks/bin/activate
    python3 scripts/extract_contours.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MASKS_DIR = ROOT_DIR / "frontend/public/images/masks"
OUT_PATH = ROOT_DIR / "frontend/src/components/home/heroContours.generated.ts"

# name -> (min contour area fraction of mask bbox area, num arc fragments to keep, keep ratio)
OBJECT_CONFIG: dict[str, dict] = {
    "foreground-left": {"key": "foregroundLeftPaths", "fragments": 5, "keep": 0.7, "min_area": 400},
    "robot": {"key": "robotPaths", "fragments": 6, "keep": 0.75, "min_area": 200},
    "robot-chair": {"key": "robotChairPaths", "fragments": 4, "keep": 0.7, "min_area": 200},
    "table": {"key": "tablePaths", "fragments": 4, "keep": 0.8, "min_area": 200},
    "human": {"key": "humanPaths", "fragments": 6, "keep": 0.75, "min_area": 200},
    "human-chair": {"key": "humanChairPaths", "fragments": 4, "keep": 0.7, "min_area": 200},
    "foreground-right": {"key": "foregroundRightPaths", "fragments": 5, "keep": 0.7, "min_area": 400},
}


class Fragment(NamedTuple):
    d: str
    length: float


def polyline_to_smooth_path(points: np.ndarray) -> str:
    """Same 'through-point' quadratic smoothing used elsewhere in this repo."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return ""
    if len(pts) == 2:
        return f"M {pts[0][0]:.1f},{pts[0][1]:.1f} L {pts[1][0]:.1f},{pts[1][1]:.1f}"
    d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        mx = (pts[i][0] + pts[i + 1][0]) / 2
        my = (pts[i][1] + pts[i + 1][1]) / 2
        d += f" Q {pts[i][0]:.1f},{pts[i][1]:.1f} {mx:.1f},{my:.1f}"
    d += f" L {pts[-1][0]:.1f},{pts[-1][1]:.1f}"
    return d


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.sqrt((diffs**2).sum(axis=1))))


def chop_into_fragments(points: np.ndarray, num_fragments: int, keep_ratio: float) -> list[Fragment]:
    """Split a closed contour polyline into `num_fragments` open arcs of
    roughly equal arc-length, drop some of them so only `keep_ratio` remain
    (leaving real gaps rather than a full outline), and skip a random-ish
    ashift each mask so the kept arcs vary in position around the shape."""
    n = len(points)
    if n < 6:
        return []
    seg_size = max(3, n // num_fragments)
    fragments: list[Fragment] = []
    for i in range(num_fragments):
        start = (i * seg_size) % n
        end = start + seg_size + 1  # +1 point overlap so fragments touch visually
        if end <= n:
            chunk = points[start:end]
        else:
            chunk = np.concatenate([points[start:n], points[0 : end - n]], axis=0)
        if len(chunk) < 3:
            continue
        length = polyline_length(chunk)
        if length < 12:
            continue
        fragments.append(Fragment(d=polyline_to_smooth_path(chunk), length=length))

    keep_n = max(1, round(len(fragments) * keep_ratio))
    # Keep the longest arcs (most visually meaningful) rather than a random
    # subset, so short noisy nubs are dropped first.
    fragments.sort(key=lambda f: f.length, reverse=True)
    return fragments[:keep_n]


def extract_object_fragments(mask_path: Path, num_fragments: int, keep_ratio: float, min_area: float) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    # A mask may legitimately contain more than one blob (e.g. foreground
    # silhouettes are several disjoint shapes) — process the largest few.
    contours = contours[:4]

    all_fragments: list[Fragment] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.5, 0.004 * perimeter)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        points = simplified.reshape(-1, 2)
        if len(points) < 6:
            continue
        all_fragments.extend(chop_into_fragments(points, num_fragments, keep_ratio))

    all_fragments.sort(key=lambda f: f.length, reverse=True)
    return [f.d for f in all_fragments]


def main() -> None:
    result: dict[str, list[str]] = {}
    for mask_name, cfg in OBJECT_CONFIG.items():
        mask_path = MASKS_DIR / f"{mask_name}.png"
        fragments = extract_object_fragments(mask_path, cfg["fragments"], cfg["keep"], cfg["min_area"])
        result[cfg["key"]] = fragments
        print(f"{mask_name}: {len(fragments)} fragments")

    total = sum(len(v) for v in result.values())
    print(f"total fragments: {total}")

    lines = [
        "// AUTO-GENERATED by scripts/extract_contours.py — do not hand-edit.",
        "// Regenerate with: source scripts/.venv-masks/bin/activate && python3 scripts/extract_contours.py",
        "// Coordinate system matches hero.png / masks: viewBox=\"0 0 1916 821\".",
        "",
    ]
    for key, paths in result.items():
        lines.append(f"export const {key}: string[] = {json.dumps(paths, indent=2)};")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
