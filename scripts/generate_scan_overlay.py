"""Generate a transparent WebM scan-light overlay for the hero image.

Architecture (deterministic, no generative video):
  1. Sobel edge map of the graded hero photo (hero-dark.webp) — flat surfaces
     stay dark, real object contours (robot, human, table, lamp) respond.
  2. A per-frame top-to-bottom scan front: sharp core + blurred local glow +
     an exponential ghost trail behind it, all spatially defined first.
  3. Final visible signal = spatial scan bands x edge map. Flat regions never
     light up regardless of scan position; only actual edges do.
  4. Frames are rendered as straight RGBA PNGs (alpha = 0 except on lit
     edges), then encoded to a VP9 WebM with a real alpha channel.

Usage:
    source scripts/.venv-masks/bin/activate
    python3 scripts/generate_scan_overlay.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_DIR / "frontend/public/images/hero-dark.webp"
FRAMES_DIR = ROOT_DIR / "scripts/.build/scan_frames"
WEBM_OUT = ROOT_DIR / "frontend/public/videos/scan-overlay.webm"
PREVIEW_OUT = ROOT_DIR / "scripts/.build/preview.mp4"

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
FPS = 30
LOOP_DURATION = 5.6        # full loop, seconds (includes a dark pause)
SCAN_DURATION = 5.0        # active outward-expansion time, seconds

CORE_WIDTH = 15              # px, band thickness — wider means each point stays lit longer
TRAIL_LENGTH = 300          # px, how far the ghost trail extends behind the front — long
                             # enough to bridge the gap between separate object clusters
                             # (lamp vs. robot/human/table) so the wave reads as one
                             # continuous pass instead of isolated flashes. Shortened from
                             # 420: at the old length + the edge-brightness boost below, the
                             # trail read as a lingering teal color cast rather than a quick
                             # afterglow.
TRAIL_DECAY = 130           # px, exponential decay constant for the trail — higher lingers
                             # longer. Lowered from 220 so the afterglow fades noticeably
                             # faster instead of staining the frame teal as it passes.

CORE_OPACITY = 1.0
GLOW_OPACITY = 0.75
TRAIL_OPACITY = 0.28
GLOW_BLUR = 22               # px, gaussian sigma for the tight local glow halo

# Second, wider/softer tier so light actually washes onto the nearby object
# surface (not just a thin illuminated outline) as the wave passes it.
BLOOM_OPACITY = 0.4
BLOOM_BLUR = 48               # px, gaussian sigma for the wide bloom wash


EDGE_BLUR_KSIZE = 9          # pre-blur before Sobel — smooths WebP compression
                             # blocking so faint real gradients (smoke, haze,
                             # soft wall falloff) survive as organic shapes
                             # instead of a grid of block artifacts
EDGE_THRESHOLD = 0.045       # normalized [0,1], below this -> 0 edge response —
                             # low enough to admit genuine low-contrast detail
EDGE_GAMMA = 0.55            # <1 boosts mid-strength edges
EDGE_DILATION = 2            # px, thickens thin edges so they aren't lost to AA

WAVE_AMPLITUDE = 6           # px-equivalent radial wobble, kept subtle
WAVE_FREQUENCY = 7.0         # angular cycles around the ellipse

# Elliptical distance-field origin/shape (normalized to image size). The wave
# is a mathematical mask only — never drawn as a visible ring — so contours
# at different Y positions can all light up in the same frame as the front
# sweeps outward through them.
ORIGIN_X_NORM = 0.53
ORIGIN_Y_NORM = 0.28
X_RADIUS_SCALE = 1.55        # wide cinematic ellipse, not a circular sonar pulse
Y_RADIUS_SCALE = 1.0

CORE_COLOR = np.array([0x78, 0xFF, 0xE7], dtype=np.float32)   # sharp line
GLOW_COLOR = np.array([0x35, 0xEB, 0xCB], dtype=np.float32)   # local halo
TRAIL_COLOR = np.array([0x14, 0xA9, 0x8F], dtype=np.float32)  # deep ghost trail

# Genuinely flat regions (fog, plain wall) have ~zero Sobel response, so a
# purely edge-gated mask skips them entirely — reading as "the scan never
# touches that part of the room". A uniform brightness floor was tried and
# rejected: multiplying a spatially-constant value across a flat area just
# reveals the wave's own ring shape, which the design explicitly forbids.
# Instead, gate the ambient contribution by a fixed, blotchy low-frequency
# noise field — like dust motes/haze in the air catching the light — so flat
# regions respond irregularly (organic), never as a clean drawn ellipse.
AMBIENT_FLOOR = 0.24
HAZE_NOISE_SEED = 7

# The periphery of the source photo (corners, edges) is mostly flat negative
# space with far fewer real Sobel edges than the busy center — so even with
# the haze floor above, the wave visibly dims as it travels outward and
# reaches the rim. Compensate by boosting overall intensity as a function of
# how far the front has travelled (normalized 0 at origin -> 1 at the
# farthest corner), so the outer edge stays clearly lit instead of trailing
# off into near-invisibility.
EDGE_RADIUS_BOOST = 1.35


def build_edge_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)

    mag = np.clip((mag - EDGE_THRESHOLD) / (1.0 - EDGE_THRESHOLD), 0.0, 1.0)
    mag = np.power(mag, EDGE_GAMMA)

    if EDGE_DILATION > 0:
        k = 2 * EDGE_DILATION + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mag = cv2.dilate(mag, kernel)

    return mag.astype(np.float32)


def build_haze_field(h: int, w: int) -> np.ndarray:
    """Fixed, blotchy low-frequency field so flat/featureless areas (fog,
    plain wall) can still catch a little light — irregularly, like dust in
    the air — instead of either staying dead or revealing the wave's own
    ring shape under a spatially-uniform floor."""
    rng = np.random.default_rng(HAZE_NOISE_SEED)
    small = rng.random((h // 6 + 2, w // 6 + 2), dtype=np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = cv2.GaussianBlur(noise, (0, 0), 6)
    noise -= noise.min()
    noise /= (noise.max() + 1e-6)
    # Push most values toward 0 with occasional brighter flecks, instead of a
    # smooth even field, so it reads as sparse motes rather than a haze wash.
    noise = np.power(noise, 2.2)
    return noise.astype(np.float32)


def build_distance_and_angle(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Elliptical distance field + angle map from the wave origin, computed
    once since only `radius` changes frame to frame."""
    origin_x = ORIGIN_X_NORM * w
    origin_y = ORIGIN_Y_NORM * h

    x_coords = np.arange(w, dtype=np.float32).reshape(1, w)
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)

    dx = (x_coords - origin_x) / X_RADIUS_SCALE
    dy = (y_coords - origin_y) / Y_RADIUS_SCALE

    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    angle = np.arctan2(np.broadcast_to(dy, (h, w)), np.broadcast_to(dx, (h, w))).astype(np.float32)
    return dist, angle


def max_corner_radius(h: int, w: int) -> float:
    origin_x = ORIGIN_X_NORM * w
    origin_y = ORIGIN_Y_NORM * h
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    best = 0.0
    for cx, cy in corners:
        dx = (cx - origin_x) / X_RADIUS_SCALE
        dy = (cy - origin_y) / Y_RADIUS_SCALE
        best = max(best, (dx * dx + dy * dy) ** 0.5)
    return best


def render_frame(
    edge_map: np.ndarray,
    ambient_gate: np.ndarray,
    dist_map: np.ndarray,
    angle_map: np.ndarray,
    radius: float,
    t: float,
    active: bool,
    radius_norm: float = 0.0,
) -> np.ndarray:
    h, w = edge_map.shape
    if not active:
        return np.zeros((h, w, 4), dtype=np.uint8)

    # Compensate for the periphery having far less real edge detail than the
    # busy center — scales glow/bloom/trail up as the front travels outward
    # so the rim reads clearly lit instead of trailing into near-invisibility.
    edge_boost = 1.0 + EDGE_RADIUS_BOOST * np.clip(radius_norm, 0.0, 1.0)

    # Very subtle low-frequency wobble on the local radius so the front
    # doesn't read as a mathematically perfect ellipse.
    wobble = WAVE_AMPLITUDE * (
        0.6 * np.sin(angle_map * WAVE_FREQUENCY + t * 0.7)
        + 0.4 * np.sin(angle_map * WAVE_FREQUENCY * 2.3 - t * 0.4)
    )
    radius_local = radius + wobble

    # Wave expands outward: delta<0 not yet reached, delta~=0 sharp front,
    # delta>0 already scanned (trail lives here, inside the current radius).
    delta = radius_local - dist_map

    core_band = np.exp(-0.5 * (delta / (CORE_WIDTH / 2.0)) ** 2).astype(np.float32)

    passed = delta >= 0
    trail_window = np.clip(1.0 - delta / TRAIL_LENGTH, 0.0, 1.0)
    trail_band = np.where(passed, np.exp(-delta / TRAIL_DECAY) * trail_window, 0.0).astype(np.float32)

    edge_gated_core = core_band * ambient_gate
    edge_gated_trail = trail_band * ambient_gate

    glow_source = edge_gated_core + edge_gated_trail * 0.7
    glow_band = cv2.GaussianBlur(glow_source, (0, 0), GLOW_BLUR)
    # Wider, softer wash from the same lit-edge source so light actually
    # spills onto the nearby object surface, not just the exact edge pixel.
    bloom_band = cv2.GaussianBlur(glow_source, (0, 0), BLOOM_BLUR)

    # edge_boost only strengthens the instantaneous glow/bloom at the front —
    # applying it to the trail too was what made the afterglow read as a
    # lingering teal color cast instead of a quick pass-through.
    a_core = np.clip(edge_gated_core * CORE_OPACITY, 0.0, 1.0)
    a_glow = np.clip(glow_band * GLOW_OPACITY * edge_boost, 0.0, 1.0)
    a_bloom = np.clip(bloom_band * BLOOM_OPACITY * edge_boost, 0.0, 1.0)
    a_trail = np.clip(edge_gated_trail * TRAIL_OPACITY, 0.0, 1.0)

    total_alpha = np.clip(a_core + a_glow + a_bloom + a_trail, 0.0, 0.97)

    denom = (a_core + a_glow + a_bloom + a_trail)[..., None] + 1e-6
    color = (
        a_core[..., None] * CORE_COLOR
        + a_glow[..., None] * GLOW_COLOR
        + a_bloom[..., None] * GLOW_COLOR
        + a_trail[..., None] * TRAIL_COLOR
    ) / denom

    # cv2.imwrite expects BGRA channel order, but CORE/GLOW/TRAIL_COLOR are
    # defined as RGB (readable as hex) — flip before writing.
    color_bgr = color[..., ::-1]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = np.clip(color_bgr, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(total_alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(SRC_PATH)

    bgr = cv2.imread(str(SRC_PATH), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {SRC_PATH}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    edge_map = build_edge_map(gray)
    haze_field = build_haze_field(h, w)
    ambient_gate = np.maximum(edge_map, AMBIENT_FLOOR * haze_field)
    dist_map, angle_map = build_distance_and_angle(h, w)

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    total_frames = round(LOOP_DURATION * FPS)
    scan_frames = round(SCAN_DURATION * FPS)

    start_radius = 0.0
    end_radius = max_corner_radius(h, w) + TRAIL_LENGTH

    for i in range(total_frames):
        t = i / FPS
        if i < scan_frames:
            progress = i / max(scan_frames - 1, 1)
            radius = start_radius + progress * (end_radius - start_radius)
            frame = render_frame(edge_map, ambient_gate, dist_map, angle_map, radius, t, active=True, radius_norm=progress)
        else:
            frame = render_frame(edge_map, ambient_gate, dist_map, angle_map, 0.0, t, active=False)

        out_path = FRAMES_DIR / f"scan_{i + 1:04d}.png"
        cv2.imwrite(str(out_path), frame)

    print(f"wrote {total_frames} frames to {FRAMES_DIR}")

    WEBM_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(FRAMES_DIR / "scan_%04d.png"),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            "-b:v", "0",
            "-crf", "30",
            "-an",
            str(WEBM_OUT),
        ],
        check=True,
    )
    print(f"wrote {WEBM_OUT}")

    PREVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(SRC_PATH),
            "-framerate", str(FPS),
            "-i", str(FRAMES_DIR / "scan_%04d.png"),
            "-filter_complex",
            "[0:v]fps=" + str(FPS) + "[bg];"
            "[bg][1:v]overlay=0:0:shortest=1,crop=trunc(iw/2)*2:trunc(ih/2)*2[out]",
            "-map", "[out]",
            "-r", str(FPS),
            "-t", str(LOOP_DURATION),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(PREVIEW_OUT),
        ],
        check=True,
    )
    print(f"wrote {PREVIEW_OUT}")


if __name__ == "__main__":
    main()
