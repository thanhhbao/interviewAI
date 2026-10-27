import * as THREE from 'three';
import { splitmix32, randRange } from '@/components/home/footer/dandelionRandom';

// ASSET PROVENANCE (placeholder, prototype-only — see the V3 task report):
// every pixel here is procedurally drawn on an offscreen <canvas> at
// runtime. Nothing is photographed, hand-drawn, licensed, or downloaded —
// there is no network access in this environment and no local botanical
// asset library to draw from. This is explicitly a stand-in for a real
// raster/atlas asset (option A/C from the brief): if licensed botanical
// photography or illustrated meadow textures become available, they
// should replace `buildLayerTexture` below without touching MeadowLayer.tsx
// or GardenFooterV3.tsx, which only consume a finished texture + its
// aspect ratio.
//
// Strategy change from V2: V2 placed one Object3D-transformed instance per
// flower (thousands of individual React/Three objects). V3 instead bakes
// HUNDREDS of flower + foliage marks into a small number of large textures
// (one per depth layer) — "one dense botanical surface" per the brief,
// not thousands of individually-readable sprites. The U-shaped meadow/sky
// boundary is baked directly into each texture's alpha mask (see
// `meadowTopFraction` below) so the silhouette reads correctly even
// without any 3D vertex displacement — the curved ground planes in
// MeadowLayer.tsx reinforce it, they don't solely create it.

export type LayerName = 'far' | 'mid' | 'near' | 'foreground';

interface LayerConfig {
  seed: number;
  width: number;
  height: number;
  flowerCount: number;
  flowerSize: [number, number];
  flowerBlurPx: number;
  flowerOpacity: [number, number];
  foliageCount: number;
  foliageSize: [number, number];
  foliageOpacity: [number, number];
  foliageBlurPx: number;
  // U-shaped meadow/sky boundary, as a fraction of canvas height, at
  // center vs. at the left/right edges — center is LARGER (meadow starts
  // lower down, more sky above it), edges are SMALLER (meadow reaches
  // higher up) — this is what actually produces the broad shallow U
  // silhouette per-layer, independent of any 3D geometry.
  meadowTopCenterFrac: number;
  meadowTopEdgeFrac: number;
  curveExponent: number;
  // Petals/centers stay within this brightness band (0=black, 1=white) —
  // near layers get the full band, far layers are compressed toward gray
  // so they recede instead of popping.
  brightnessRange: [number, number];
}

// Pure neutral grays (equal R/G/B) — the previous set had a faint green
// bias, which reads as "foliage-colored" even at low brightness. The
// brief is explicit: dark graphite, not dark green.
const FOLIAGE_GRAYS = ['#212321', '#262826', '#2b2d2b', '#333533'];

const LAYER_CONFIGS: Record<LayerName, LayerConfig> = {
  far: {
    seed: 71001,
    width: 1600,
    height: 700,
    flowerCount: 850,
    flowerSize: [3, 6],
    flowerBlurPx: 1.4,
    flowerOpacity: [0.35, 0.58],
    foliageCount: 300,
    foliageSize: [8, 16],
    foliageOpacity: [0.22, 0.4],
    foliageBlurPx: 1.2,
    meadowTopCenterFrac: 0.62,
    meadowTopEdgeFrac: 0.18,
    curveExponent: 1.7,
    brightnessRange: [0.55, 0.78],
  },
  mid: {
    seed: 71002,
    width: 1600,
    height: 760,
    flowerCount: 650,
    flowerSize: [7, 13],
    flowerBlurPx: 0.6,
    flowerOpacity: [0.55, 0.85],
    foliageCount: 420,
    foliageSize: [14, 26],
    foliageOpacity: [0.38, 0.62],
    foliageBlurPx: 0.6,
    meadowTopCenterFrac: 0.5,
    meadowTopEdgeFrac: 0.06,
    curveExponent: 1.9,
    brightnessRange: [0.72, 0.95],
  },
  near: {
    seed: 71003,
    width: 1600,
    height: 900,
    flowerCount: 380,
    flowerSize: [14, 26],
    flowerBlurPx: 0,
    flowerOpacity: [0.78, 1],
    foliageCount: 460,
    foliageSize: [22, 42],
    foliageOpacity: [0.5, 0.78],
    foliageBlurPx: 0,
    meadowTopCenterFrac: 0.32,
    meadowTopEdgeFrac: 0.0,
    curveExponent: 2.1,
    brightnessRange: [0.85, 1],
  },
  foreground: {
    seed: 71004,
    width: 1600,
    height: 700,
    flowerCount: 70,
    flowerSize: [10, 16],
    flowerBlurPx: 0.4,
    flowerOpacity: [0.35, 0.55],
    foliageCount: 620,
    foliageSize: [28, 52],
    foliageOpacity: [0.6, 0.88],
    foliageBlurPx: 0,
    meadowTopCenterFrac: 0.1,
    meadowTopEdgeFrac: 0.0,
    curveExponent: 2.2,
    brightnessRange: [0.75, 0.95],
  },
};

function meadowTopFraction(nx: number, cfg: LayerConfig): number {
  // nx in [-1, 1] across canvas width, 0 = center.
  const t = Math.pow(Math.abs(nx), cfg.curveExponent);
  return cfg.meadowTopEdgeFrac + (cfg.meadowTopCenterFrac - cfg.meadowTopEdgeFrac) * (1 - t);
}

// NOTE: no per-mark `ctx.filter = 'blur()'` here — Canvas2D's blur filter
// is expensive per invocation, and calling it hundreds/thousands of times
// (once per flower/foliage mark) was measured to stall the main thread
// long enough to hang the browser's own screenshot/compositor pipeline.
// Softness is instead applied ONCE per whole layer at the end of
// `buildLayerTexture` — a single filtered drawImage pass, independent of
// mark count.
function drawFoliageMark(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, color: string) {
  ctx.globalAlpha = opacity;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.ellipse(x, y, size, size * 0.62, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawFlowerMark(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, brightness: number, rand: () => number) {
  const gray = Math.round(brightness * 255);
  const petalColor = `rgb(${gray},${gray},${gray})`;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.fillStyle = petalColor;
  const petalCount = 5 + Math.floor(rand() * 3);
  let angle = randRange(rand, 0, Math.PI * 2);
  const step = (Math.PI * 2) / petalCount;
  for (let i = 0; i < petalCount; i++) {
    angle += step + randRange(rand, -0.15, 0.15);
    const len = size * randRange(rand, 0.75, 1.05);
    const width = size * randRange(rand, 0.4, 0.55);
    const px = x + Math.cos(angle) * len * 0.5;
    const py = y + Math.sin(angle) * len * 0.5;
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, len * 0.5, width * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  ctx.globalAlpha = Math.min(1, opacity + 0.15);
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(x, y, size * 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function buildLayerTexture(name: LayerName): HTMLCanvasElement {
  const cfg = LAYER_CONFIGS[name];
  const canvas = document.createElement('canvas');
  canvas.width = cfg.width;
  canvas.height = cfg.height;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, cfg.width, cfg.height);

  const rand = splitmix32(cfg.seed);

  // Foliage first (the "structure" layer), then flowers on top (the
  // "highlight" layer) — both constrained to sit below each column's
  // meadowTopFraction so the U silhouette is baked into the alpha mask.
  for (let i = 0; i < cfg.foliageCount; i++) {
    const xFrac = rand();
    const nx = xFrac * 2 - 1;
    const topFrac = meadowTopFraction(nx, cfg);
    const yFrac = randRange(rand, topFrac, 1.08); // slight overflow below the bottom edge
    const x = xFrac * cfg.width;
    const y = yFrac * cfg.height;
    const size = randRange(rand, cfg.foliageSize[0], cfg.foliageSize[1]);
    const opacity = randRange(rand, cfg.foliageOpacity[0], cfg.foliageOpacity[1]);
    const color = FOLIAGE_GRAYS[Math.floor(rand() * FOLIAGE_GRAYS.length)];
    drawFoliageMark(ctx, x, y, size, opacity, color);
  }

  for (let i = 0; i < cfg.flowerCount; i++) {
    const xFrac = rand();
    const nx = xFrac * 2 - 1;
    const topFrac = meadowTopFraction(nx, cfg);
    // Flowers stay slightly inset from the very top of the meadow band so
    // the boundary itself reads as foliage/haze, not a hard flower edge.
    const yFrac = randRange(rand, topFrac + 0.02, 1.05);
    const x = xFrac * cfg.width;
    const y = yFrac * cfg.height;
    const size = randRange(rand, cfg.flowerSize[0], cfg.flowerSize[1]);
    const opacity = randRange(rand, cfg.flowerOpacity[0], cfg.flowerOpacity[1]);
    const brightness = randRange(rand, cfg.brightnessRange[0], cfg.brightnessRange[1]);
    drawFlowerMark(ctx, x, y, size, opacity, brightness, rand);
  }

  // Single whole-canvas blur pass (cheap — one filtered drawImage call,
  // independent of how many marks were drawn) instead of per-mark blur.
  const layerBlurPx = Math.max(cfg.flowerBlurPx, cfg.foliageBlurPx);
  if (layerBlurPx > 0) {
    const blurred = document.createElement('canvas');
    blurred.width = cfg.width;
    blurred.height = cfg.height;
    const bctx = blurred.getContext('2d')!;
    bctx.filter = `blur(${layerBlurPx}px)`;
    bctx.drawImage(canvas, 0, 0);
    return blurred;
  }

  return canvas;
}

export interface MeadowTextures {
  far: THREE.CanvasTexture;
  mid: THREE.CanvasTexture;
  near: THREE.CanvasTexture;
  foreground: THREE.CanvasTexture;
}

let cached: MeadowTextures | null = null;

export function getMeadowTextures(): MeadowTextures {
  if (cached) return cached;
  if (typeof document === 'undefined') {
    throw new Error('getMeadowTextures() must only be called on the client');
  }
  const make = (name: LayerName) => {
    const tex = new THREE.CanvasTexture(buildLayerTexture(name));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  };
  cached = { far: make('far'), mid: make('mid'), near: make('near'), foreground: make('foreground') };
  return cached;
}

export function getLayerAspect(name: LayerName): number {
  const cfg = LAYER_CONFIGS[name];
  return cfg.width / cfg.height;
}
