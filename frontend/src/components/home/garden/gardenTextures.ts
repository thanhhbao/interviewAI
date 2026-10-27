import * as THREE from 'three';
import { splitmix32, randRange } from '@/components/home/footer/dandelionRandom';

// ASSET PROVENANCE: these textures are procedurally generated on an
// offscreen <canvas> at runtime — nothing is photographed, drawn by hand,
// or downloaded from any third party, and nothing is copied from the
// Framer reference component (we are reproducing its composition/
// interaction principles, never its asset files). No hotlinked or
// externally-sourced image assets are used anywhere in this scene. This
// is a deliberate prototype placeholder: if/when real licensed meadow-
// flower photography or botanical renders are sourced, they should
// replace `buildFlowerTexture` below (keep the `FLOWER_STATE_COUNT`
// export and texture shape so FlowerField.tsx doesn't need to change).
//
// Each variant is deterministic (seeded splitmix32) — same seed always
// produces the same canvas pixels, so texture generation never causes
// hydration or frame-to-frame drift.

const TEXTURE_SIZE = 96; // small sprite — hundreds of these on screen at once, not a handful of hero flowers

// Simple daisy/meadow-flower silhouette instead of the earlier dandelion
// starburst — this pass explicitly deprioritizes exact botanical species
// in favor of density/composition matching the reference (a "carpet" of
// small round flowers reads correctly at small scale; a thin dandelion
// filament burst does not — it disappears or aliases badly at this size).
// A handful of overlapping petal shapes around a bright center, each
// petal/variant using the requested monochrome palette.
const PETAL_COLORS = ['#ECEDE9', '#D8DDDA', '#BFC6C2'];
const CENTER_COLOR = '#FFFFFF';

interface FlowerStatePreset {
  seed: number;
  petalColor: string;
  petalCount: number;
  openness: number; // 0.7-1.15 — how far petals spread from center
}

const FLOWER_STATES: FlowerStatePreset[] = [
  { seed: 51001, petalColor: PETAL_COLORS[0], petalCount: 7, openness: 1.0 },
  { seed: 51002, petalColor: PETAL_COLORS[1], petalCount: 6, openness: 0.85 },
  { seed: 51003, petalColor: PETAL_COLORS[2], petalCount: 8, openness: 1.1 },
];
export const FLOWER_STATE_COUNT = FLOWER_STATES.length;

function buildFlowerTexture(preset: FlowerStatePreset): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = TEXTURE_SIZE;
  canvas.height = TEXTURE_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

  const rand = splitmix32(preset.seed);
  const cx = TEXTURE_SIZE * 0.5;
  const cy = TEXTURE_SIZE * 0.5;
  const petalR = TEXTURE_SIZE * 0.3 * preset.openness;

  // Soft halo first (cheap glow without a bloom pass) — same "never a
  // single bright point alone" principle as the earlier dandelion pass.
  ctx.save();
  ctx.filter = 'blur(3px)';
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = '#FFFFFF';
  ctx.beginPath();
  ctx.arc(cx, cy, petalR * 0.85, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Petals — simple overlapping ellipses radiating from center, angle/
  // size jitter per petal so the silhouette isn't a mechanically perfect
  // flower icon.
  ctx.fillStyle = preset.petalColor;
  const angleStep = (Math.PI * 2) / preset.petalCount;
  let angle = randRange(rand, 0, Math.PI * 2);
  for (let i = 0; i < preset.petalCount; i++) {
    angle += angleStep + randRange(rand, -0.12, 0.12);
    const len = petalR * randRange(rand, 0.82, 1.05);
    const width = petalR * randRange(rand, 0.36, 0.48);
    const px = cx + Math.cos(angle) * len * 0.55;
    const py = cy + Math.sin(angle) * len * 0.55;
    ctx.save();
    ctx.globalAlpha = randRange(rand, 0.75, 0.95);
    ctx.translate(px, py);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, len * 0.5, width * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Bright center — the flower's visual anchor, always the brightest
  // point but small and soft, never a lens-flare-style flash.
  ctx.beginPath();
  ctx.fillStyle = CENTER_COLOR;
  ctx.globalAlpha = 0.95;
  ctx.arc(cx, cy, petalR * 0.22, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  return canvas;
}

// A short thin stalk, tiny achene, and a small asymmetric pappus fan — the
// same detached-seed shape used in earlier passes, rasterized for use as
// a billboard/instance texture in 3D space. Kept for the sparse
// atmospheric seed layer (unchanged this pass).
function buildSeedTexture(seed: number): HTMLCanvasElement {
  const size = 96;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const rand = splitmix32(seed);

  const topX = size * 0.5;
  const topY = size * 0.3;
  const bottomY = size * 0.78;

  const hairCount = Math.round(randRange(rand, 6, 10));
  const coneCenterDeg = randRange(rand, -100, -80);
  const coneSpreadDeg = randRange(rand, 100, 150);
  ctx.strokeStyle = 'rgba(255,255,255,0.65)';
  ctx.lineWidth = 1;
  for (let i = 0; i < hairCount; i++) {
    const t = hairCount === 1 ? 0.5 : i / (hairCount - 1);
    const angleDeg = coneCenterDeg - coneSpreadDeg / 2 + t * coneSpreadDeg + randRange(rand, -9, 9);
    const angleRad = (angleDeg * Math.PI) / 180;
    const len = randRange(rand, size * 0.22, size * 0.34);
    ctx.beginPath();
    ctx.moveTo(topX, topY);
    ctx.lineTo(topX + Math.cos(angleRad) * len, topY + Math.sin(angleRad) * len);
    ctx.globalAlpha = randRange(rand, 0.4, 0.75);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const stalkBend = randRange(rand, -3, 3);
  ctx.beginPath();
  ctx.moveTo(topX, topY);
  ctx.quadraticCurveTo(topX + stalkBend, (topY + bottomY) / 2, topX, bottomY);
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 0.8;
  ctx.stroke();

  ctx.beginPath();
  ctx.ellipse(topX, bottomY + size * 0.045, size * 0.035, size * 0.06, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.fill();

  return canvas;
}

// Foliage: soft, irregular dark shapes — not individual grass blades, just
// enough shape variation that a field of them doesn't tile visibly. Always
// darker than the flower petals/centers so flowers stay the brightest
// element; exists to act as the field's foundation mass, filling the
// black gaps between flower heads at carpet density (see
// gardenField.ts's buildFoliage). Footer-scale pass: four distinct
// silhouettes (was three near-identical blob variants) and a darker
// palette per spec, so the "foundation" reads as varied dark botanical
// mass rather than a repeating dot.
const FOLIAGE_COLORS = ['#242927', '#2e3531', '#39413c', '#464e49'];

type FoliageShape = 'cluster' | 'tuft' | 'broad' | 'branch';

interface FoliagePreset {
  seed: number;
  color: string;
  lobeCount: number;
  shape: FoliageShape;
}

const FOLIAGE_STATES: FoliagePreset[] = [
  { seed: 61001, color: FOLIAGE_COLORS[0], lobeCount: 4, shape: 'cluster' },
  { seed: 61002, color: FOLIAGE_COLORS[1], lobeCount: 6, shape: 'tuft' },
  { seed: 61003, color: FOLIAGE_COLORS[2], lobeCount: 3, shape: 'broad' },
  { seed: 61004, color: FOLIAGE_COLORS[3], lobeCount: 5, shape: 'branch' },
];
export const FOLIAGE_STATE_COUNT = FOLIAGE_STATES.length;

function buildFoliageTexture(preset: FoliagePreset): HTMLCanvasElement {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, size, size);

  const rand = splitmix32(preset.seed);
  const cx = size * 0.5;
  const cy = size * 0.58;
  const r = size * 0.34;

  ctx.filter = 'blur(2px)';
  ctx.fillStyle = preset.color;

  if (preset.shape === 'tuft') {
    // Narrow grass tuft: thin tall triangles fanning from a base point.
    const base = { x: cx, y: cy + r * 0.5 };
    for (let i = 0; i < preset.lobeCount; i++) {
      const t = preset.lobeCount === 1 ? 0.5 : i / (preset.lobeCount - 1);
      const angle = -Math.PI / 2 + (t - 0.5) * 1.1 + randRange(rand, -0.08, 0.08);
      const len = r * randRange(rand, 1.1, 1.6);
      const width = r * randRange(rand, 0.1, 0.16);
      const tipX = base.x + Math.cos(angle) * len;
      const tipY = base.y + Math.sin(angle) * len;
      const nx = -Math.sin(angle) * width;
      const ny = Math.cos(angle) * width;
      ctx.globalAlpha = randRange(rand, 0.6, 0.85);
      ctx.beginPath();
      ctx.moveTo(base.x - nx, base.y - ny);
      ctx.lineTo(base.x + nx, base.y + ny);
      ctx.lineTo(tipX, tipY);
      ctx.closePath();
      ctx.fill();
    }
  } else if (preset.shape === 'broad') {
    // Broad low foliage: a small number of large, flattened, overlapping
    // lobes — a low ground-hugging mass rather than upright shape.
    for (let i = 0; i < preset.lobeCount; i++) {
      const angle = (i / preset.lobeCount) * Math.PI * 2 + randRange(rand, -0.25, 0.25);
      const dist = r * randRange(rand, 0.1, 0.3);
      const lobeR = r * randRange(rand, 0.75, 1.05);
      ctx.globalAlpha = randRange(rand, 0.55, 0.8);
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist * 0.5, lobeR, lobeR * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (preset.shape === 'branch') {
    // Branching leaf cluster: a few thin stalks each ending in a small
    // leaf blob, radiating from a shared base.
    const base = { x: cx, y: cy + r * 0.55 };
    for (let i = 0; i < preset.lobeCount; i++) {
      const angle = -Math.PI / 2 + randRange(rand, -0.9, 0.9);
      const len = r * randRange(rand, 0.7, 1.2);
      const endX = base.x + Math.cos(angle) * len;
      const endY = base.y + Math.sin(angle) * len;
      ctx.globalAlpha = randRange(rand, 0.5, 0.75);
      ctx.lineWidth = r * 0.08;
      ctx.strokeStyle = preset.color;
      ctx.beginPath();
      ctx.moveTo(base.x, base.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      const leafR = r * randRange(rand, 0.22, 0.34);
      ctx.beginPath();
      ctx.ellipse(endX, endY, leafR, leafR * 0.65, angle, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    // Short leaf cluster (default): overlapping rounded lobes around a
    // center point — the original blob shape, kept as one of four now.
    for (let i = 0; i < preset.lobeCount; i++) {
      const angle = (i / preset.lobeCount) * Math.PI * 2 + randRange(rand, -0.3, 0.3);
      const dist = r * randRange(rand, 0.15, 0.45);
      const lobeR = r * randRange(rand, 0.55, 0.9);
      ctx.globalAlpha = randRange(rand, 0.55, 0.85);
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist * 0.7, lobeR, lobeR * 0.75, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  return canvas;
}

export interface GardenTextures {
  flowers: THREE.CanvasTexture[];
  seed: THREE.CanvasTexture;
  foliage: THREE.CanvasTexture[];
}

let cached: GardenTextures | null = null;

// Lazily built once, on the client only (canvas isn't available during
// SSR) — cached at module scope so every consumer of the garden scene
// shares the same texture instances instead of re-rasterizing.
export function getGardenTextures(): GardenTextures {
  if (cached) return cached;
  if (typeof document === 'undefined') {
    throw new Error('getGardenTextures() must only be called on the client');
  }

  const flowers = FLOWER_STATES.map((preset) => {
    const tex = new THREE.CanvasTexture(buildFlowerTexture(preset));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  });

  const seedTex = new THREE.CanvasTexture(buildSeedTexture(4471));
  seedTex.colorSpace = THREE.SRGBColorSpace;
  seedTex.needsUpdate = true;

  const foliage = FOLIAGE_STATES.map((preset) => {
    const tex = new THREE.CanvasTexture(buildFoliageTexture(preset));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  });

  cached = { flowers, seed: seedTex, foliage };
  return cached;
}
