import { splitmix32, randRange } from '@/components/home/footer/dandelionRandom';
import { FLOWER_STATE_COUNT, FOLIAGE_STATE_COUNT } from './gardenTextures';

// World-space layout. The camera travels along -Z (see CameraRig.tsx),
// gliding just above a dense, continuous flower carpet rather than
// weaving between a handful of giant plants — see the reference-pass
// rewrite of `buildFlowers` below.
//
// Footer-scale pass: this was a 78-unit camera traverse ("a separate 3D
// chapter"), sized for a full-screen standalone scene. A footer only
// needs enough travel to show parallax/depth, not a journey — shrunk to
// a 24-unit traverse. This does NOT touch the approved valley arc
// (that's a function of X via terrainHeight, untouched below); it only
// shortens how far along Z the corridor/camera extend.
export const CORRIDOR_NEAR_Z = 15;
export const CORRIDOR_FAR_Z = -34;
export const CAMERA_START_Z = 13;
export const CAMERA_END_Z = -11;

// Geometry pass: a broad, shallow U/bowl — the terrain rise at the edges
// must be large enough to read as a valley SILHOUETTE on screen (the
// field/sky boundary sitting visibly higher at the left/right edges than
// at center), not just a subtle bump under the flowers. The earlier
// VALLEY_RISE=0.9 was tuned for "gentle undulation under a flat-looking
// field" and produced only a ~2pt screen-space difference between center
// and edge horizon height — invisible in practice. These values were
// solved by directly projecting (x, terrainHeight(x,z), z) sample points
// through the real camera (same technique as the density passes): a
// VALLEY_HALF_WIDTH roughly matching the flower field's own outer extent
// (so the rise is "used up" by where flowers actually reach, not wasted
// past it) combined with VALLEY_RISE=8 produces close to a 15-point
// screen-space separation between the center and edge horizon at the
// establishing view — within the requested ~12-17pt / ~25-33% vs
// ~40-47% target band.
const VALLEY_HALF_WIDTH = 16;
const VALLEY_RISE = 8;
const VALLEY_EXPONENT = 2.0;

export function terrainHeight(x: number, z: number): number {
  const edgeT = Math.min(1, Math.abs(x) / VALLEY_HALF_WIDTH);
  const edgeRise = Math.pow(edgeT, VALLEY_EXPONENT) * VALLEY_RISE;
  // Side rise eases out gradually with depth rather than vanishing — the
  // valley silhouette should stay coherent as the camera travels, not
  // flatten out partway through the corridor.
  const depthEase = 1 - Math.min(1, Math.max(0, (CORRIDOR_NEAR_Z - z) / (CORRIDOR_NEAR_Z - CORRIDOR_FAR_Z))) * 0.15;
  const roll = Math.sin(x * 0.07 + z * 0.045) * 0.22 + Math.sin(z * 0.05 - x * 0.02 + 0.6) * 0.16;
  return edgeRise * depthEase + roll;
}

export interface FlowerInstance {
  x: number;
  y: number; // root y (terrain height at x,z)
  z: number;
  stemHeight: number;
  scale: number;
  rotationY: number;
  tilt: number;
  variant: number;
  windPhase: number;
  windAmp: number;
  // 'center' = fine dense band the camera is always looking through
  // (kept aggressively on mobile/tablet); 'outer' = coarser band only
  // relevant once the wide far cone reaches it (thinned more aggressively
  // on mobile/tablet) — see the tier-subsample comment in buildFlowers.
  zone: 'center' | 'outer';
  // Per-instance depth brightness (near ~0.9-1.0, far ~0.3-0.5), applied
  // via InstancedMesh.setColorAt in FlowerField.tsx — softens the
  // distant field into the fog instead of letting hundreds of equally-
  // bright far sprites read as a hard white horizon line.
  brightness: number;
}

export interface FoliageInstance {
  x: number;
  y: number;
  z: number;
  scale: number;
  rotationY: number;
  variant: number;
  zone: 'center' | 'outer';
  brightness: number;
}

export interface SeedInstance {
  x: number;
  y: number;
  z: number;
  scale: number;
  phase: number;
  speed: number;
  driftX: number;
  wobble: number;
}

export type GardenTier = 'desktop' | 'tablet' | 'mobile';

// Continuous carpet, not named clusters — a jittered grid, most cells
// filled (roughly-uniform inclusion probability) so the field reads as
// one continuous meadow.
//
// The grid is split by X, not Z, into a fine CENTRAL band and coarser
// OUTER bands — found by directly projecting candidate distributions
// through the actual camera and measuring where instances land on
// screen (not by eyeballing screenshots): a camera that's elevated and
// forward-looking has a NARROW view cone at close range (a few world
// units wide) and a WIDE one at long range. A uniform grid, or one only
// split by depth, puts most of its density where the cone is wide (far)
// and starves where it's narrow (near) — which is exactly why the first
// version of this pass showed almost nothing below the screen's
// vertical midpoint. Splitting by X instead keeps the central corridor
// (which is what's actually in view at close range, at ANY point along
// the camera's path, not just at the start) consistently dense, while
// the outer bands — only relevant once the wide far cone reaches them —
// stay at the coarser original spacing.
//
// Reference-pass follow-up (carpet density pass): a projection-verified
// X-band split was necessary but not sufficient — the lower-frame screen
// coverage was still too thin because the near cone, while correctly
// prioritized, wasn't dense enough in absolute terms, and there was no
// mass filling the gaps *between* flower heads (see FoliageField). Cell
// sizes below were tightened and the center band widened so the always-
// in-view corridor reads as a continuous carpet, not a sparse line.
// Reference-pass follow-up #2: even with the near-field billboard fix
// (see FlowerField.tsx/FoliageField.tsx), coverage was still too sparse
// low in frame. The reason is geometric, not a density bug per se: with
// the camera pitched down and close to the ground, the ground area that
// projects into the BOTTOM HALF of the screen is a small wedge only ~1-2
// world units deep and ~2-3 units wide directly ahead of the camera —
// verified by projecting that wedge back through the actual camera math.
// A cell size tuned for "the whole corridor reads as dense from a
// distance" is nowhere near fine enough to also densely fill that small
// near wedge. Cell size shrunk again here specifically to cover that
// wedge, not just the corridor-at-a-glance.
const CENTER_X_HALF = 6.5;
const CENTER_CELL = 0.65;
const OUTER_X_HALF = 15;
const OUTER_CELL_X = 1.7;
const OUTER_CELL_Z = 1.9;
const BASE_INCLUSION = 0.84;

// Footer-scale pass: the camera travels through the ENTIRE Z corridor
// over the course of the scroll, so every instance is momentarily "the
// closest thing to the camera" at some point — including ones placed
// almost exactly on the camera's own X wander line (x≈0). At that
// moment, distance collapses to roughly just the camera's height above
// the ground (~1.1-1.3 units), and a billboard that close blows up into
// a huge foreground blob under perspective regardless of its assigned
// "scale" — this was the "giant flower" bug flagged in review.
//
// First attempt pushed near-center points outward in X to keep them out
// of the flight line — this technically capped the max blob size, but it
// hollowed out a visible void straight down the center of the frame at
// EVERY scroll position (points are static in world space; the push
// affects all Z, not just the moment the camera happens to be nearby),
// which is worse than the bug it fixed. Replaced with a scale clamp
// instead: instances near the camera's X wander line are proportionally
// shrunk (not moved), so the position/density grid stays intact and only
// their projected size near a close pass is bounded.
// Diagnosed with fixed reference markers rendered at the same world
// position (proved the camera/render pipeline itself was fine at every
// checkpoint, including the footer's end) plus a live per-instance data
// dump (proved real, reasonably-positioned flowers exist directly ahead
// of the camera at every checkpoint too): the near-empty look at 100%
// scroll was this clamp being too aggressive exactly where it matters
// most — the camera sits at x≈0 for most of the journey (see
// WANDER_KEYFRAMES in CameraRig.tsx), which is exactly the zone this
// clamp shrinks hardest. 0.22 was tuned only against the worst-case
// theoretical collision distance, not against how it reads over an
// entire journey where the camera spends most of its time near x=0.
const NEAR_CLAMP_X = 2.0;
const NEAR_CLAMP_MIN = 0.55;

function nearScaleMultiplier(x: number): number {
  const t = Math.min(1, Math.abs(x) / NEAR_CLAMP_X);
  return NEAR_CLAMP_MIN + (1 - NEAR_CLAMP_MIN) * t;
}

// Depth taper reference span — deliberately much longer than the actual
// corridor (CORRIDOR_NEAR_Z to CORRIDOR_FAR_Z, ~49 units). This was
// found the hard way: brightness/scale taper used to be computed against
// the corridor's own length, so a flower's "how near is it" was really
// "how near is it to the STARTING edge of the whole corridor" — fine
// when the corridor was long relative to the camera's reach, but once
// the corridor was shortened for the footer-scale pass, the camera's own
// END position already sat deep in the "far/dim/small" part of that
// gradient, making the field collapse to near-nothing exactly at 100%
// scroll (verified via a live per-instance dump: flowers directly ahead
// of the camera at progress=1 existed with reasonable positions, but
// their scale (~0.05-0.28) and brightness (~0.5-0.6) were computed as if
// they were background clutter, because they were absolutely far from
// CORRIDOR_NEAR_Z even though they were right in front of the camera).
// A wide, fixed reference span decouples the taper from corridor length,
// so shortening the corridor for the footer doesn't also compress the
// "near" portion of the brightness/scale gradient.
const DEPTH_GRADIENT_SPAN = 140;

function depthGradientT(z: number): number {
  return Math.min(1, Math.max(0, (CORRIDOR_NEAR_Z - z) / DEPTH_GRADIENT_SPAN));
}

// Depth brightness — near stays close to full brightness, far dims
// toward the fog/background color well before fog alone would fully
// hide it, so a dense far band reads as a soft dissolve rather than a
// crisp line of equally-bright dots. Floor raised from 0.25 — verified
// with solid-color (non-textured) reference geometry that positions/
// counts were correct throughout the corridor, but the thin multi-petal
// texture read as visually sparse at the lower brightness values,
// especially compounded with the small carpet scale.
function depthBrightness(depthT: number, rand: () => number): number {
  const base = 0.95 + (0.55 - 0.95) * depthT;
  const jitter = (rand() * 2 - 1) * 0.05;
  return Math.min(1, Math.max(0.45, base + jitter));
}

function scatterBand(
  rand: () => number,
  xLo: number,
  xHi: number,
  zLo: number,
  zHi: number,
  cellX: number,
  cellZ: number
): Array<{ x: number; z: number }> {
  const cols = Math.max(1, Math.round((xHi - xLo) / cellX));
  const rows = Math.max(1, Math.round((zHi - zLo) / cellZ));
  const points: Array<{ x: number; z: number }> = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const centerX = xLo + (c + 0.5) * ((xHi - xLo) / cols);
      const centerZ = zLo + (r + 0.5) * ((zHi - zLo) / rows);
      if (rand() > BASE_INCLUSION) continue;
      points.push({
        x: centerX + randRange(rand, -cellX * 0.4, cellX * 0.4),
        z: centerZ + randRange(rand, -cellZ * 0.4, cellZ * 0.4),
      });
    }
  }
  return points;
}

function buildFlowers(seedValue: number, tier: GardenTier): FlowerInstance[] {
  const rand = splitmix32(seedValue);
  const all: FlowerInstance[] = [];

  const zMin = CORRIDOR_FAR_Z + 6;
  const zMax = CORRIDOR_NEAR_Z;
  const zoned: Array<{ x: number; z: number; zone: 'center' | 'outer' }> = [
    ...scatterBand(rand, -CENTER_X_HALF, CENTER_X_HALF, zMin, zMax, CENTER_CELL, CENTER_CELL).map((p) => ({
      ...p,
      zone: 'center' as const,
    })),
    ...scatterBand(rand, -OUTER_X_HALF, -CENTER_X_HALF, zMin, zMax, OUTER_CELL_X, OUTER_CELL_Z).map((p) => ({
      ...p,
      zone: 'outer' as const,
    })),
    ...scatterBand(rand, CENTER_X_HALF, OUTER_X_HALF, zMin, zMax, OUTER_CELL_X, OUTER_CELL_Z).map((p) => ({
      ...p,
      zone: 'outer' as const,
    })),
  ];

  for (const rawPoint of zoned) {
      const { x, z, zone } = rawPoint;
      // Depth taper: 0 near CORRIDOR_NEAR_Z, approaching 1 far beyond the
      // camera's actual reach — see DEPTH_GRADIENT_SPAN above for why
      // this is NOT simply "0 at the near edge, 1 at the far edge" of the
      // (now much shorter) corridor.
      const depthT = depthGradientT(z);
      const depthScale = 1 - depthT * 0.5;
      // Narrow scale distribution ("many related small flowers", not
      // "tiny/tiny/GIANT/tiny") — combined with the near-camera scale
      // clamp, this keeps every flower's projected screen size within a
      // small-to-moderate band instead of relying on distance alone to
      // avoid huge foreground blobs.
      const scale = randRange(rand, 0.26, 0.48) * depthScale * nearScaleMultiplier(x);
      const stemHeight = scale * randRange(rand, 0.35, 0.7);

      all.push({
        x,
        y: terrainHeight(x, z),
        z,
        stemHeight,
        scale,
        rotationY: randRange(rand, 0, Math.PI * 2),
        tilt: randRange(rand, -0.05, 0.05),
        variant: Math.floor(rand() * FLOWER_STATE_COUNT),
        windPhase: randRange(rand, 0, Math.PI * 2),
        windAmp: randRange(rand, 0.8, 1.2) * (1 - depthT * 0.5),
        zone,
        brightness: depthBrightness(depthT, rand),
      });
  }

  // Tablet/mobile: retain by ZONE, not a uniform stride — the center
  // zone is what's actually in view at close range at any point along
  // the camera's path (the near/mid field), so it's kept far more
  // aggressively than the outer zone (the far field, which perspective
  // compresses anyway). A uniform stride was tested first and produced
  // a mobile field that thinned evenly everywhere, including exactly
  // the near band that most needs to stay dense.
  if (tier === 'desktop') return all;
  const centerStride = tier === 'tablet' ? 1 : 2; // tablet keeps ~100% of center, mobile ~50%
  const outerStride = tier === 'tablet' ? 2 : 4; // tablet ~50% of outer, mobile ~25%
  let centerI = 0;
  let outerI = 0;
  return all.filter((f) => {
    if (f.zone === 'center') return centerI++ % centerStride === 0;
    return outerI++ % outerStride === 0;
  });
}

const SEED_COUNTS: Record<GardenTier, number> = { desktop: 16, tablet: 11, mobile: 7 };

// Cheap dark vegetation mass — small blob billboards rooted at the same
// terrain surface as the flowers, scattered across a wider band than the
// flower field itself. Purpose is purely to eliminate empty black gaps
// BETWEEN flower heads at carpet density; never brighter than the
// flowers (see gardenTextures.ts's foliage palette).
const FOLIAGE_X_HALF = 15;
const FOLIAGE_CELL_X = 0.7;
const FOLIAGE_CELL_Z = 0.8;
const FOLIAGE_INCLUSION = 0.68;

function buildFoliage(seedValue: number, tier: GardenTier): FoliageInstance[] {
  const rand = splitmix32(seedValue);
  const zMin = CORRIDOR_FAR_Z + 6;
  const zMax = CORRIDOR_NEAR_Z;
  const cols = Math.max(1, Math.round((FOLIAGE_X_HALF * 2) / FOLIAGE_CELL_X));
  const rows = Math.max(1, Math.round((zMax - zMin) / FOLIAGE_CELL_Z));
  const all: FoliageInstance[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (rand() > FOLIAGE_INCLUSION) continue;
      const x = -FOLIAGE_X_HALF + (c + 0.5) * ((FOLIAGE_X_HALF * 2) / cols) + randRange(rand, -FOLIAGE_CELL_X * 0.4, FOLIAGE_CELL_X * 0.4);
      const z = zMin + (r + 0.5) * ((zMax - zMin) / rows) + randRange(rand, -FOLIAGE_CELL_Z * 0.4, FOLIAGE_CELL_Z * 0.4);
      const depthT = depthGradientT(z);
      // Narrower than the previous pass — foliage is foundation mass, not
      // a second layer of large shapes competing with the flowers. Near-
      // camera clamp applied more gently than flowers (foliage isn't the
      // "hero" element a close pass would make look wrong).
      const scale = randRange(rand, 0.3, 0.55) * (1 - depthT * 0.4) * (0.5 + 0.5 * nearScaleMultiplier(x));
      all.push({
        x,
        y: terrainHeight(x, z),
        z,
        scale,
        rotationY: randRange(rand, 0, Math.PI * 2),
        variant: Math.floor(rand() * FOLIAGE_STATE_COUNT),
        zone: Math.abs(x) <= CENTER_X_HALF ? 'center' : 'outer',
        brightness: depthBrightness(depthT, rand),
      });
    }
  }

  if (tier === 'desktop') return all;
  const centerStride = tier === 'tablet' ? 1 : 2;
  const outerStride = tier === 'tablet' ? 2 : 4;
  let centerI = 0;
  let outerI = 0;
  return all.filter((f) => {
    if (f.zone === 'center') return centerI++ % centerStride === 0;
    return outerI++ % outerStride === 0;
  });
}

function buildSeeds(seedValue: number, tier: GardenTier): SeedInstance[] {
  const rand = splitmix32(seedValue);
  const count = SEED_COUNTS[tier];
  const seeds: SeedInstance[] = [];
  for (let i = 0; i < count; i++) {
    seeds.push({
      x: randRange(rand, -12, 12),
      y: randRange(rand, 0.3, 3.2),
      z: randRange(rand, CORRIDOR_FAR_Z + 10, CORRIDOR_NEAR_Z - 2),
      scale: randRange(rand, 0.14, 0.32),
      phase: randRange(rand, 0, Math.PI * 2),
      speed: randRange(rand, 0.15, 0.4),
      driftX: randRange(rand, -0.2, 0.5),
      wobble: randRange(rand, 0.3, 0.9),
    });
  }
  return seeds;
}

const FIELD_SEED = 552013;

const flowerCache = new Map<GardenTier, FlowerInstance[]>();
const seedCache = new Map<GardenTier, SeedInstance[]>();
const foliageCache = new Map<GardenTier, FoliageInstance[]>();

export function getFlowerField(tier: GardenTier): FlowerInstance[] {
  if (!flowerCache.has(tier)) flowerCache.set(tier, buildFlowers(FIELD_SEED, tier));
  return flowerCache.get(tier)!;
}

export function getSeedField(tier: GardenTier): SeedInstance[] {
  if (!seedCache.has(tier)) seedCache.set(tier, buildSeeds(FIELD_SEED ^ 0x2545f491, tier));
  return seedCache.get(tier)!;
}

export function getFoliageField(tier: GardenTier): FoliageInstance[] {
  if (!foliageCache.has(tier)) foliageCache.set(tier, buildFoliage(FIELD_SEED ^ 0x9e3779b9, tier));
  return foliageCache.get(tier)!;
}
