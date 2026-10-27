import { splitmix32, randRange } from './dandelionRandom';

// Fixed seeds — not "random each load". Changing these numbers changes the
// scene deterministically; leaving them alone reproduces the exact same
// composition on every render, every environment, forever.
const FLOWER_SEED = 4021;
const SEED_FIELD_SEED = 90210;

export type FlowerLayer = 'background' | 'midground' | 'foreground';
// Progressive reveal: 'always' shows on every breakpoint (mobile's 2-3
// "primary flowers"), 'tablet-up' adds the tablet layer, 'desktop-up' is
// desktop-only — this is what gives the composition its responsive depth
// without changing how many DOM nodes React has to reconcile per breakpoint.
export type FlowerVisibility = 'always' | 'tablet-up' | 'desktop-up';

export interface FlowerConfig {
  id: string;
  layer: FlowerLayer;
  visibility: FlowerVisibility;
  leftPercent: number;
  // Stem-base offset from the footer's bottom edge. Negative = the stem
  // continues past the edge, i.e. the flower is cropped by the footer
  // boundary (used for the 1-2 foreground flowers the brief calls for).
  bottomOffsetPx: number;
  heightPx: number;
  filamentCount: number;
  coreOpacity: number;
  glowOpacityValue: number;
  swayDurationS: number;
  swayDelayS: number;
  swayRotateDeg: number;
  swayDriftPx: number;
  // Static per-flower head rotation (not animated — this is what makes six
  // flowers built from the same code feel like six individual plants
  // rather than six clones of one rotated instance).
  headTiltDeg: number;
  jitterSeed: number;
}

export const FLOWER_FIELD: FlowerConfig[] = (() => {
  const rand = splitmix32(FLOWER_SEED);
  const specs: Array<Pick<FlowerConfig, 'id' | 'layer' | 'visibility' | 'leftPercent'>> = [
    { id: 'bg-1', layer: 'background', visibility: 'desktop-up', leftPercent: 11 },
    { id: 'bg-2', layer: 'background', visibility: 'desktop-up', leftPercent: 83 },
    // Pulled in from 26/66 to 16/84 (and given an extra tablet-only scale
    // tier in DandelionFooter.tsx): at 834px the previous position put this
    // flower's head directly over "Practice." — see the design review.
    // Moving both midground flowers further toward their edge keeps them
    // clear of the centered text column at tablet width and reads as a
    // more deliberately balanced left/right composition at every width.
    { id: 'mid-1', layer: 'midground', visibility: 'tablet-up', leftPercent: 16 },
    { id: 'mid-2', layer: 'midground', visibility: 'tablet-up', leftPercent: 84 },
    { id: 'fg-1', layer: 'foreground', visibility: 'always', leftPercent: 9 },
    { id: 'fg-2', layer: 'foreground', visibility: 'always', leftPercent: 91 },
  ];

  return specs.map((spec) => {
    const isBg = spec.layer === 'background';
    const isMid = spec.layer === 'midground';
    const isFg = spec.layer === 'foreground';

    const heightPx = isBg
      ? randRange(rand, 130, 165)
      : isMid
        ? randRange(rand, 190, 230)
        : randRange(rand, 300, 360);

    const baseFilamentCount = isBg ? 24 : isMid ? 30 : 38;

    return {
      ...spec,
      bottomOffsetPx: isFg ? -randRange(rand, 30, 70) : randRange(rand, 4, 20),
      heightPx,
      // +/- a handful of filaments per instance so density itself is part
      // of what makes each flower distinct, not just its geometry noise.
      filamentCount: Math.round(baseFilamentCount + randRange(rand, -4, 4)),
      // Background flowers used to skip glow and sit at low opacity behind
      // a 1.5px blur, which made them read as a shapeless smudge instead
      // of a receded flower — raised so the radial structure survives the
      // blur (see also the lighter blur value in DandelionFooter.tsx).
      coreOpacity: isBg ? randRange(rand, 0.34, 0.44) : isMid ? randRange(rand, 0.45, 0.55) : randRange(rand, 0.78, 0.92),
      glowOpacityValue: isBg ? randRange(rand, 0.08, 0.12) : isMid ? randRange(rand, 0.1, 0.14) : randRange(rand, 0.16, 0.22),
      swayDurationS: randRange(rand, 5, 10),
      swayDelayS: randRange(rand, 0, 4),
      swayRotateDeg: randRange(rand, 1, 3),
      swayDriftPx: randRange(rand, 2, 6),
      headTiltDeg: randRange(rand, -7, 7),
      jitterSeed: Math.floor(rand() * 1e9),
    };
  });
})();

export interface SeedConfig {
  id: string;
  leftPercent: number;
  startBottomPercent: number;
  scale: number;
  durationS: number;
  delayS: number;
  driftXPx: number;
  driftYPx: number;
  rotateDeg: number;
  alwaysVisible: boolean; // shown on mobile too (the 5-8 count); false = desktop/tablet-only (fills out to 10-18)
  shapeSeed: number; // per-seed procedural pappus/achene geometry — see DandelionSeed.tsx
}

// Shared wind bias: every seed drifts mostly the same direction (up and
// gently rightward) with its own small turbulence on top, rather than each
// seed picking a fully independent, uncorrelated heading — that's what
// separates "one wind field" from "confetti".
const WIND_X_BIAS_PX = 40;
const WIND_X_TURBULENCE_PX = 20;

export const SEED_FIELD: SeedConfig[] = (() => {
  const rand = splitmix32(SEED_FIELD_SEED);
  const TOTAL = 16;
  const ALWAYS_VISIBLE_COUNT = 6; // within the requested mobile range of 5-8

  const configs: SeedConfig[] = [];
  for (let i = 0; i < TOTAL; i++) {
    const alwaysVisible = i < ALWAYS_VISIBLE_COUNT;
    // A few of the desktop-only seeds are allowed to start higher up and
    // drift further, so the otherwise-empty upper-middle region gets a
    // little faint depth without adding a new element to the scene.
    const reachesUpper = !alwaysVisible && i % 3 === 0;

    const leftPercent = randRange(rand, 4, 96);
    // The footer's content column (headline/CTA/eyebrow) sits centered in
    // roughly the middle third of the width and the lower half of the
    // height. A seed whose horizontal position falls in that band starts
    // above it instead — its modest wind drift (see WIND_X_*_PX) isn't
    // enough to carry it back down into the text on the way up.
    const inContentColumn = leftPercent > 32 && leftPercent < 68;
    const startBottomPercent = inContentColumn
      ? randRange(rand, 46, reachesUpper ? 78 : 62)
      : randRange(rand, 5, reachesUpper ? 70 : 55);

    configs.push({
      id: `seed-${i}`,
      leftPercent,
      startBottomPercent,
      scale: randRange(rand, 0.7, 1.4),
      durationS: randRange(rand, 6, 12),
      delayS: randRange(rand, 0, 10),
      driftXPx: WIND_X_BIAS_PX + randRange(rand, -WIND_X_TURBULENCE_PX, WIND_X_TURBULENCE_PX),
      driftYPx: randRange(rand, 90, 220),
      rotateDeg: randRange(rand, 50, 160) * (rand() > 0.5 ? 1 : -1),
      alwaysVisible,
      shapeSeed: Math.floor(rand() * 1e9),
    });
  }
  return configs;
})();
