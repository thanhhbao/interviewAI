'use client';

import React, { useMemo } from 'react';
import { splitmix32, randRange } from './dandelionRandom';

// Fixed internal coordinate space — every flower is authored at this size
// and scaled up/down via the SVG's rendered width/height, so filament math
// only has to be written once.
const VB_WIDTH = 120;
const VB_HEIGHT = 160;
const HEAD_CX = 60;
const HEAD_CY = 40;
const HEAD_RADIUS = 34;
const STEM_TOP_Y = HEAD_CY + HEAD_RADIUS * 0.55;

interface DandelionSVGProps {
  heightPx: number;
  filamentCount: number;
  coreOpacity: number;
  glow: boolean;
  glowOpacity: number;
  jitterSeed: number;
  className?: string;
}

// One dandelion seed head, built from radiating filaments rather than a
// raster asset. Filaments do NOT all originate from one exact point (that
// reads as a firework/sparkler burst, not a fluffy seed head) — instead
// each root is scattered within a small irregular inner disc, angular
// spacing is deliberately uneven, filament length falls into short/
// medium/long bands, each filament bends slightly, and filaments are
// grouped into three opacity/weight bands so some hairs visually recede.
// A glow halo is a second, blurred, low-opacity copy of the same geometry
// sitting behind a crisp, unblurred core — never the only thing rendered.
export function DandelionSVG({
  heightPx,
  filamentCount,
  coreOpacity,
  glow,
  glowOpacity,
  jitterSeed,
  className,
}: DandelionSVGProps) {
  const { haloPath, faintPath, mediumPath, prominentPath, stemPath, tuftDots } = useMemo(() => {
    const rand = splitmix32(jitterSeed);

    // Stem bend — drawn from the same seed as the head, so every flower's
    // stem curves a little differently instead of reusing one fixed curve.
    // Kept small (a few px of lateral bend over ~120px) so it still reads
    // as "thin and elegant", not a wobble.
    const stemBendDir = rand() > 0.5 ? 1 : -1;
    const stemBendAmount = randRange(rand, 3, 9);
    const stemCtrl1X = HEAD_CX + stemBendDir * randRange(rand, 2, 5);
    const stemCtrl2X = HEAD_CX - stemBendDir * randRange(rand, 1, 4);
    const headAttachX = HEAD_CX + stemBendDir * stemBendAmount * 0.3;
    const stem = `M${HEAD_CX},${VB_HEIGHT} C${stemCtrl1X.toFixed(1)},${(VB_HEIGHT - 40).toFixed(1)} ${stemCtrl2X.toFixed(1)},${(STEM_TOP_Y + 30).toFixed(1)} ${headAttachX.toFixed(1)},${STEM_TOP_Y.toFixed(1)}`;

    // Irregular angular spacing: cumulative random weights instead of an
    // even 2*PI/N division with tiny jitter — produces genuinely uneven
    // gaps between filaments (a controlled irregularity, not a polar
    // grid) while still covering the full circle.
    const weights: number[] = [];
    for (let i = 0; i < filamentCount; i++) weights.push(0.5 + rand());
    const totalWeight = weights.reduce((a, b) => a + b, 0);

    const haloSegments: string[] = [];
    const bucketSegments: [string[], string[], string[]] = [[], [], []];
    const dots: Array<{ x: number; y: number; r: number }> = [];

    let cumAngle = randRange(rand, 0, Math.PI * 2);
    for (let i = 0; i < filamentCount; i++) {
      cumAngle += (weights[i] / totalWeight) * Math.PI * 2;
      const angle = cumAngle;

      // Origin spread: root point scattered inside a small inner disc
      // rather than the exact center — this is what turns "lines from a
      // point" into "hairs from a fluffy little sphere".
      const originR = randRange(rand, 0, HEAD_RADIUS * 0.22);
      const originAngle = rand() * Math.PI * 2;
      const ox = HEAD_CX + Math.cos(originAngle) * originR;
      const oy = HEAD_CY + Math.sin(originAngle) * originR;

      // Depth-banded length: short / medium / long, instead of one narrow
      // range that makes every filament read as visually equivalent.
      const bandRoll = rand();
      const length =
        bandRoll < 0.3
          ? randRange(rand, HEAD_RADIUS * 0.42, HEAD_RADIUS * 0.58)
          : bandRoll < 0.72
            ? randRange(rand, HEAD_RADIUS * 0.62, HEAD_RADIUS * 0.82)
            : randRange(rand, HEAD_RADIUS * 0.86, HEAD_RADIUS * 1.08);

      const tipX = ox + Math.cos(angle) * length;
      const tipY = oy + Math.sin(angle) * length;

      // Restrained curvature: a small perpendicular bow at the midpoint,
      // via a quadratic control point — enough to feel organic, nowhere
      // near "curly".
      const curveAmount = randRange(rand, -2.6, 2.6);
      const midX = (ox + tipX) / 2 - Math.sin(angle) * curveAmount;
      const midY = (oy + tipY) / 2 + Math.cos(angle) * curveAmount;

      const seg = `M${ox.toFixed(1)},${oy.toFixed(1)} Q${midX.toFixed(1)},${midY.toFixed(1)} ${tipX.toFixed(1)},${tipY.toFixed(1)}`;
      haloSegments.push(seg);

      // Opacity/weight bucket — some hairs recede, some are prominent.
      // Grouped into 3 combined <path> elements (not one per filament) so
      // this stays cheap: 3 DOM nodes regardless of filament count.
      const weightRoll = rand();
      const bucket = weightRoll < 0.3 ? 0 : weightRoll < 0.75 ? 1 : 2;
      bucketSegments[bucket].push(seg);

      // A tiny fork near the tip on a minority of filaments — enough to
      // read as pappus fluff up close without one DOM node per hair.
      if (rand() > 0.68) {
        const forkAngle = angle + randRange(rand, -0.4, 0.4);
        const forkLen = length * randRange(rand, 0.15, 0.28);
        const fx = tipX + Math.cos(forkAngle) * forkLen;
        const fy = tipY + Math.sin(forkAngle) * forkLen;
        bucketSegments[bucket].push(`M${tipX.toFixed(1)},${tipY.toFixed(1)} L${fx.toFixed(1)},${fy.toFixed(1)}`);
      }

      if (i % 4 === 0) {
        dots.push({ x: tipX, y: tipY, r: randRange(rand, 0.45, 0.95) });
      }
    }

    return {
      haloPath: haloSegments.join(' '),
      faintPath: bucketSegments[0].join(' '),
      mediumPath: bucketSegments[1].join(' '),
      prominentPath: bucketSegments[2].join(' '),
      stemPath: stem,
      tuftDots: dots,
    };
  }, [filamentCount, jitterSeed]);

  const aspect = VB_WIDTH / VB_HEIGHT;

  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox={`0 0 ${VB_WIDTH} ${VB_HEIGHT}`}
      width={heightPx * aspect}
      height={heightPx}
      className={className}
      style={{ overflow: 'visible' }}
    >
      {/* Halo — same geometry, blurred and low opacity, sitting behind the
          crisp core below. The soft center circle here is deliberately
          dim/large/secondary — there is no bright pinpoint anywhere in
          this file, which is what previously read as a lens-flare flash. */}
      {glow && (
        <g style={{ filter: 'blur(4px)' }} opacity={glowOpacity}>
          <path d={haloPath} stroke="#FFFFFF" strokeWidth={1.4} fill="none" strokeLinecap="round" />
          <circle cx={HEAD_CX} cy={HEAD_CY} r={HEAD_RADIUS * 0.5} fill="#FFFFFF" opacity={0.12} />
        </g>
      )}

      <path d={stemPath} stroke="#FFFFFF" strokeWidth={0.6} fill="none" opacity={coreOpacity * 0.45} strokeLinecap="round" />

      <g opacity={coreOpacity}>
        <path d={faintPath} stroke="#FFFFFF" strokeWidth={0.5} strokeOpacity={0.4} fill="none" strokeLinecap="round" />
        <path d={mediumPath} stroke="#FFFFFF" strokeWidth={0.65} strokeOpacity={0.68} fill="none" strokeLinecap="round" />
        <path d={prominentPath} stroke="#FFFFFF" strokeWidth={0.85} strokeOpacity={0.92} fill="none" strokeLinecap="round" />
        {tuftDots.map((d, i) => (
          <circle key={i} cx={d.x} cy={d.y} r={d.r} fill="#FFFFFF" opacity={0.6} />
        ))}
      </g>
    </svg>
  );
}
