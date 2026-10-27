'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { splitmix32, randRange } from './dandelionRandom';
import type { SeedConfig } from './dandelionField';

const VB_WIDTH = 12;
const VB_HEIGHT = 20;
const STALK_TOP: [number, number] = [6, 4];
const STALK_BOTTOM: [number, number] = [6, 15];

// A single detached dandelion seed: a short thin stalk with a tiny
// elongated achene (seed body) at one end and a small, asymmetric fan of
// pappus hairs at the other — not a symmetric 8-point star. Geometry is
// deterministic per instance via `shapeSeed` (see dandelionField.ts), so
// no two of the sixteen seeds in the field are identical, and hydration
// stays stable (same seed -> same shape, every render).
export function DandelionSeed({ config, active }: { config: SeedConfig; active: boolean }) {
  const { hairs, stalkBend } = useMemo(() => {
    const rand = splitmix32(config.shapeSeed);
    const hairCount = Math.round(randRange(rand, 5, 9));
    // Pappus points broadly "away" from the achene (upward, off the top of
    // the stalk) in an irregular fan, not a full-circle burst.
    const coneCenterDeg = randRange(rand, -100, -80);
    const coneSpreadDeg = randRange(rand, 100, 150);
    const hairList: Array<{ x2: number; y2: number; opacity: number }> = [];
    for (let i = 0; i < hairCount; i++) {
      const t = hairCount === 1 ? 0.5 : i / (hairCount - 1);
      const angleDeg = coneCenterDeg - coneSpreadDeg / 2 + t * coneSpreadDeg + randRange(rand, -9, 9);
      const angleRad = (angleDeg * Math.PI) / 180;
      const len = randRange(rand, 3.2, 5.6);
      hairList.push({
        x2: STALK_TOP[0] + Math.cos(angleRad) * len,
        y2: STALK_TOP[1] + Math.sin(angleRad) * len,
        opacity: randRange(rand, 0.4, 0.75),
      });
    }
    return { hairs: hairList, stalkBend: randRange(rand, -0.9, 0.9) };
  }, [config.shapeSeed]);

  return (
    <motion.div
      className="absolute pointer-events-none"
      aria-hidden="true"
      style={{
        left: `${config.leftPercent}%`,
        bottom: `${config.startBottomPercent}%`,
        willChange: 'transform, opacity',
      }}
      initial={{ x: 0, y: 0, rotate: 0, opacity: 0 }}
      animate={
        active
          ? {
              x: [0, config.driftXPx * 0.4, config.driftXPx],
              y: [0, -config.driftYPx * 0.5, -config.driftYPx],
              rotate: [0, config.rotateDeg * 0.5, config.rotateDeg],
              opacity: [0, 1, 1, 0],
            }
          : undefined
      }
      transition={{
        duration: config.durationS,
        delay: config.delayS,
        repeat: Infinity,
        ease: 'easeInOut',
        times: [0, 0.15, 0.85, 1],
      }}
    >
      <svg
        width={(VB_WIDTH / VB_HEIGHT) * 22 * config.scale}
        height={22 * config.scale}
        viewBox={`0 0 ${VB_WIDTH} ${VB_HEIGHT}`}
        style={{ overflow: 'visible' }}
      >
        <g stroke="#FFFFFF" strokeWidth={0.4} strokeLinecap="round">
          {hairs.map((h, i) => (
            <line key={i} x1={STALK_TOP[0]} y1={STALK_TOP[1]} x2={h.x2} y2={h.y2} opacity={h.opacity} />
          ))}
        </g>
        <path
          d={`M${STALK_TOP[0]},${STALK_TOP[1]} Q${STALK_TOP[0] + stalkBend},${(STALK_TOP[1] + STALK_BOTTOM[1]) / 2} ${STALK_BOTTOM[0]},${STALK_BOTTOM[1]}`}
          stroke="#FFFFFF"
          strokeWidth={0.35}
          fill="none"
          opacity={0.5}
        />
        <ellipse cx={STALK_BOTTOM[0]} cy={STALK_BOTTOM[1] + 1.3} rx={1} ry={1.6} fill="#FFFFFF" opacity={0.85} />
      </svg>
    </motion.div>
  );
}
