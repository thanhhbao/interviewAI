'use client';

import React, { useCallback, useRef } from 'react';
import { useCanvasLoop } from './useCanvasLoop';

// Native clone of the Framer "Stars Galaxy" component's actual mechanism —
// true perspective-projected depth (not a flat random-dot background):
// each point has a Z position that decreases over time (moving toward the
// viewer), is projected through a pinhole-camera transform so closer points
// render larger and further from center, and respawns at the far plane once
// it passes the camera. Interpreted as sparse, dim data points rather than
// literal stars — no nebula, no bright field, no nebula nozzle.

const COUNT = 110;
const SPEED = 0.055; // z-units per second
const MIN_Z = 0.06;
const MAX_Z = 1;

interface Point {
  x: number; // -1..1 space coordinate
  y: number;
  z: number;
  tone: 'dim' | 'teal' | 'bright';
  phase: number;
  prevSX: number | null;
  prevSY: number | null;
}

function spawn(near = false): Point {
  const roll = Math.random();
  return {
    x: (Math.random() * 2 - 1) * 1.4,
    y: (Math.random() * 2 - 1) * 1.4,
    z: near ? MAX_Z : MIN_Z + Math.random() * (MAX_Z - MIN_Z),
    tone: roll > 0.94 ? 'bright' : roll > 0.72 ? 'teal' : 'dim',
    phase: Math.random() * Math.PI * 2,
    prevSX: null,
    prevSY: null,
  };
}

export function StarsDataField({ className }: { className?: string }) {
  const pointsRef = useRef<Point[] | null>(null);
  const lastT = useRef(0);

  if (!pointsRef.current) {
    pointsRef.current = Array.from({ length: COUNT }, () => spawn(true));
  }

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    const dt = Math.min(0.05, t - lastT.current || 0.016);
    lastT.current = t;

    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const focal = Math.min(w, h) * 0.5;
    const points = pointsRef.current!;

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      p.z -= SPEED * dt;
      if (p.z <= MIN_Z) {
        points[i] = spawn(false);
        continue;
      }

      const sx = cx + (p.x / p.z) * focal * 0.5;
      const sy = cy + (p.y / p.z) * focal * 0.5;
      if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) {
        p.prevSX = null;
        p.prevSY = null;
        continue;
      }

      const depthFactor = 1 - (p.z - MIN_Z) / (MAX_Z - MIN_Z); // 0 far .. 1 near
      const size = 0.7 + depthFactor * 2.6;
      const twinkle = 0.55 + 0.45 * Math.sin(t * 1.2 + p.phase);
      const baseAlpha = (0.3 + depthFactor * 0.65) * twinkle;

      const color =
        p.tone === 'bright' ? `120, 255, 231` : p.tone === 'teal' ? `85, 229, 207` : `233, 239, 236`;
      const alphaCap = p.tone === 'bright' ? 1 : p.tone === 'teal' ? 0.8 : 0.55;
      const alpha = Math.min(baseAlpha, alphaCap);

      if (p.prevSX != null && p.prevSY != null && depthFactor > 0.55) {
        ctx.beginPath();
        ctx.moveTo(p.prevSX, p.prevSY);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = `rgba(${color}, ${alpha * 0.4})`;
        ctx.lineWidth = size * 0.5;
        ctx.stroke();
      }

      // Real radial falloff (not a flat-alpha disc) so each point reads as
      // starlight glowing outward, not a solid blob.
      if (p.tone !== 'dim') {
        const glowR = size * 7;
        const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowR);
        glow.addColorStop(0, `rgba(${color}, ${alpha * 0.55})`);
        glow.addColorStop(0.35, `rgba(${color}, ${alpha * 0.16})`);
        glow.addColorStop(1, `rgba(${color}, 0)`);
        ctx.beginPath();
        ctx.arc(sx, sy, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Bright points get a thin four-point sparkle flare — the classic
      // "glowing star" cue a flat dot never gives.
      if (p.tone === 'bright') {
        const flareLen = size * 6;
        ctx.strokeStyle = `rgba(${color}, ${alpha * 0.5})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(sx - flareLen, sy);
        ctx.lineTo(sx + flareLen, sy);
        ctx.moveTo(sx, sy - flareLen);
        ctx.lineTo(sx, sy + flareLen);
        ctx.stroke();
      }

      // Colored body, then a small hot-white core on top — this is the part
      // that actually reads as "lit" rather than a flat colored disc.
      ctx.beginPath();
      ctx.arc(sx, sy, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${color}, ${alpha})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(sx, sy, size * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(1, alpha * 1.3)})`;
      ctx.fill();

      p.prevSX = sx;
      p.prevSY = sy;
    }
  }, []);

  const canvasRef = useCanvasLoop(draw);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  );
}
