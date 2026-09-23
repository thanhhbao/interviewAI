'use client';

import React, { useCallback, useEffect, useRef } from 'react';
import { useCanvasLoop } from './useCanvasLoop';

// Native clone of the Framer "Dot Grid Background" component's actual
// mechanism (not a CSS dotted pattern): an evenly spaced canvas grid where
// dots inside the cursor's interaction radius physically leave their grid
// position and orbit the cursor at a per-dot phase/speed, easing smoothly
// back to their ordered position once the cursor moves away. Unaffected
// dots never move at all.
const SPACING = 42;
const RADIUS_PX = 1.15;
const INTERACTION_RADIUS = 190;
const MAX_ORBIT = 15; // px the dot displaces off its grid position at peak influence
const EASE = 0.14; // per-frame lerp factor toward the current target position

interface Dot {
  gx: number;
  gy: number;
  x: number;
  y: number;
  phase: number;
  phaseSpeed: number;
  influence: number; // smoothed 0-1, drives color/size too
}

export function DotGrid({ className }: { className?: string }) {
  const dotsRef = useRef<Dot[]>([]);
  const pointer = useRef({ x: -9999, y: -9999 });
  const lastT = useRef(0);

  const buildGrid = useCallback((w: number, h: number) => {
    const cols = Math.ceil(w / SPACING) + 1;
    const rows = Math.ceil(h / SPACING) + 1;
    const dots: Dot[] = [];
    for (let iy = 0; iy < rows; iy++) {
      for (let ix = 0; ix < cols; ix++) {
        const gx = ix * SPACING;
        const gy = iy * SPACING;
        dots.push({
          gx,
          gy,
          x: gx,
          y: gy,
          phase: Math.random() * Math.PI * 2,
          phaseSpeed: 0.6 + Math.random() * 0.8,
          influence: 0,
        });
      }
    }
    dotsRef.current = dots;
  }, []);

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    const dt = Math.min(0.05, t - lastT.current || 0.016);
    lastT.current = t;

    ctx.clearRect(0, 0, w, h);
    const px = pointer.current.x;
    const py = pointer.current.y;

    for (const d of dotsRef.current) {
      const dx = px - d.gx;
      const dy = py - d.gy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const targetInfluence = dist < INTERACTION_RADIUS ? 1 - dist / INTERACTION_RADIUS : 0;
      // Ease influence itself so engagement/release both feel smooth.
      d.influence += (targetInfluence - d.influence) * 0.12;

      let targetX = d.gx;
      let targetY = d.gy;
      if (d.influence > 0.002) {
        d.phase += dt * d.phaseSpeed * (0.4 + d.influence);
        const orbitR = MAX_ORBIT * d.influence;
        const pull = d.influence * 0.25; // slight bias toward the cursor itself
        targetX = d.gx + Math.cos(d.phase) * orbitR + dx * pull;
        targetY = d.gy + Math.sin(d.phase) * orbitR + dy * pull;
      }

      d.x += (targetX - d.x) * EASE;
      d.y += (targetY - d.y) * EASE;

      const shimmer = 0.6 + 0.4 * Math.sin(t * 0.4 + d.gx * 0.02 + d.gy * 0.02);
      const idleAlpha = 0.1 + shimmer * 0.08;
      const alpha = idleAlpha + d.influence * (0.7 - idleAlpha);
      const radius = RADIUS_PX + d.influence * 1.1;

      // Idle: rgba(133,142,138, a). Active blends toward #55E5CF, and the
      // closest dots (highest influence) push further toward #78FFE7.
      const r = Math.round(133 + (120 - 133) * d.influence);
      const g = Math.round(142 + (255 - 142) * d.influence);
      const b = Math.round(138 + (231 - 138) * d.influence);

      ctx.beginPath();
      ctx.arc(d.x, d.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fill();

      if (d.influence > 0.4) {
        // Soft bloom on the innermost, most-engaged dots only.
        ctx.beginPath();
        ctx.arc(d.x, d.y, radius * 2.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(120, 255, 231, ${(d.influence - 0.4) * 0.12})`;
        ctx.fill();
      }
    }
  }, []);

  const canvasRef = useCanvasLoop(draw, buildGrid);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    function onMove(e: MouseEvent) {
      const rect = el!.getBoundingClientRect();
      pointer.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
    function onLeave() {
      pointer.current = { x: -9999, y: -9999 };
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseleave', onLeave);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseleave', onLeave);
    };
  }, [canvasRef]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  );
}
