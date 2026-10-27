'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useCanvasLoop } from './useCanvasLoop';

// Cinematic perspective-projected starfield.
// Stars live in 3-D world space (x, y, z). Each frame z decreases so stars
// drift toward the viewer, projecting outward from the canvas centre as they
// approach. Near stars are larger and brighter; almost-invisible trails mark
// their motion. A subtle cursor-parallax nudges the origin so the field
// responds to mouse movement without any layout shift.
//
// IntersectionObserver pause, devicePixelRatio cap, and prefers-reduced-motion
// static-frame are all handled by the shared useCanvasLoop hook.

const COUNT = 200;
const FOCAL = 420;        // perspective focal length (px at full canvas scale)
const SPEED = 48;         // z-units per second toward viewer
const Z_FAR = 680;        // spawn depth (far plane)
const Z_NEAR = 1;         // cull/reset threshold (near plane)
const PARALLAX = 0.018;   // fraction of half-canvas to shift per unit cursor

interface Star {
  x: number;        // world-space x (centred at 0)
  y: number;        // world-space y (centred at 0)
  z: number;        // depth: Z_FAR = far, Z_NEAR = near viewer
  prevSx: number;
  prevSy: number;
  hasPrev: boolean;
}

function makeStar(w: number, h: number, scatter = true): Star {
  // Spread in world-space so stars fill the screen at both Z_FAR and Z_NEAR.
  // At Z_FAR the half-spread that just reaches the screen edge is
  // (w/2) * (Z_FAR / FOCAL). We use 1.15× that so fresh spawns aren't
  // all bunched exactly at the screen boundary.
  const sx = (w / 2) * (Z_FAR / FOCAL) * 1.15;
  const sy = (h / 2) * (Z_FAR / FOCAL) * 1.15;
  return {
    x: (Math.random() - 0.5) * 2 * sx,
    y: (Math.random() - 0.5) * 2 * sy,
    z: scatter ? Math.random() * Z_FAR + Z_NEAR : Z_FAR,
    prevSx: 0,
    prevSy: 0,
    hasPrev: false,
  };
}

export function StarsDataField({ className }: { className?: string }) {
  const starsRef  = useRef<Star[]>([]);
  const sizeRef   = useRef({ w: 0, h: 0 });
  const mouseRef  = useRef({ x: 0, y: 0 }); // normalised [-1, 1]
  const lastTRef  = useRef(0);

  // Cursor parallax — passive listener, minimal overhead.
  useEffect(() => {
    function onMove(e: MouseEvent) {
      mouseRef.current = {
        x: (e.clientX / window.innerWidth  - 0.5) * 2,
        y: (e.clientY / window.innerHeight - 0.5) * 2,
      };
    }
    window.addEventListener('mousemove', onMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  const onResize = useCallback((w: number, h: number) => {
    sizeRef.current = { w, h };
    // Reinitialise all stars so their world-space spread matches new dimensions.
    starsRef.current = Array.from({ length: COUNT }, () => makeStar(w, h, true));
  }, []);

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    const dt = Math.min(t - lastTRef.current, 0.05); // cap at 50 ms (tab-hidden catch-up)
    lastTRef.current = t;

    // Lazy init on the very first frame (before onResize has fired).
    if (starsRef.current.length === 0) {
      starsRef.current = Array.from({ length: COUNT }, () => makeStar(w, h, true));
    }

    const cx = w / 2;
    const cy = h / 2;

    // Parallax nudge (screen pixels).
    const px = mouseRef.current.x * PARALLAX * w;
    const py = mouseRef.current.y * PARALLAX * h;

    // Dark background — must fill, not clearRect, so trails compound correctly.
    ctx.fillStyle = '#07080a';
    ctx.fillRect(0, 0, w, h);

    const stars  = starsRef.current;
    const isStatic = dt === 0; // reduced-motion path: t=0, dt=0

    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];

      // Advance depth toward viewer.
      if (!isStatic) s.z -= SPEED * dt;

      // Project onto screen.
      const inv = FOCAL / s.z;
      const sx  = cx + s.x * inv + px;
      const sy  = cy + s.y * inv + py;

      // Depth factor: 0 = at Z_FAR (far), 1 = at Z_NEAR (near viewer).
      const depth = 1 - (s.z - Z_NEAR) / (Z_FAR - Z_NEAR);

      // Cull past near plane or far off-screen, then respawn at far plane.
      const offscreen = sx < -w * 0.08 || sx > w * 1.08 || sy < -h * 0.08 || sy > h * 1.08;
      if (s.z <= Z_NEAR || offscreen) {
        stars[i] = makeStar(w, h, false);
        continue;
      }

      const radius = Math.max(0.15, depth * 2.8);
      const alpha  = 0.07 + depth * 0.82;

      // Almost-invisible trail for near stars only (depth > 0.45).
      if (s.hasPrev && depth > 0.45) {
        const trailAlpha = (depth - 0.45) * 0.18; // max ≈ 0.10
        ctx.beginPath();
        ctx.moveTo(s.prevSx, s.prevSy);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = `rgba(255,255,255,${trailAlpha})`;
        ctx.lineWidth   = radius * 0.7;
        ctx.stroke();
      }

      // Soft glow corona for nearby stars (depth > 0.6) — creates the
      // "emerging from the dark" feel without bright halos everywhere.
      if (depth > 0.6) {
        const glowR = radius * 5;
        const glow  = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowR);
        const ga    = (depth - 0.6) * 0.22;
        glow.addColorStop(0, `rgba(255,255,255,${ga})`);
        glow.addColorStop(1, `rgba(255,255,255,0)`);
        ctx.beginPath();
        ctx.arc(sx, sy, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Thin four-point sparkle flare for the very nearest stars only,
      // drawn before the core dot so the dot sits on top.
      if (depth > 0.82) {
        const flen = radius * 5.5;
        ctx.strokeStyle = `rgba(255,255,255,${(depth - 0.82) * 0.6})`;
        ctx.lineWidth   = 0.5;
        ctx.beginPath();
        ctx.moveTo(sx - flen, sy);
        ctx.lineTo(sx + flen, sy);
        ctx.moveTo(sx, sy - flen);
        ctx.lineTo(sx, sy + flen);
        ctx.stroke();
      }

      // Star core — grey at depth, white at near.
      const grey  = Math.round(200 + depth * 55); // 200 → 255
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${grey},${grey},${grey},${alpha})`;
      ctx.fill();

      // Hot-white centre pinpoint.
      if (depth > 0.3) {
        ctx.beginPath();
        ctx.arc(sx, sy, radius * 0.45, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.min(1, alpha * 1.25)})`;
        ctx.fill();
      }

      s.prevSx = sx;
      s.prevSy = sy;
      s.hasPrev = true;
    }
  }, []);

  const canvasRef = useCanvasLoop(draw, onResize);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  );
}
