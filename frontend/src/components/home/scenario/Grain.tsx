'use client';

import React from 'react';

// Reusable film-grain overlay. The noise texture is a single static SVG
// feTurbulence tile encoded once as a data URI and repeated via CSS
// background — there is no per-frame canvas/JS noise generation, so this
// costs nothing at runtime no matter how many cards use it.
const GRAIN_SVG =
  "<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'>" +
  "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/>" +
  "<feColorMatrix type='matrix' values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.6 0'/></filter>" +
  "<rect width='100%' height='100%' filter='url(#n)'/></svg>";
const GRAIN_URL = `url("data:image/svg+xml,${encodeURIComponent(GRAIN_SVG)}")`;

export function Grain({ opacity = 0.1, className = '' }: { opacity?: number; className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 ${className}`}
      style={{
        backgroundImage: GRAIN_URL,
        backgroundSize: '140px 140px',
        opacity,
        mixBlendMode: 'overlay',
      }}
    />
  );
}
