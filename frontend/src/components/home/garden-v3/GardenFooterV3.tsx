'use client';

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { PALETTE } from '@/lib/motion';
import { MeadowLayer } from './MeadowLayer';
import { getMeadowTextures, getLayerAspect } from './meadowTextures';

const productLinks = ['Questions', 'Interview Room', 'Resume Match', 'Reports'];

// `getMeadowTextures()` touches `document` (canvas rasterization), so it
// must only ever run on the client. R3F's <Canvas> doesn't mount its
// scene-graph children during SSR (only the wrapping <div>), so calling
// it from a component rendered INSIDE <Canvas> — not from GardenFooterV3
// itself — is what keeps this SSR-safe (same pattern as the V2 garden's
// FlowerField.tsx).
function MeadowLayers() {
  const textures = useMemo(() => getMeadowTextures(), []);
  return (
    <>
      <MeadowLayer
        texture={textures.far}
        aspect={getLayerAspect('far')}
        width={40}
        z={-16}
        y={1.6}
        curveHalfWidth={17}
        curveRise={3.4}
        curveExponent={1.8}
        renderOrder={0}
      />
      <MeadowLayer
        texture={textures.mid}
        aspect={getLayerAspect('mid')}
        width={34}
        z={-7}
        y={0.7}
        curveHalfWidth={15}
        curveRise={3.0}
        curveExponent={1.9}
        renderOrder={1}
      />
      <MeadowLayer
        texture={textures.near}
        aspect={getLayerAspect('near')}
        width={28}
        z={1}
        y={0}
        curveHalfWidth={13}
        curveRise={2.4}
        curveExponent={2.0}
        renderOrder={2}
      />
      <MeadowLayer
        texture={textures.foreground}
        aspect={getLayerAspect('foreground')}
        width={24}
        z={6.5}
        y={-0.3}
        curveHalfWidth={11}
        curveRise={1.8}
        curveExponent={2.1}
        renderOrder={3}
      />
    </>
  );
}

// Phase 1 (this task): a completely STATIC composition — no useFrame, no
// scroll listener, no pointer listener, no wind, no camera travel. The
// Canvas renders exactly once (`frameloop="demand"`) and never again. If
// this still image doesn't read as a dense monochrome meadow, animating
// it later won't fix that — see the task's own framing for why static
// comes first.
export function GardenFooterV3() {
  return (
    <section className="relative w-full" style={{ backgroundColor: '#020302' }}>
      <div className="relative w-full overflow-hidden h-[min(74vh,760px)] min-h-[480px] sm:h-[min(78vh,800px)] sm:min-h-[560px]">
        <div className="absolute inset-0">
          <Canvas
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: false, toneMapping: THREE.NoToneMapping }}
            camera={{ fov: 50, near: 0.5, far: 60, position: [0, 2.3, 9.5] }}
            frameloop="demand"
            onCreated={({ scene, camera }) => {
              scene.background = new THREE.Color('#020302');
              camera.lookAt(0, 0.5, -6);
            }}
          >
            <fog attach="fog" args={['#020302', 10, 34]} />
            <MeadowLayers />
          </Canvas>
        </div>

        {/* Vignette — same purpose as V2's: keeps the canvas edges from
            looking like a pasted rectangle. Static, no progress-driven
            change (there is no progress in this phase). */}
        <div
          className="absolute inset-0 pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              'radial-gradient(ellipse 70% 60% at 50% 55%, transparent 40%, rgba(0,0,0,0.5) 100%), linear-gradient(to bottom, rgba(0,0,0,0.35), transparent 22%, transparent 70%, rgba(0,0,0,0.55))',
          }}
        />

        {/* Placeholder footer content — evaluating the garden composition
            here, not UI styling, so this is deliberately plain: no
            stagger, no scroll-driven reveal (out of scope for this
            static-only task). */}
        <div className="absolute inset-0 flex flex-col items-center justify-start pt-10 sm:pt-14 px-6 text-center pointer-events-none">
          <p
            className="text-[11px] font-mono tracking-[0.24em] uppercase mb-3"
            style={{ color: PALETTE.textSecondary }}
          >
            InterviewAI
          </p>
          <h2
            className="text-2xl sm:text-3xl lg:text-4xl font-semibold tracking-tight"
            style={{ color: PALETTE.textPrimary }}
          >
            Practice. Reflect. Improve.
          </h2>
          <div className="mt-7 pointer-events-auto">
            <button
              className="px-6 py-3 rounded-[10px] text-sm font-semibold inline-flex items-center gap-2"
              style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
            >
              Start Interview
            </button>
          </div>
          <nav
            aria-label="Footer"
            className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs pointer-events-auto"
            style={{ color: PALETTE.textSecondary }}
          >
            {productLinks.map((l) => (
              <span key={l} className="opacity-90">
                {l}
              </span>
            ))}
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accent }} />
              All systems operational
            </span>
            <span className="opacity-60">© 2026 InterviewAI</span>
          </nav>
        </div>
      </div>
    </section>
  );
}
