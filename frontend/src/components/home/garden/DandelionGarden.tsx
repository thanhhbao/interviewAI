'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { PALETTE, useSafeReducedMotion } from '@/lib/motion';
import { useInterview } from '@/context/InterviewContext';
import { FlowerField } from './FlowerField';
import { FoliageField } from './FoliageField';
import { SeedField } from './SeedField';
import { CameraRig } from './CameraRig';
import { Terrain } from './Terrain';
import { getFlowerField, getFoliageField, getSeedField, type GardenTier } from './gardenField';

const productLinks = [
  { label: 'Questions', href: '/setup' },
  { label: 'Interview Room', href: '/interview' },
  { label: 'Resume Match', href: '/resume-job' },
  { label: 'Reports', href: '/report' },
];

function useGardenTier(): GardenTier {
  const [tier, setTier] = useState<GardenTier>('desktop');
  useEffect(() => {
    const mqTablet = window.matchMedia('(max-width: 1023px)');
    const mqMobile = window.matchMedia('(max-width: 639px)');
    const update = () => setTier(mqMobile.matches ? 'mobile' : mqTablet.matches ? 'tablet' : 'desktop');
    update();
    mqTablet.addEventListener('change', update);
    mqMobile.addEventListener('change', update);
    return () => {
      mqTablet.removeEventListener('change', update);
      mqMobile.removeEventListener('change', update);
    };
  }, []);
  return tier;
}

// V2 "glide through a forest" scene — SCROLL MOVES A REAL PerspectiveCamera
// through a 3D field of instanced dandelions (see CameraRig.tsx /
// FlowerField.tsx), not a DOM/CSS parallax simulation. See
// docs/... (task report) for the full architecture writeup. Isolated at
// /dev/dandelion-garden-v2 — NOT wired into production Home yet.
export function DandelionGarden() {
  const prefersReducedMotion = useSafeReducedMotion();
  const router = useRouter();
  const { hasInterviewContext } = useInterview();
  const tier = useGardenTier();

  const sectionRef = useRef<HTMLElement>(null);
  const progressRef = useRef(0);
  const pointerRef = useRef({ x: 0, y: 0 });
  const [visible, setVisible] = useState(false);
  const [progressState, setProgressState] = useState(0); // for the DOM content fade only, throttled
  // Dev-only terrain visualization for validating the valley shape in
  // isolation — ?terrainDebug=1 on this route only, never a real prop
  // anyone would set from the outside. Not shipped/exposed anywhere else.
  const [terrainDebug, setTerrainDebug] = useState(false);
  useEffect(() => {
    setTerrainDebug(new URLSearchParams(window.location.search).get('terrainDebug') === '1');
  }, []);

  const handleStart = () => {
    router.push(hasInterviewContext ? '/setup' : '/resume-job');
  };

  // Scroll -> progress, rAF-batched so we never do this work more than
  // once per frame and never touch React state for it (CameraRig reads
  // the ref directly inside useFrame).
  useEffect(() => {
    let raf = 0;
    let scheduled = false;
    const compute = () => {
      scheduled = false;
      const section = sectionRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const scrollable = rect.height - window.innerHeight;
      const p = scrollable > 0 ? (0 - rect.top) / scrollable : 0;
      const clamped = Math.min(1, Math.max(0, p));
      progressRef.current = clamped;
      // Content-layer opacity/position only needs a coarse update, not
      // every scroll tick — still cheap, but rounding avoids re-rendering
      // on sub-percent scroll jitter.
      setProgressState(Math.round(clamped * 100) / 100);
    };
    const onScroll = () => {
      if (scheduled) return;
      scheduled = true;
      raf = requestAnimationFrame(compute);
    };
    compute();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  // Pointer -> normalized -1..1, also ref-only (read inside useFrame).
  useEffect(() => {
    if (prefersReducedMotion) return;
    const onMove = (e: PointerEvent) => {
      pointerRef.current = {
        x: (e.clientX / window.innerWidth) * 2 - 1,
        y: (e.clientY / window.innerHeight) * 2 - 1,
      };
    };
    window.addEventListener('pointermove', onMove);
    return () => window.removeEventListener('pointermove', onMove);
  }, [prefersReducedMotion]);

  // Offscreen pause: fully stops the WebGL render loop (frameloop="never")
  // rather than merely pausing individual animations — cheaper than the
  // footer's approach since nothing runs at all, not even idle scheduling.
  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: 0.01 });
    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  const active = !prefersReducedMotion && visible;
  const flowers = getFlowerField(tier);
  const seeds = getSeedField(tier);
  const foliage = getFoliageField(tier);

  // Staged reveal — the user should experience the forest first, then
  // arrive at the footer destination, not have UI floating over it from
  // the very first frame. Timings match the clearing itself (~85-100%,
  // see CLEARING_* constants in gardenField.ts) so content only appears
  // once the scene has actually opened up to hold it.
  const showBrand = prefersReducedMotion || progressState > 0.66;
  const showHeadline = prefersReducedMotion || progressState > 0.78;
  const showCta = prefersReducedMotion || progressState > 0.85;
  const showNav = prefersReducedMotion || progressState > 0.91;

  return (
    // Footer-scale pass: this was a 200vh standalone "chapter" — a footer
    // only needs enough scroll room to show restrained parallax as it
    // resolves, not a separate scene lasting several screen lengths.
    // Shrunk to 120vh; see CAMERA_START_Z/CAMERA_END_Z in gardenField.ts
    // for the matching camera-travel reduction.
    <section ref={sectionRef} className="relative w-full" style={{ height: '120vh', backgroundColor: '#020302' }}>
      <div className="sticky top-0 w-full overflow-hidden h-[min(74vh,760px)] min-h-[480px] sm:h-[min(78vh,800px)] sm:min-h-[560px]">
        {/* 3D scene */}
        <div className="absolute inset-0 pointer-events-none">
          <Canvas
            dpr={[1, tier === 'desktop' ? 1.75 : 1.25]}
            gl={{ antialias: true, alpha: false, toneMapping: THREE.NoToneMapping }}
            // Back to a moderate, reference-like FOV — the earlier macro
            // pass narrowed this for a telephoto/close-up feel, which is
            // wrong for a broad meadow glide (we want a wide field of
            // small flowers in view at once, not a few large ones).
            camera={{ fov: 54, near: 0.5, far: 100 }}
            frameloop={prefersReducedMotion ? 'demand' : visible ? 'always' : 'never'}
            onCreated={({ scene }) => {
              scene.background = new THREE.Color('#020302');
            }}
          >
            {/* Fog range shrunk to match the shorter corridor (was
                16-58 over a 78-unit corridor; corridor is now ~49
                units) so the horizon still dissolves within the
                populated field instead of fogging empty space. */}
            <fog attach="fog" args={['#020302', 9, 32]} />
            <CameraRig
              progressRef={progressRef}
              pointerRef={pointerRef}
              active={active}
              // Reduced-motion lands near the end of the traversal — a
              // static frame needs the footer content to actually be
              // visible/usable, not just a pretty dense composition.
              staticProgress={prefersReducedMotion ? 0.93 : undefined}
            />
            <Terrain debugWireframe={terrainDebug} />
            {!terrainDebug && (
              <>
                <FoliageField instances={foliage} active={active} />
                <FlowerField instances={flowers} active={active} />
                <SeedField instances={seeds} active={active} />
              </>
            )}
          </Canvas>
        </div>

        {/* Vignette + top/bottom fade — DOM overlay, cheap, keeps the
            canvas edges from looking like a pasted rectangle. */}
        <div
          className="absolute inset-0 pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              'radial-gradient(ellipse 70% 60% at 50% 55%, transparent 40%, rgba(0,0,0,0.55) 100%), linear-gradient(to bottom, rgba(0,0,0,0.4), transparent 20%, transparent 75%, rgba(0,0,0,0.6))',
          }}
        />

        {/* Content — staged in, not all at once, and anchored to the
            UPPER portion of the frame rather than dead-center: the
            reference composition keeps a large open/dark area up top for
            title content, with the dense flower carpet filling the lower
            frame underneath it (see the camera's downward pitch in
            CameraRig.tsx). The user experiences the field first (0-65%:
            no UI at all), then the brand mark, headline, CTA, and footer
            nav stage in as the traversal nears its end. */}
        {/* Fixed (not vh-based) top offset — the previous pt-[16vh] was
            sized for a full-screen 100vh panel; against the compact
            ~78vh/800px-capped footer panel now in use, a vh value would
            overflow the "dark negative space" budget instead of
            preserving it. */}
        <div className="absolute inset-0 flex flex-col items-center justify-start pt-10 sm:pt-14 px-6 text-center">
          <div className="pointer-events-none">
            <motion.p
              animate={{ opacity: showBrand ? 1 : 0, y: showBrand ? 0 : 12 }}
              transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
              className="text-[11px] font-mono tracking-[0.24em] uppercase mb-3"
              style={{ color: PALETTE.textSecondary }}
            >
              InterviewPal
            </motion.p>
            <motion.h2
              animate={{ opacity: showHeadline ? 1 : 0, y: showHeadline ? 0 : 14 }}
              transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
              className="text-2xl sm:text-3xl lg:text-4xl font-semibold tracking-tight"
              style={{ color: PALETTE.textPrimary }}
            >
              Practice. Reflect. Improve.
            </motion.h2>

            <motion.div
              animate={{ opacity: showCta ? 1 : 0, y: showCta ? 0 : 14 }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="mt-7 pointer-events-auto"
            >
              <button
                onClick={handleStart}
                className="group px-6 py-3 rounded-[10px] text-sm font-semibold transition-opacity hover:opacity-90 active:opacity-80 inline-flex items-center gap-2"
                style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
              >
                Start Interview
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </button>
            </motion.div>

            <motion.nav
              aria-label="Footer"
              animate={{ opacity: showNav ? 1 : 0 }}
              transition={{ duration: 0.6 }}
              className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs pointer-events-auto"
              style={{ color: PALETTE.textSecondary }}
            >
              {productLinks.map((l) => (
                <Link key={l.label} href={l.href} className="transition-colors hover:text-white">
                  {l.label}
                </Link>
              ))}
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accent }} />
                All systems operational
              </span>
              <span className="opacity-60">© 2026 InterviewPal</span>
            </motion.nav>
          </div>

          {!prefersReducedMotion && (
            <motion.p
              animate={{ opacity: progressState < 0.15 ? 1 : 0 }}
              transition={{ duration: 0.6 }}
              className="absolute bottom-10 text-[11px] font-mono tracking-[0.2em] uppercase pointer-events-none"
              style={{ color: PALETTE.textSecondary }}
            >
              Scroll to enter the field
            </motion.p>
          )}
        </div>
      </div>
    </section>
  );
}
