'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';
import Link from 'next/link';
import { Twitter, Linkedin, Github } from 'lucide-react';
import { PALETTE, useSafeReducedMotion } from '@/lib/motion';
import { DandelionSVG } from './DandelionSVG';
import { DandelionSeed } from './DandelionSeed';
import { FLOWER_FIELD, SEED_FIELD, type FlowerVisibility } from './dandelionField';

// Progressive reveal by breakpoint — see dandelionField.ts for why each
// flower got the tier it did. 'always' covers mobile's "2-3 primary
// flowers" requirement on its own (3 of the 6 configs are 'always').
const VISIBILITY_CLASS: Record<FlowerVisibility, string> = {
  always: '',
  'tablet-up': 'hidden sm:block',
  'desktop-up': 'hidden lg:block',
};

const FOOTER_COLS = [
  {
    title: 'Product',
    links: [
      { label: 'Interview Room',  href: '/interview'  },
      { label: 'Resume & Job',    href: '/resume-job' },
      { label: 'Reports',         href: '/report'     },
      { label: 'History',         href: '/history'    },
      { label: 'Settings',        href: '/settings'   },
    ],
  },
  {
    title: 'Practice',
    links: [
      { label: 'Quick Warm-up',       href: '/setup' },
      { label: 'Behavioral',          href: '/setup' },
      { label: 'Technical Design',    href: '/setup' },
      { label: 'Coding Interview',    href: '/setup' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { label: 'Help Center',     href: '#' },
      { label: 'Interview Tips',  href: '#' },
      { label: 'Resume Guide',    href: '#' },
      { label: 'FAQ',             href: '#' },
    ],
  },
  {
    title: 'Legal',
    links: [
      { label: 'Privacy Policy',   href: '#' },
      { label: 'Terms of Service', href: '#' },
      { label: 'Cookie Policy',    href: '#' },
    ],
  },
];

// Cinematic footer scene — layered white dandelions in 2.5D (CSS
// transforms + SVG, no WebGL) behind the real footer content. Adapted from
// the *idea* of Framer's "Ultimate Garden Scroll" (layered botanical
// depth, gentle sway, drifting particles) — not its scroll-driven garden
// experience, its color palette, or its assets. This is the production
// Home footer (src/app/page.tsx); it also renders standalone at
// /dev/dandelion-footer for isolated visual regression checks.
export function DandelionFooter() {
  const prefersReducedMotion = useSafeReducedMotion();
  const sceneRef = useRef<HTMLDivElement>(null);

  // `hasEntered` latches true the first time the footer is seen and stays
  // true forever — it drives the one-shot entrance animation, which should
  // never replay. `isVisible` is live and flips back to false once the
  // footer scrolls back out of view — it drives whether the continuous
  // sway/seed-drift loops are actually running (see `active` below), the
  // same IntersectionObserver-gated pause pattern KineticMetalFlow.tsx
  // uses for its canvas loop, adapted for Framer Motion's declarative
  // `animate` prop: passing `undefined` holds the last rendered transform
  // in place rather than resetting it, so resuming blends back into the
  // loop instead of snapping.
  const hasEnteredOnce = useInView(sceneRef, { once: true, margin: '-80px' });
  const isVisible = useInView(sceneRef, { margin: '-80px' });
  const [hasEntered, setHasEntered] = useState(false);
  useEffect(() => {
    if (hasEnteredOnce) setHasEntered(true);
  }, [hasEnteredOnce]);

  const revealed = prefersReducedMotion || hasEntered;
  const active = !prefersReducedMotion && isVisible;

  return (
    <footer
      ref={sceneRef}
      className="relative w-full overflow-hidden h-[480px] sm:h-[560px] lg:h-[660px]"
      style={{
        // Starts at exactly #050807 — the same colour FinalCtaSection fades
        // to at its bottom — so there is no colour jump at the boundary.
        // The gradient then eases to the page bg and deepens at the very
        // bottom for the scene's own atmosphere.
        background: `linear-gradient(to bottom, #050807 0%, #050807 6%, ${PALETTE.bg} 28%, #030404 100%)`,
      }}
    >
      {/* Top vignette removed — FinalCtaSection's own bottom fade now handles
          the transition, so a second dark overlay here would double-darken
          the seam instead of smoothing it. */}

      {/* Botanical scene — decorative only, sits behind the content layer.
          Background depth (this wrapper's own opacity) appears first, then
          each flower layer translates up 15-30px with a small stagger. */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
        initial={prefersReducedMotion ? false : { opacity: 0 }}
        animate={{ opacity: revealed ? 1 : 0 }}
        transition={{ duration: 1.1, ease: 'easeOut' }}
      >
        {FLOWER_FIELD.map((f) => {
          // Anchor to whichever horizontal edge the flower is closer to,
          // rather than always measuring from the left: a wide foreground
          // flower positioned via `left: 9%` still visually centers itself
          // on a narrow viewport once its own rendered width is added on.
          // Anchoring foreground/midground flowers to their nearest edge
          // keeps their heads clear of the centered text column at every
          // width instead of just on desktop.
          const anchorRight = f.leftPercent > 50;
          const edgePercent = anchorRight ? 100 - f.leftPercent : f.leftPercent;

          return (
            <motion.div
              key={f.id}
              className={`absolute ${VISIBILITY_CLASS[f.visibility]}`}
              style={{
                [anchorRight ? 'right' : 'left']: `${edgePercent}%`,
                bottom: f.bottomOffsetPx,
              }}
              initial={prefersReducedMotion ? false : { opacity: 0, y: 24 }}
              animate={{ opacity: revealed ? 1 : 0, y: revealed ? 0 : 24 }}
              transition={{
                duration: 0.9,
                delay: f.layer === 'background' ? 0 : f.layer === 'midground' ? 0.12 : 0.22,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              {/* Static per-flower head tilt — its own wrapper, separate
                  from the responsive scale/translate below, so the two
                  transforms compose instead of one Tailwind-generated
                  transform overwriting the other on the same element. */}
              <div style={{ transform: `rotate(${f.headTiltDeg}deg)`, transformOrigin: '50% 100%' }}>
                {/* Smaller and pushed further down on mobile — the footer
                    is only ~440px tall there, so a foreground flower sized
                    for the 700px desktop footer reaches straight into the
                    nav row otherwise. An extra tablet-only tier keeps the
                    midground pair clear of the centered text column at
                    834px specifically (see dandelionField.ts). Scales/
                    translates from the stem base so the anchor point never
                    moves. */}
                <div
                  className="scale-[0.4] sm:scale-[0.55] lg:scale-100 translate-y-14 sm:translate-y-0"
                  style={{ transformOrigin: '50% 100%' }}
                >
                  {/* Sway lives on its own inner wrapper so it composes
                      with the entrance transform instead of fighting it
                      (same reason the nav's glow rotor was split into an
                      outer static + inner animated layer — Framer
                      Motion's inline transform overwrites rather than
                      merges with a sibling animation on one node).
                      `animate={active ? ... : undefined}` is the offscreen
                      pause: dropping the animate target holds the last
                      rendered transform instead of resetting it, so
                      re-entering the viewport blends back into the loop. */}
                  <motion.div
                    animate={
                      active
                        ? {
                            rotate: [-f.swayRotateDeg, f.swayRotateDeg, -f.swayRotateDeg],
                            x: [-f.swayDriftPx, f.swayDriftPx, -f.swayDriftPx],
                          }
                        : undefined
                    }
                    transition={{ duration: f.swayDurationS, delay: f.swayDelayS, repeat: Infinity, ease: 'easeInOut' }}
                    style={{ transformOrigin: '50% 100%' }}
                  >
                    <DandelionSVG
                      heightPx={f.heightPx}
                      filamentCount={f.filamentCount}
                      coreOpacity={f.coreOpacity}
                      glow
                      glowOpacity={f.glowOpacityValue}
                      jitterSeed={f.jitterSeed}
                      className={f.layer === 'background' ? 'blur-[0.6px]' : undefined}
                    />
                  </motion.div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Seed drift — mounted once the entrance has fired and left
          mounted (never remounted just because the footer scrolls out of
          view, which would reset every seed's phase); each seed's own
          `active` prop pauses/resumes its loop instead. Never mounted at
          all under reduced motion. */}
      {!prefersReducedMotion && hasEntered && (
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
          {SEED_FIELD.map((s) => (
            <div key={s.id} className={s.alwaysVisible ? '' : 'hidden lg:block'}>
              <DandelionSeed config={s} active={active} />
            </div>
          ))}
        </div>
      )}

      {/* Content layer — always on top, always readable. */}
      <div className="relative z-10 h-full flex flex-col justify-end px-6">

        {/* ── Footer links section — pinned to bottom ── */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: revealed ? 1 : 0 }}
          transition={{ duration: 0.6, delay: 0.62 }}
          className="w-full max-w-5xl mx-auto pb-8 sm:pb-10 lg:pb-12"
        >
          {/* Divider */}
          <div className="mb-8" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }} />

          {/* 4-column grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            {FOOTER_COLS.map((col) => (
              <div key={col.title} className="space-y-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.14em]" style={{ color: PALETTE.textSecondary }}>
                  {col.title}
                </p>
                <ul className="space-y-2">
                  {col.links.map((l) => (
                    <li key={l.label}>
                      <Link
                        href={l.href}
                        className="text-[12px] transition-colors hover:text-white"
                        style={{ color: 'rgba(133,142,138,0.75)' }}
                      >
                        {l.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Bottom bar */}
          <div
            className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-5 text-[11px]"
            style={{ borderTop: '1px solid rgba(255,255,255,0.05)', color: 'rgba(133,142,138,0.6)' }}
          >
            <div className="flex items-center gap-4">
              <span>© 2026 InterviewPal. All rights reserved.</span>
              <span className="hidden sm:flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accent }} />
                All systems operational
              </span>
            </div>
            <div className="flex items-center gap-3">
              <a href="#" aria-label="Twitter / X" className="transition-colors hover:text-white">
                <Twitter className="w-3.5 h-3.5" />
              </a>
              <a href="#" aria-label="LinkedIn" className="transition-colors hover:text-white">
                <Linkedin className="w-3.5 h-3.5" />
              </a>
              <a href="#" aria-label="GitHub" className="transition-colors hover:text-white">
                <Github className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}
