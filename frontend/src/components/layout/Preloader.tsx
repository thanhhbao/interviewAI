'use client';

// Inspired by the "Counter Preloader" Framer component
// (framer.com/marketplace/components/counter-preloader) — no live preview
// or source was reachable to port verbatim, so this reproduces its
// documented shape (a large counting number, a thin top "laser" progress
// line, a centered brand signature, session-memory so it only shows once
// per browser session, one of its "10 exit reveal styles" — a staggered
// panel wipe) rebuilt with this app's own tokens and motion language
// rather than guessed pixel-for-pixel values.
//
// Mounted once in the root layout, so it only exists for the lifetime of a
// real page load (Next.js layouts persist across client-side navigation —
// this never remounts when moving between /resume-job, /setup, etc).
import React, { useEffect, useRef, useState } from 'react';
import { motion, type Variants } from 'framer-motion';
import { MessageSquare } from 'lucide-react';
import { PALETTE, useSafeReducedMotion } from '@/lib/motion';

const COUNT_DURATION_MS = 1700;
const HOLD_MS = 200;
const SESSION_KEY = 'interviewpal-preloaded';

// Exit reveal: the counter/brand content fades out fast, then a row of
// panels — still solid, still covering the real page — peel away upward
// one after another, each slightly later than the last, so the reveal
// reads as a single wipe sweeping left-to-right rather than a plain fade.
const PANEL_COUNT = 7;
const PANEL_DELAY_BEFORE_S = 0.15; // lets the content finish fading first
const PANEL_STAGGER_S = 0.05;
const PANEL_DURATION_S = 0.65;
const PANEL_EASE = [0.76, 0, 0.24, 1] as const;
const EXIT_MS = Math.round(
  (PANEL_DELAY_BEFORE_S + (PANEL_COUNT - 1) * PANEL_STAGGER_S + PANEL_DURATION_S) * 1000
) + 100;

function easeOutExpo(t: number) {
  return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

const contentVariants: Variants = {
  idle: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -16, transition: { duration: 0.3, ease: 'easeIn' } },
};

const panelsContainerVariants: Variants = {
  idle: {},
  exit: {
    transition: { staggerChildren: PANEL_STAGGER_S, delayChildren: PANEL_DELAY_BEFORE_S },
  },
};

const panelVariants: Variants = {
  idle: { y: '0%' },
  exit: { y: '-100%', transition: { duration: PANEL_DURATION_S, ease: PANEL_EASE } },
};

export function Preloader() {
  const prefersReducedMotion = useSafeReducedMotion();
  // Both the server render and the client's first render assume "show it"
  // — sessionStorage isn't readable until after hydration, so starting
  // from the same value on both sides avoids a hydration mismatch. A
  // repeat visit within the same session corrects this a frame later
  // (see below), which reads as an instant skip, not a real flash.
  const [shouldShow, setShouldShow] = useState(true);
  const [count, setCount] = useState(0);
  const [exiting, setExiting] = useState(false);
  const skipAnimationRef = useRef(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (sessionStorage.getItem(SESSION_KEY)) {
      skipAnimationRef.current = true;
      setShouldShow(false);
      return;
    }

    let raf = 0;

    const finish = () => {
      setExiting(true);
      setTimeout(() => {
        sessionStorage.setItem(SESSION_KEY, '1');
        setShouldShow(false);
      }, EXIT_MS);
    };

    if (prefersReducedMotion) {
      setCount(100);
      const t = setTimeout(finish, HOLD_MS);
      return () => clearTimeout(t);
    }

    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / COUNT_DURATION_MS);
      setCount(Math.round(easeOutExpo(progress) * 100));
      if (progress < 1) {
        raf = requestAnimationFrame(tick);
      } else {
        setTimeout(finish, HOLD_MS);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [prefersReducedMotion]);

  if (!shouldShow) return null;

  const exitState = skipAnimationRef.current ? undefined : exiting ? 'exit' : 'idle';

  return (
    <div className="fixed inset-0 z-[999] select-none" aria-hidden="true">
      {/* Panel layer — the actual solid background. Idle: seven abutting
          panels of the same color read as one flat screen. Exit: they peel
          upward in a staggered wave, progressively uncovering the real
          page mounted underneath this fixed overlay the whole time. */}
      <motion.div
        className="absolute inset-0 flex"
        variants={panelsContainerVariants}
        initial="idle"
        animate={skipAnimationRef.current ? 'idle' : exiting ? 'exit' : 'idle'}
      >
        {Array.from({ length: PANEL_COUNT }).map((_, i) => (
          <motion.div
            key={i}
            variants={panelVariants}
            className="h-full"
            style={{ width: `${100 / PANEL_COUNT}%`, backgroundColor: PALETTE.bg }}
          />
        ))}
      </motion.div>

      {/* Content layer — fades out quickly before the panels start moving,
          so nothing looks like it's being dragged along with the wipe. */}
      <motion.div
        className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
        variants={contentVariants}
        initial="idle"
        animate={exitState}
      >
        {/* Top laser progress line */}
        <div
          className="absolute top-0 left-0 right-0 h-[2px] overflow-hidden"
          style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}
        >
          <div
            className="h-full transition-[width] duration-100 ease-linear"
            style={{ backgroundColor: PALETTE.accentBright, width: `${count}%` }}
          />
        </div>

        {/* Ambient glow — one static bloom, not the interactive dot field;
            this is on screen for under two seconds, not worth the cursor
            listener. */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(255,255,255,0.06), transparent 70%)' }}
        />

        {/* Brand signature */}
        <div className="relative flex items-center gap-2 mb-8">
          <div
            className="w-6 h-6 rounded-md flex items-center justify-center"
            style={{ backgroundColor: PALETTE.accentBright, color: '#07080a' }}
          >
            <MessageSquare className="w-3.5 h-3.5" />
          </div>
          <span className="text-[11px] font-mono tracking-[0.2em] uppercase" style={{ color: PALETTE.textSecondary }}>
            InterviewPal
          </span>
        </div>

        {/* Monumental counter */}
        <div className="relative flex items-start" style={{ color: PALETTE.textPrimary }}>
          <span
            className="font-semibold tabular-nums leading-none"
            style={{ fontSize: 'clamp(4.5rem, 15vw, 10rem)' }}
          >
            {count}
          </span>
          <span className="mt-3 sm:mt-6 text-2xl sm:text-4xl font-medium" style={{ color: PALETTE.textSecondary }}>
            %
          </span>
        </div>

        <p
          className="relative mt-5 text-[11px] font-mono tracking-[0.2em] uppercase"
          style={{ color: PALETTE.textSecondary }}
        >
          Preparing your interview room
        </p>
      </motion.div>
    </div>
  );
}
