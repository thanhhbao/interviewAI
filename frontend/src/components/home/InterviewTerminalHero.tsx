'use client';

import React, { useRef, useCallback } from 'react';
import { motion, useSpring, useMotionValue, useTransform } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { ChevronDown } from 'lucide-react';
import { useSafeReducedMotion, PALETTE } from '@/lib/motion';
import { useInterview } from '@/context/InterviewContext';
import { useAuth } from '@/context/AuthContext';
import { AuthLink } from '@/components/auth/AuthLink';
import { TerminalScreen } from '@/components/home/TerminalScreen';

const ACCENT = PALETTE.accent;       // '#E7E8E9'
const TEXT_PRIMARY = '#F2F0EA';
const TEXT_SECONDARY = '#97958F';
const BG = '#07080a';

export const InterviewTerminalHero: React.FC = () => {
  const router = useRouter();
  const prefersReducedMotion = useSafeReducedMotion();
  const { hasInterviewContext } = useInterview();
  const { user, openLoginModal } = useAuth();
  const heroRef = useRef<HTMLDivElement>(null);

  // Raw mouse position mapped to -1..1 range relative to hero center
  const rawX = useMotionValue(0);
  const rawY = useMotionValue(0);

  // Spring-smoothed values — stiffness/damping chosen for premium, restrained feel
  const springX = useSpring(rawX, { stiffness: 60, damping: 22 });
  const springY = useSpring(rawY, { stiffness: 60, damping: 22 });

  // Map to rotation (max ±4deg X, ±3deg Y) and translate (max ±6px)
  const rotateY = useTransform(springX, [-1, 1], [-4, 4]);
  const rotateX = useTransform(springY, [-1, 1], [3, -3]);
  const translateX = useTransform(springX, [-1, 1], [-6, 6]);
  const translateY = useTransform(springY, [-1, 1], [-4, 4]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (prefersReducedMotion || !heroRef.current) return;
      const rect = heroRef.current.getBoundingClientRect();
      rawX.set(((e.clientX - rect.left) / rect.width - 0.5) * 2);
      rawY.set(((e.clientY - rect.top) / rect.height - 0.5) * 2);
    },
    [prefersReducedMotion, rawX, rawY]
  );

  const handleMouseLeave = useCallback(() => {
    rawX.set(0);
    rawY.set(0);
  }, [rawX, rawY]);

  const handleStartNow = () => {
    if (!user) { openLoginModal(); return; }
    router.push(hasInterviewContext ? '/setup' : '/resume-job');
  };

  const handleScrollToContent = () => {
    document.getElementById('explore')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section
      ref={heroRef}
      className="relative w-full min-h-screen flex flex-col overflow-hidden select-none"
      style={{ background: BG }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Very faint ambient glow behind computer — right side on desktop, center on mobile */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 55% 45% at 68% 48%, rgba(255,255,255,0.018) 0%, transparent 70%)',
        }}
      />

      {/* ── Main layout ───────────────────────────────────────────────────── */}
      <div className="relative z-10 flex-1 flex flex-col lg:flex-row items-center w-full max-w-7xl mx-auto px-6 lg:px-10 pt-6 lg:pt-0 pb-4 gap-8 lg:gap-0">

        {/* ── LEFT: editorial copy ──────────────────────────────────────── */}
        <motion.div
          className="w-full lg:w-[44%] flex flex-col justify-center pt-20 lg:pt-0 order-1"
          initial={prefersReducedMotion ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: 'easeOut' }}
        >
          {/* Eyebrow */}
          <p
            className="text-[10px] font-mono tracking-[0.22em] mb-5"
            style={{ color: TEXT_SECONDARY }}
          >
            BUILT FOR REAL INTERVIEWS
          </p>

          {/* Headline */}
          <h1
            className="text-[clamp(32px,4.5vw,52px)] font-semibold tracking-tight leading-[1.08] mb-5"
            style={{ color: TEXT_PRIMARY }}
          >
            Practice interviews
            <br />
            built like the
            <br />
            real thing.
          </h1>

          {/* Body */}
          <p
            className="text-[clamp(13px,1.1vw,15px)] leading-relaxed max-w-[360px] mb-9"
            style={{ color: TEXT_SECONDARY }}
          >
            Train with realistic technical, behavioral, and coding interviews
            powered by an AI interviewer that adapts to your answers and shows
            you exactly where to improve.
          </p>

          {/* CTAs */}
          <motion.div
            className="flex flex-wrap gap-3"
            initial={prefersReducedMotion ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.18, ease: 'easeOut' }}
          >
            <button
              onClick={handleStartNow}
              className="px-6 py-3 rounded-md text-sm font-semibold transition-all hover:brightness-105 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
              style={{ backgroundColor: ACCENT, color: BG }}
            >
              Start an interview
            </button>
            <AuthLink
              href="/setup"
              className="px-6 py-3 rounded-md text-sm font-medium border transition-colors hover:bg-white/[0.03] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
              style={{
                borderColor: 'rgba(255,255,255,0.12)',
                color: TEXT_PRIMARY,
              }}
            >
              Configure session
            </AuthLink>
          </motion.div>
        </motion.div>

        {/* ── RIGHT: computer visual ────────────────────────────────────── */}
        <motion.div
          className="w-full lg:w-[56%] flex items-center justify-center relative order-2 pb-6 lg:pb-0"
          initial={prefersReducedMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.08 }}
        >
          {/* 3D tilt wrapper */}
          <motion.div
            style={{
              rotateX: prefersReducedMotion ? 0 : rotateX,
              rotateY: prefersReducedMotion ? 0 : rotateY,
              translateX: prefersReducedMotion ? 0 : translateX,
              translateY: prefersReducedMotion ? 0 : translateY,
              transformStyle: 'preserve-3d',
              transformPerspective: 900,
            }}
          >
            {/* Idle float animation — extremely subtle */}
            <motion.div
              animate={
                prefersReducedMotion
                  ? {}
                  : {
                      y: [0, -3, 0, 2, 0],
                    }
              }
              transition={{
                duration: 6,
                repeat: Infinity,
                ease: 'easeInOut',
                repeatType: 'loop',
              }}
            >
              {/* Computer image container — square (1254×1254 source) */}
              <div
                className="relative"
                style={{
                  width: 'clamp(280px, 46vw, 580px)',
                  aspectRatio: '1 / 1',
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/images/computer-removebg-preview.png"
                  alt="Vintage computer terminal showing an AI interview session"
                  draggable={false}
                  style={{
                    width: '100%',
                    height: '100%',
                    display: 'block',
                    userSelect: 'none',
                  }}
                />

                {/* CRT screen overlay — terminal text + effects */}
                <TerminalScreen />
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>

      {/* Scroll-down affordance */}
      <div className="relative z-10 pb-6 flex justify-center">
        <button
          onClick={handleScrollToContent}
          aria-label="Scroll to explore"
          className="group p-1.5 rounded-full border border-white/[0.1] transition-colors hover:border-white/[0.2]"
          style={{ color: TEXT_SECONDARY }}
        >
          <motion.span
            className="block"
            animate={prefersReducedMotion ? {} : { y: [0, 4, 0] }}
            transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
          >
            <ChevronDown className="w-4 h-4" />
          </motion.span>
        </button>
      </div>

      {/* Bottom fade into next section */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-28"
        style={{
          background: `linear-gradient(to bottom, transparent, ${BG})`,
        }}
      />
    </section>
  );
};
