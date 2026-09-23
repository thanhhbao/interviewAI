'use client';

import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ChevronDown, Mic, FileCheck, CheckCircle2 } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { HeroScanOverlay } from '@/components/home/HeroScanOverlay';
import { useSafeReducedMotion, PALETTE } from '@/lib/motion';

// Restrained "Cinematic Technical Instrument" palette. The scene and the scan
// overlay are the visual hero; the UI stays quiet around them. The primary
// accent is the same teal used by every section below the hero (Intelligence
// Console, Scenario, Lifecycle, Candidate Portal, FAQ, Final CTA) and by the
// scan overlay itself — an earlier indigo accent here read as a different
// product once the rest of the site settled on teal.
const ACCENT = PALETTE.accent;
const ACCENT_TEXT = '#07080a';
const TEXT_PRIMARY = '#F2F0EA'; // warm off-white, not pure white
const TEXT_SECONDARY = '#97958F'; // neutral gray, sufficient contrast

export const HeroEnergyStage: React.FC = () => {
  const router = useRouter();
  // useSafeReducedMotion (not framer-motion's useReducedMotion directly):
  // the server always assumes no preference, so branching `initial` on the
  // raw client-only matchMedia result produced a server/client style
  // mismatch on hydration under prefers-reduced-motion. This variant stays
  // deterministically `false` through hydration and only reflects the real
  // preference one tick later — the freeze-avoidance behavior described
  // below is unaffected.
  const prefersReducedMotion = useSafeReducedMotion();
  const { startInterviewSession, targetJob } = useInterview();

  const handleStartNow = () => {
    startInterviewSession('full');
    router.push('/interview');
  };

  const handleScrollToContent = () => {
    document.getElementById('explore')?.scrollIntoView({ behavior: 'smooth' });
  };

  // One short, composed entrance (Motion section of the brief): the scene is
  // already present, so only three quiet beats follow — headline+body, then
  // the CTA, then the instrumentation overlays. No per-element stagger chain.
  //
  // `initial={false}` (rather than omitting initial/animate) is required for
  // prefers-reduced-motion: useReducedMotion() returns null on first render
  // and only resolves to true/false after mount. If the animated props were
  // removed outright on that later render, framer-motion freezes the element
  // at its half-finished in-between state (invisible) instead of snapping to
  // the resting state — initial={false} tells it to skip straight there.
  const contentMotion = {
    initial: prefersReducedMotion ? false : ({ opacity: 0, y: 8 } as const),
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: 'easeOut' as const },
  };
  const ctaMotion = {
    initial: prefersReducedMotion ? false : ({ opacity: 0, y: 6 } as const),
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.4, delay: 0.18, ease: 'easeOut' as const },
  };
  const hudMotion = {
    initial: prefersReducedMotion ? false : ({ opacity: 0 } as const),
    animate: { opacity: 1 },
    transition: { duration: 0.6, delay: 0.35, ease: 'easeOut' as const },
  };

  return (
    <section className="relative w-full min-h-screen flex flex-col overflow-hidden bg-[#07080a] text-[#F2F0EA] select-none">
      {/* Cinematic scan scene — static graded still + pre-rendered transparent
          WebM overlay (scripts/generate_scan_overlay.py), replacing the
          earlier realtime WebGL shader. */}
      <div className="absolute inset-0 w-full h-full pointer-events-none z-0">
        <HeroScanOverlay
          imageSrc="/images/hero-dark.webp"
          videoSrc="/videos/scan-overlay.webm"
          alt="AI robot interviewer and human candidate facing each other across a table"
          className="w-full h-full"
        />
      </div>

      {/* Minimal legibility gradient — bottom only, where text now lives */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-[55%] bg-gradient-to-t from-[#07080a] via-[#07080a]/70 to-transparent z-10" />

      {/* Instrumentation overlays (Phase D) — compact technical readouts, not
          floating glass cards. Hidden below lg: they'd damage a small screen's
          already-tight composition (Responsive section of the brief). */}
      <motion.div
        {...hudMotion}
        className="relative z-20 w-full max-w-7xl mx-auto px-6 pt-20 hidden lg:flex items-start justify-between pointer-events-none"
      >
        <div className="pointer-events-auto flex items-center gap-2.5 px-3 py-2 rounded-md bg-black/30 border border-white/[0.08] backdrop-blur-sm">
          <Mic className="w-3.5 h-3.5 text-[#97958F] shrink-0" />
          <div className="flex flex-col leading-tight">
            <div className="flex items-center gap-1.5 text-[11px] font-mono tracking-wide text-[#97958F]">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400" />
              VOICE PIPELINE · 280ms
            </div>
            <span className="text-[11px] text-[#6f6d68]">Duplex, natural interruptions</span>
          </div>
        </div>

        <div className="pointer-events-auto flex items-center gap-2.5 px-3 py-2 rounded-md bg-black/30 border border-white/[0.08] backdrop-blur-sm">
          <FileCheck className="w-3.5 h-3.5 text-[#97958F] shrink-0" />
          <div className="flex flex-col leading-tight">
            <div className="flex items-center gap-1.5 text-[11px] font-mono tracking-wide text-[#97958F]">
              <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              TARGET MATCH · 94%
            </div>
            <span className="text-[11px] text-[#6f6d68] truncate max-w-[220px]">
              {targetJob.title} · {targetJob.company}
            </span>
          </div>
        </div>
      </motion.div>

      {/* Main content — anchored toward the lower third so it never sits over
          either subject's face (Composition section of the brief). */}
      <div className="relative z-20 flex-1 flex flex-col justify-end">
        <div className="w-full max-w-3xl mx-auto px-6 text-center pb-10 md:pb-14">
          <motion.div {...contentMotion}>
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight leading-[1.12] max-w-2xl mx-auto" style={{ color: TEXT_PRIMARY }}>
              Nail every interview.
              <br />
              Land every job offer.
            </h1>
            <p className="mt-4 text-sm md:text-base max-w-md mx-auto leading-relaxed" style={{ color: TEXT_SECONDARY }}>
              Practice technical, system design, and behavioral interviews with an AI
              interviewer that tests your code and listens in real time.
            </p>
          </motion.div>

          <motion.div {...ctaMotion} className="mt-7 flex flex-wrap items-center justify-center gap-3">
            <button
              onClick={handleStartNow}
              className="px-6 py-3 rounded-md text-sm font-semibold transition-opacity hover:opacity-90 active:opacity-80"
              style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
            >
              Launch interview room
            </button>

            <Link
              href="/setup"
              className="px-6 py-3 rounded-md text-sm font-medium border border-white/[0.12] hover:border-white/[0.2] hover:bg-white/[0.03] transition-colors"
              style={{ color: TEXT_PRIMARY }}
            >
              Customize rubric
            </Link>
          </motion.div>

          <motion.p {...ctaMotion} className="mt-5 text-xs" style={{ color: TEXT_SECONDARY }}>
            Ready for <span style={{ color: TEXT_PRIMARY }}>{targetJob.title}</span> at{' '}
            <span style={{ color: TEXT_PRIMARY }}>{targetJob.company}</span>
          </motion.p>
        </div>

        <div className="pb-6 flex justify-center">
          <button
            onClick={handleScrollToContent}
            className="group p-1.5 rounded-full border border-white/[0.1] text-[#97958F] hover:text-[#F2F0EA] hover:border-white/[0.2] transition-colors"
            aria-label="Scroll to explore"
          >
            <motion.span
              className="block"
              animate={{ y: prefersReducedMotion ? 0 : [0, 4, 0] }}
              transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
            >
              <ChevronDown className="w-4 h-4" />
            </motion.span>
          </button>
        </div>
      </div>
    </section>
  );
};
