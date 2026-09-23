'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';
import { StarsDataField } from '@/components/home/acts/StarsDataField';

export const FinalCtaSection: React.FC = () => {
  const router = useRouter();
  const { startInterviewSession } = useInterview();

  const handleStart = () => {
    startInterviewSession('full');
    router.push('/interview');
  };

  return (
    <section className="relative w-full py-32 md:py-44 overflow-hidden" style={{ backgroundColor: '#050807' }}>
      {/* Static perspective grid — the path ahead, not a scroll-scrubbed prop */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage:
            'linear-gradient(rgba(160,220,210,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(160,220,210,0.05) 1px, transparent 1px)',
          backgroundSize: '64px 64px',
          maskImage: 'radial-gradient(ellipse 60% 60% at 50% 40%, black, transparent)',
          WebkitMaskImage: 'radial-gradient(ellipse 60% 60% at 50% 40%, black, transparent)',
        }}
      />

      {/* A continuation of Act III's autonomous data field — "the path ahead
          is open" — still fully self-driven, never tied to scroll. */}
      <StarsDataField className="w-full h-full" />

      <div
        className="pointer-events-none absolute left-1/2 top-1/3 -translate-x-1/2 -translate-y-1/2 w-[560px] h-[560px] rounded-full"
        aria-hidden="true"
      >
        <div className="w-full h-full rounded-full" style={{ background: `radial-gradient(circle, rgba(20,169,143,0.16), transparent 70%)` }} />
      </div>

      <div className="relative z-10 max-w-2xl mx-auto px-6 text-center">
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5 }}
          className="text-[11px] font-mono tracking-[0.16em] mb-4"
          style={{ color: PALETTE.accent }}
        >
          READY FOR THE REAL THING?
        </motion.p>
        <div className="overflow-hidden">
          <motion.h2
            initial={{ y: '100%' }}
            whileInView={{ y: '0%' }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-4xl md:text-6xl font-semibold tracking-tight leading-[1.1]"
            style={{ color: PALETTE.textPrimary }}
          >
            Practice the interview
            <br />
            before it counts.
          </motion.h2>
        </div>
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="mt-5 text-base md:text-lg"
          style={{ color: PALETTE.textSecondary }}
        >
          Train against realistic technical and behavioral rounds, then review exactly where to improve.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-3"
        >
          <button
            onClick={handleStart}
            className="group px-7 py-3.5 rounded-[10px] text-sm font-semibold transition-opacity hover:opacity-90 active:opacity-80 flex items-center gap-2"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Start an interview
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </button>
          <Link
            href="/setup"
            className="px-7 py-3.5 rounded-[10px] text-sm font-medium transition-colors"
            style={{ color: PALETTE.textPrimary, border: `1px solid ${PALETTE.border}` }}
          >
            Configure session
          </Link>
        </motion.div>
      </div>
    </section>
  );
};
