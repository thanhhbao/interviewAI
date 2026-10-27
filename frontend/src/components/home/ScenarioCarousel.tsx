'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PALETTE, fadeUp, fadeUpSmall, viewportOnce, useSafeReducedMotion } from '@/lib/motion';
import { ChapterMarker } from '@/components/home/acts/ChapterMarker';

// Section 02 / SIMULATE
// Shows HOW the AI adapts dynamically to a candidate's answers —
// four distinct adaptive moments, presented as a master-detail selector.
// Desktop: intro left, selector + detail right.
// Mobile: stacked, tap to expand.

interface AdaptiveMoment {
  id: string;
  index: string;       // zero-padded e.g. "01"
  label: string;       // short name shown in the list
  title: string;       // ALLCAPS heading in the detail panel
  trigger: string;     // what the candidate did
  quote: string;       // the AI's follow-up question (as a quoted string)
  contextLabel: string;
  contextValue: string;
  metricLabel: string;
  metricValue: string;
}

const MOMENTS: AdaptiveMoment[] = [
  {
    id: 'probe',
    index: '01',
    label: 'Probe deeper',
    title: 'PROBE DEEPER',
    trigger: 'Candidate mentioned caching.',
    quote:
      `"You added a cache layer — walk me through how you'd handle invalidation when the underlying data changes, and what happens when that cache misses under peak load."`,
    contextLabel: 'Signal detected',
    contextValue: 'Surface-level answer',
    metricLabel: 'Follow-up generated',
    metricValue: '380 ms',
  },
  {
    id: 'challenge',
    index: '02',
    label: 'Challenge assumptions',
    title: 'CHALLENGE ASSUMPTIONS',
    trigger: 'Candidate chose eventual consistency.',
    quote:
      '"You chose eventual consistency — what happens if two regions accept conflicting writes at the same time? How do you surface that conflict to the end user?"',
    contextLabel: 'Context detected',
    contextValue: 'Distributed systems',
    metricLabel: 'Follow-up generated',
    metricValue: '420 ms',
  },
  {
    id: 'raise',
    index: '03',
    label: 'Raise the bar',
    title: 'RAISE THE BAR',
    trigger: 'Candidate gave a strong, confident answer.',
    quote:
      `"That's a solid foundation. Now assume you're designing this for 100× the traffic with a five-nines SLA. What breaks first, and how do you get ahead of it?"`,
    contextLabel: 'Signal detected',
    contextValue: 'Strong performance',
    metricLabel: 'Difficulty adjusted',
    metricValue: '+2 levels',
  },
  {
    id: 'signal',
    index: '04',
    label: 'Follow the signal',
    title: 'FOLLOW THE SIGNAL',
    trigger: 'Candidate revealed an interesting gap.',
    quote:
      `"You mentioned you haven't worked with Kafka directly — tell me about a time you had to reason about message ordering without a dedicated broker. What tradeoffs did you make?"`,
    contextLabel: 'Signal detected',
    contextValue: 'Experience gap',
    metricLabel: 'Pivot generated',
    metricValue: '295 ms',
  },
];

const EASE = [0.22, 1, 0.36, 1] as const;

// Detail panel — transitions on every moment change
function DetailPanel({ moment, reduced }: { moment: AdaptiveMoment; reduced: boolean }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={moment.id}
        initial={reduced ? false : { opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: reduced ? 0 : -6 }}
        transition={{ duration: 0.28, ease: EASE }}
        className="flex flex-col"
      >
        {/* Title */}
        <p
          className="text-[10px] font-mono tracking-[0.18em]"
          style={{ color: PALETTE.accent }}
        >
          {moment.title}
        </p>

        {/* Trigger context */}
        <p
          className="mt-3 text-[12px] font-mono"
          style={{ color: PALETTE.textSecondary }}
        >
          {moment.trigger}
        </p>

        {/* AI's follow-up question */}
        <blockquote
          className="mt-4 text-[clamp(14px,1.3vw,17px)] leading-[1.6] font-normal"
          style={{ color: PALETTE.textPrimary }}
        >
          {moment.quote}
        </blockquote>

        {/* Metadata row */}
        <div
          className="mt-6 pt-5 grid grid-cols-2 gap-x-6 gap-y-3"
          style={{ borderTop: `1px solid ${PALETTE.border}` }}
        >
          <div>
            <p className="text-[10px] font-mono tracking-[0.12em] uppercase" style={{ color: PALETTE.textSecondary }}>
              {moment.contextLabel}
            </p>
            <p className="mt-1 text-[13px]" style={{ color: PALETTE.textPrimary }}>
              {moment.contextValue}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-mono tracking-[0.12em] uppercase" style={{ color: PALETTE.textSecondary }}>
              {moment.metricLabel}
            </p>
            <p className="mt-1 text-[13px] font-mono" style={{ color: PALETTE.accent }}>
              {moment.metricValue}
            </p>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// One row in the selector list
function SelectorRow({
  moment,
  isActive,
  onActivate,
}: {
  moment: AdaptiveMoment;
  isActive: boolean;
  onActivate: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      aria-label={`${moment.index} — ${moment.label}`}
      onClick={onActivate}
      onMouseEnter={onActivate}
      className="group w-full text-left flex items-center gap-4 py-4 transition-colors outline-none focus-visible:ring-1 focus-visible:ring-white/30 rounded-sm"
      style={{ borderBottom: `1px solid ${PALETTE.border}` }}
    >
      {/* Index */}
      <span
        className="text-[11px] font-mono tabular-nums shrink-0 w-6 transition-colors duration-200"
        style={{ color: isActive ? PALETTE.accent : PALETTE.textSecondary }}
      >
        {moment.index}
      </span>

      {/* Label */}
      <span
        className="flex-1 text-[14px] font-medium transition-colors duration-200"
        style={{ color: isActive ? PALETTE.textPrimary : PALETTE.textSecondary }}
      >
        {moment.label}
      </span>

      {/* Active indicator bar */}
      <motion.span
        className="shrink-0 h-[1px] rounded-full"
        animate={{
          width: isActive ? 24 : 0,
          opacity: isActive ? 1 : 0,
          backgroundColor: PALETTE.accent,
        }}
        transition={{ duration: 0.22, ease: EASE }}
        style={{ display: 'block' }}
      />

      {/* Arrow */}
      <span
        className="shrink-0 text-[11px] font-mono transition-colors duration-200"
        style={{ color: isActive ? PALETTE.accent : PALETTE.border }}
      >
        →
      </span>
    </button>
  );
}

export const ScenarioCarousel: React.FC = () => {
  const reduced = useSafeReducedMotion();
  const [activeIndex, setActiveIndex] = useState(0);
  const activeMoment = MOMENTS[activeIndex];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') setActiveIndex(i => Math.min(i + 1, MOMENTS.length - 1));
    if (e.key === 'ArrowUp') setActiveIndex(i => Math.max(i - 1, 0));
  };

  return (
    <section className="relative w-full py-24 md:py-32">
      <div className="max-w-[1240px] mx-auto px-6">

        {/* ── Desktop: 2-column; Mobile: stacked ─────────────────────── */}
        <div className="lg:grid lg:grid-cols-[1fr_1.4fr] lg:gap-20 xl:gap-28">

          {/* LEFT: section intro */}
          <motion.div
            variants={fadeUp}
            initial="hidden"
            whileInView="show"
            viewport={viewportOnce}
            className="mb-14 lg:mb-0 lg:sticky lg:top-28 lg:self-start"
          >
            <ChapterMarker number="02" label="SIMULATE" />

            <p
              className="text-[11px] font-mono tracking-[0.16em] uppercase mt-4 mb-4"
              style={{ color: PALETTE.textSecondary }}
            >
              ADAPTIVE INTERVIEWING
            </p>

            <h2
              className="font-semibold tracking-tight leading-[1.08]"
              style={{ fontSize: 'clamp(2rem, 3.8vw, 3.2rem)', color: PALETTE.textPrimary }}
            >
              No two interviews
              <br />
              follow the same script.
            </h2>

            <p
              className="mt-5 text-[14px] leading-relaxed max-w-[340px]"
              style={{ color: PALETTE.textSecondary }}
            >
              InterviewPal listens to how you answer, then decides whether to probe
              deeper, challenge an assumption, raise the difficulty, or follow an
              important signal.
            </p>

            {/* Terminal-style cue */}
            <p
              className="mt-8 text-[11px] font-mono"
              style={{ color: PALETTE.textSecondary, opacity: 0.45 }}
            >
              {`0${activeIndex + 1}_`}
            </p>
          </motion.div>

          {/* RIGHT: selector + detail */}
          <motion.div
            variants={fadeUp}
            initial="hidden"
            whileInView="show"
            viewport={viewportOnce}
          >
            {/* Selector list */}
            <div
              role="tablist"
              aria-label="Adaptive interview moments"
              onKeyDown={handleKeyDown}
              tabIndex={0}
              className="outline-none"
              style={{ borderTop: `1px solid ${PALETTE.border}` }}
            >
              {MOMENTS.map((moment, i) => (
                <SelectorRow
                  key={moment.id}
                  moment={moment}
                  isActive={activeIndex === i}
                  onActivate={() => setActiveIndex(i)}
                />
              ))}
            </div>

            {/* Detail panel */}
            <div
              className="mt-8 pt-8"
              role="tabpanel"
              aria-label={`Detail: ${activeMoment.label}`}
            >
              <DetailPanel moment={activeMoment} reduced={reduced} />
            </div>
          </motion.div>
        </div>

      </div>
    </section>
  );
};
