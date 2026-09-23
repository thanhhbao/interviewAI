'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2 } from 'lucide-react';
import { ChapterMarker } from '@/components/home/acts/ChapterMarker';
import { fadeUp, viewportOnce } from '@/lib/motion';

// Continues the hero's "Cinematic Technical Instrument" palette rather than
// the old four-color SaaS-card system. Bright/deep teal match the hero scan
// overlay's own core/trail colors (#78FFE7 / #14A98F) so this section reads
// as the same product, not a different visual system.
const SURFACE = '#0A1110';
const BORDER = 'rgba(255,255,255,0.07)';
const BORDER_TEAL = 'rgba(160,220,210,0.08)';
const TEXT_PRIMARY = '#ECEDE9';
const TEXT_SECONDARY = '#858E8A';
const ACCENT = '#55E5CF';
const ACCENT_BRIGHT = '#78FFE7';
const ACCENT_DEEP = '#14A98F';

const WAVEFORM_HEIGHTS = [5, 11, 7, 16, 9, 13, 6, 18, 10, 8, 14, 6, 12, 9, 5, 15, 7, 11];

const behavioralSignals = [
  { label: 'Situation / Context', value: 92 },
  { label: 'Action / Ownership', value: 96 },
  { label: 'Result / Impact', value: 88 },
];

const CODE_LINES = [
  { text: 'function twoSum(nums, target) {', tone: 'primary' as const },
  { text: '  const seen = new Map();', tone: 'secondary' as const },
  { text: '  for (let i = 0; i < nums.length; i++) {', tone: 'secondary' as const },
  { text: '    const need = target - nums[i];', tone: 'secondary' as const },
  { text: '    if (seen.has(need)) return [seen.get(need), i];', tone: 'accent' as const },
  { text: '    seen.set(nums[i], i);', tone: 'secondary' as const },
  { text: '  }', tone: 'secondary' as const },
  { text: '}', tone: 'primary' as const },
];

function ProgressBar({ value, delay = 0 }: { value: number; delay?: number }) {
  return (
    <div className="h-[3px] w-full rounded-full bg-white/[0.06] overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{ backgroundColor: ACCENT }}
        initial={{ width: 0 }}
        whileInView={{ width: `${value}%` }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.7, delay, ease: 'easeOut' }}
      />
    </div>
  );
}

function PanelLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-mono tracking-[0.12em] text-[#6f6d68]">{children}</p>
  );
}

export const IntelligenceConsole: React.FC = () => {
  return (
    <section className="relative w-full">
      <div className="max-w-[1160px] mx-auto px-6 py-24 md:py-32">
        {/* Header */}
        <motion.div
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={viewportOnce}
          className="max-w-2xl mx-auto text-center mb-14 md:mb-16"
        >
          <div className="flex justify-center">
            <ChapterMarker number="01" label="UNDERSTAND" />
          </div>
          <p className="text-[11px] font-mono tracking-[0.14em]" style={{ color: ACCENT }}>
            INTERVIEW INTELLIGENCE
          </p>
          <h2
            className="mt-4 text-3xl sm:text-4xl md:text-[2.75rem] font-semibold tracking-tight leading-[1.15]"
            style={{ color: TEXT_PRIMARY }}
          >
            One interview.
            <br />
            Every signal analyzed.
          </h2>
          <p className="mt-5 text-sm md:text-base leading-relaxed" style={{ color: TEXT_SECONDARY }}>
            InterviewAI listens to your answer, evaluates your reasoning, tests your code, and
            measures how well your experience matches the role — all within the same session.
          </p>
        </motion.div>

        {/* Console */}
        <motion.div
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={viewportOnce}
          className="rounded-[14px] overflow-hidden"
          style={{ backgroundColor: SURFACE, border: `1px solid ${BORDER}` }}
        >
          {/* Live interview */}
          <div className="p-6 md:p-8">
            <div className="flex items-center justify-between">
              <PanelLabel>LIVE INTERVIEW</PanelLabel>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5 text-[11px] font-mono tracking-wide" style={{ color: TEXT_SECONDARY }}>
                  <span
                    className="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
                    style={{ backgroundColor: ACCENT_BRIGHT }}
                  />
                  LISTENING
                </div>
                <span className="text-[11px] font-mono" style={{ color: TEXT_SECONDARY }}>
                  280&nbsp;ms
                </span>
              </div>
            </div>

            <div className="mt-6 space-y-4">
              <div>
                <p className="text-[11px] mb-1.5" style={{ color: TEXT_SECONDARY }}>
                  AI Interviewer
                </p>
                <p className="text-[15px] md:text-base leading-relaxed" style={{ color: TEXT_PRIMARY }}>
                  &ldquo;How would you prevent a cache stampede in this architecture?&rdquo;
                </p>
              </div>
              <div>
                <p className="text-[11px] mb-1.5" style={{ color: TEXT_SECONDARY }}>
                  Candidate
                </p>
                <p className="text-[15px] md:text-base leading-relaxed" style={{ color: '#c7cac5' }}>
                  &ldquo;I&rsquo;d separate request coalescing from cache expiration, then add a
                  short-lived lock per key so only one request repopulates it&hellip;&rdquo;
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-end gap-[3px] h-6">
              {WAVEFORM_HEIGHTS.map((h, i) => (
                <motion.span
                  key={i}
                  className="w-[3px] rounded-full"
                  style={{ backgroundColor: ACCENT_DEEP, height: h }}
                  animate={{ scaleY: [0.4, 1, 0.5, 0.9, 0.4] }}
                  transition={{
                    duration: 1.6 + (i % 5) * 0.15,
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: i * 0.05,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Code reasoning / Role context */}
          <div className="grid grid-cols-1 md:grid-cols-2" style={{ borderTop: `1px solid ${BORDER_TEAL}` }}>
            <div className="order-2 md:order-1 p-6 md:p-8 md:border-r" style={{ borderColor: BORDER_TEAL }}>
              <PanelLabel>CODE REASONING</PanelLabel>
              <div className="mt-4 rounded-[10px] p-4 font-mono text-[12.5px] leading-[1.7] overflow-x-auto" style={{ backgroundColor: 'rgba(255,255,255,0.025)', border: `1px solid ${BORDER}` }}>
                {CODE_LINES.map((line, i) => (
                  <div
                    key={i}
                    style={{
                      color: line.tone === 'accent' ? ACCENT_BRIGHT : line.tone === 'primary' ? '#c7cac5' : '#7d8783',
                      whiteSpace: 'pre',
                    }}
                  >
                    {line.text}
                  </div>
                ))}
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12px]" style={{ color: TEXT_SECONDARY }}>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" style={{ color: ACCENT }} />
                  3/3 tests
                </span>
                <span>O(n)</span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" style={{ color: ACCENT }} />
                  Reasoning captured
                </span>
              </div>
            </div>

            <div className="order-1 md:order-2 p-6 md:p-8 border-b md:border-b-0" style={{ borderColor: BORDER_TEAL }}>
              <PanelLabel>ROLE CONTEXT</PanelLabel>
              <div className="mt-4">
                <p className="text-[11px] mb-1" style={{ color: TEXT_SECONDARY }}>
                  Target role
                </p>
                <p className="text-lg font-medium" style={{ color: TEXT_PRIMARY }}>
                  Staff Engineer
                </p>
              </div>

              <div className="mt-5">
                <div className="flex items-center justify-between text-[12px] mb-1.5">
                  <span style={{ color: TEXT_SECONDARY }}>Alignment</span>
                  <span style={{ color: ACCENT_BRIGHT }}>94%</span>
                </div>
                <ProgressBar value={94} />
              </div>

              <div className="mt-5">
                <p className="text-[11px] mb-2" style={{ color: TEXT_SECONDARY }}>
                  Signals
                </p>
                <div className="flex flex-wrap gap-2">
                  {['Distributed Systems', 'Kafka', 'Concurrency'].map((s) => (
                    <span
                      key={s}
                      className="text-[12px] px-2.5 py-1 rounded-[6px]"
                      style={{ color: '#c7cac5', border: `1px solid ${BORDER}` }}
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Behavioral signals */}
          <div className="p-6 md:p-8" style={{ borderTop: `1px solid ${BORDER_TEAL}` }}>
            <PanelLabel>BEHAVIORAL SIGNALS</PanelLabel>
            <div className="mt-5 space-y-4 max-w-xl">
              {behavioralSignals.map((s, i) => (
                <div key={s.label} className="flex items-center gap-4">
                  <span className="text-[13px] w-40 shrink-0" style={{ color: TEXT_SECONDARY }}>
                    {s.label}
                  </span>
                  <ProgressBar value={s.value} delay={i * 0.1} />
                  <span className="text-[13px] w-8 text-right font-mono" style={{ color: TEXT_PRIMARY }}>
                    {s.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
