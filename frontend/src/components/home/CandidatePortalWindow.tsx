'use client';

import React, { useRef } from 'react';
import { motion, useMotionValue, useInView, animate } from 'framer-motion';
import { Home, FileText, Mic, BarChart3, Settings, CheckCircle2, ArrowUpRight } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { samplePastInterviews } from '@/lib/mockData';
import { PALETTE, fadeUp, viewportOnce } from '@/lib/motion';
import { ChapterMarker } from '@/components/home/acts/ChapterMarker';

function CountUp({ value, suffix = '' }: { value: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const mv = useMotionValue(0);

  React.useEffect(() => {
    if (!inView) return;
    const controls = animate(mv, value, { duration: 1, ease: [0.16, 1, 0.3, 1] });
    const unsub = mv.on('change', (v) => {
      if (ref.current) ref.current.textContent = `${Math.round(v)}${suffix}`;
    });
    return () => {
      controls.stop();
      unsub();
    };
  }, [inView, mv, value, suffix]);

  return <span ref={ref}>0{suffix}</span>;
}

const navItems = [
  { icon: Home, label: 'Overview', active: true },
  { icon: FileText, label: 'Resume Match' },
  { icon: Mic, label: 'Interview Room' },
  { icon: BarChart3, label: 'Reports' },
  { icon: Settings, label: 'Settings' },
];

export const CandidatePortalWindow: React.FC = () => {
  const { candidate, targetJob } = useInterview();

  const readiness = 82;
  const chartValues = [40, 55, 48, 66, 60, 74, readiness];

  return (
    <section className="relative w-full py-24 md:py-32 overflow-hidden">
      <div className="max-w-[1160px] mx-auto px-6">
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={viewportOnce} className="max-w-2xl mb-14 space-y-3">
          <ChapterMarker number="03" label="IMPROVE" />
          <p className="text-[11px] font-mono tracking-[0.12em]" style={{ color: PALETTE.accent }}>
            YOUR PORTAL
          </p>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            Every session, tracked in one place.
          </h2>
          <p className="text-sm" style={{ color: PALETTE.textSecondary }}>
            Jump back into your curated interview track and see exactly where you stand.
          </p>
        </motion.div>

        <div className="relative flex justify-center py-10">
          {/* Background context chips — reveal once when the section enters
              view, then hold still. No scroll-position dependence. */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={viewportOnce}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="hidden md:flex absolute -top-2 left-2 items-center gap-1.5 text-[11px] font-mono px-3 py-1.5 rounded-full"
          >
            <span
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full"
              style={{ border: `1px solid ${PALETTE.borderTeal}`, color: PALETTE.accent, backgroundColor: PALETTE.surfaceFaint }}
            >
              {targetJob.matchScore}% MATCH
            </span>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={viewportOnce}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="hidden md:block absolute top-16 right-6"
          >
            <span
              className="px-3 py-1.5 rounded-full text-[11px] font-mono"
              style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary, backgroundColor: PALETTE.surfaceFaint }}
            >
              {targetJob.questionCount} QUESTIONS READY
            </span>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={viewportOnce}
            transition={{ duration: 0.5, delay: 0.25 }}
            className="hidden md:block absolute bottom-10 right-16"
          >
            <span
              className="px-3 py-1.5 rounded-full text-[11px] font-mono"
              style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary, backgroundColor: PALETTE.surfaceFaint }}
            >
              L5 SENIOR
            </span>
          </motion.div>

          {/* Product window */}
          <motion.div
            style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={viewportOnce}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="relative w-full max-w-[820px] rounded-[16px] overflow-hidden shadow-[0_40px_100px_-30px_rgba(0,0,0,0.6)]"
          >
            {/* Title bar */}
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: `1px solid ${PALETTE.borderTeal}` }}>
              <div className="flex gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.12)' }} />
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.12)' }} />
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.12)' }} />
              </div>
              <span className="ml-2 text-[11px] font-mono" style={{ color: PALETTE.textSecondary }}>
                InterviewPal — Candidate Portal
              </span>
            </div>

            <div className="grid grid-cols-[160px_1fr]">
              {/* Sidebar */}
              <motion.div
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={viewportOnce}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="p-4 space-y-1 hidden sm:block"
                style={{ borderRight: `1px solid ${PALETTE.borderTeal}` }}
              >
                {navItems.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center gap-2.5 px-2.5 py-2 rounded-[8px] text-[12px]"
                    style={{
                      color: item.active ? PALETTE.textPrimary : PALETTE.textSecondary,
                      backgroundColor: item.active ? 'rgba(255,255,255,0.04)' : 'transparent',
                    }}
                  >
                    <item.icon className="w-3.5 h-3.5" />
                    {item.label}
                  </div>
                ))}
              </motion.div>

              {/* Main content */}
              <div className="p-6 md:p-7 space-y-6">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-[11px]" style={{ color: PALETTE.textSecondary }}>{targetJob.company}</p>
                    <h3 className="text-lg font-medium mt-0.5" style={{ color: PALETTE.textPrimary }}>
                      {targetJob.title}
                    </h3>
                  </div>
                  <a
                    href="/setup"
                    className="flex items-center gap-1 text-[12px] font-medium px-3 py-1.5 rounded-[8px] transition-colors"
                    style={{ color: '#07080a', backgroundColor: PALETTE.accent }}
                  >
                    Launch
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </a>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>Readiness</span>
                    <span className="text-[13px] font-mono" style={{ color: PALETTE.accentBright }}>
                      <CountUp value={readiness} />
                    </span>
                  </div>
                  <div className="h-[4px] rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: PALETTE.accent }}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${readiness}%` }}
                      viewport={viewportOnce}
                      transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                </div>

                {/* Mini chart */}
                <div>
                  <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Performance trend</p>
                  <div className="flex items-end gap-1.5 h-16">
                    {chartValues.map((v, i) => (
                      <motion.div
                        key={i}
                        className="flex-1 rounded-t-[3px]"
                        style={{ backgroundColor: i === chartValues.length - 1 ? PALETTE.accentBright : 'rgba(85,229,207,0.25)' }}
                        initial={{ height: 0 }}
                        whileInView={{ height: `${v}%` }}
                        viewport={viewportOnce}
                        transition={{ duration: 0.6, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                      />
                    ))}
                  </div>
                </div>

                {/* Recent sessions */}
                <div className="space-y-2 pt-2" style={{ borderTop: `1px solid ${PALETTE.borderTeal}` }}>
                  {samplePastInterviews.slice(0, 2).map((item) => (
                    <div key={item.id} className="flex items-center justify-between py-1.5">
                      <div className="flex items-center gap-2 min-w-0">
                        <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: PALETTE.accent }} />
                        <span className="text-[12px] truncate" style={{ color: PALETTE.textPrimary }}>
                          {item.role} · {item.company}
                        </span>
                      </div>
                      <span className="text-[12px] font-mono shrink-0" style={{ color: PALETTE.textSecondary }}>
                        {item.score}/100
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};
