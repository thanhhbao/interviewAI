'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useScroll, useTransform, useMotionValue, type MotionValue } from 'framer-motion';
import { FileText, Mic, Code2, Users2, BarChart3, CheckCircle2 } from 'lucide-react';
import { PALETTE, fadeUp, viewportOnce, useSafeReducedMotion } from '@/lib/motion';

// Scroll-driven path: strokeDashoffset tracks scrollYProgress so the fill
// advances as the user scrolls down and retreats when they scroll back up.
// No CSS @keyframes loop — the MotionValue is the only source of truth.

interface Stage {
  id: string;
  index: string;
  label: string;
  icon: React.ElementType;
  description: string;
  point: [number, number];
  side: 'left' | 'right';
}

const VIEWBOX_W = 1000;
const VIEWBOX_H = 1400;

const desktopStages: Stage[] = [
  { id: 'resume',    index: '01', label: 'Resume + JD Sync',     icon: FileText,  description: 'Role context is extracted before the interview begins.',           point: [180,  90],   side: 'right' },
  { id: 'interview', index: '02', label: 'Live AI Interview',     icon: Mic,       description: 'A natural voice conversation with adaptive follow-ups.',           point: [800, 340],   side: 'left'  },
  { id: 'technical', index: '03', label: 'Technical Round',       icon: Code2,     description: 'Code and reasoning are evaluated together, live.',                 point: [210, 640],   side: 'right' },
  { id: 'behavioral',index: '04', label: 'Behavioral Signals',    icon: Users2,    description: 'Ownership, clarity and impact are measured as you speak.',         point: [780, 970],   side: 'left'  },
  { id: 'report',    index: '05', label: 'Diagnostic Report',     icon: BarChart3, description: 'Every signal converges into one actionable report.',               point: [260, 1270],  side: 'right' },
];

const mobileStages: Stage[] = desktopStages.map((s, i) => ({
  ...s,
  point: [i % 2 === 0 ? 130 : 170, 90 + i * 300] as [number, number],
}));

function buildPath(points: [number, number][]): string {
  let d = `M ${points[0][0]},${points[0][1]}`;
  for (let i = 1; i < points.length; i++) {
    const [ax, ay] = points[i - 1];
    const [bx, by] = points[i];
    const midX = (ax + bx) / 2;
    d += ` C ${midX},${ay} ${midX},${by} ${bx},${by}`;
  }
  return d;
}

function measureLength(d: string): number {
  if (typeof document === 'undefined') return 0;
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  return path.getTotalLength();
}

function usePathProgress(stages: Stage[]) {
  const [pathLength, setPathLength] = useState(0);
  const d = useMemo(() => buildPath(stages.map((s) => s.point)), [stages]);

  useEffect(() => {
    setPathLength(measureLength(d));
  }, [d]);

  const fractions = useMemo(() => {
    if (pathLength <= 0) return stages.map((_, i) => i / (stages.length - 1));
    return stages.map((_, i) => {
      if (i === 0) return 0;
      const prefix = buildPath(stages.slice(0, i + 1).map((s) => s.point));
      return measureLength(prefix) / pathLength;
    });
  }, [stages, pathLength]);

  return { d, pathLength, fractions };
}

// ── Stage visual content (unchanged) ────────────────────────────────────────

function StageVisual({ id }: { id: string }) {
  switch (id) {
    case 'resume':
      return (
        <div className="flex items-center gap-3">
          <span className="text-[11px] px-2 py-1 rounded-[6px]" style={{ border: `1px solid ${PALETTE.border}`, color: '#c7cac5' }}>
            Resume.pdf
          </span>
          <span style={{ color: PALETTE.textSecondary }}>↘</span>
          <span className="text-[13px] font-mono" style={{ color: PALETTE.accentBright }}>94% match</span>
        </div>
      );
    case 'interview':
      return (
        <div className="flex items-end gap-[3px] h-5">
          {[5, 10, 7, 14, 8, 12, 6].map((h, i) => (
            <span key={i} className="w-[3px] rounded-full" style={{ height: h, backgroundColor: PALETTE.deepTeal }} />
          ))}
        </div>
      );
    case 'technical':
      return (
        <div className="flex items-center gap-2 text-[12px]" style={{ color: PALETTE.textSecondary }}>
          <CheckCircle2 className="w-3.5 h-3.5" style={{ color: PALETTE.accent }} />
          3/3 tests passed
        </div>
      );
    case 'behavioral':
      return (
        <div className="space-y-1.5 w-32">
          {[92, 96, 88].map((v, i) => (
            <div key={i} className="h-[3px] rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
              <div className="h-full rounded-full" style={{ width: `${v}%`, backgroundColor: PALETTE.accent }} />
            </div>
          ))}
        </div>
      );
    case 'report':
      return (
        <div className="flex items-baseline gap-1.5">
          <span className="text-xl font-semibold font-mono" style={{ color: PALETTE.accentBright }}>92</span>
          <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>composite score</span>
        </div>
      );
    default:
      return null;
  }
}

// ── Stage block — card highlight driven by scroll progress ───────────────────

function StageBlock({
  stage,
  delay,
  fillFraction,
  scrollProgress,
  isDesktop,
  viewBoxH,
}: {
  stage: Stage;
  delay: number;
  fillFraction: number;
  scrollProgress: MotionValue<number>;
  isDesktop: boolean;
  viewBoxH: number;
}) {
  const Icon = stage.icon;

  // Window around when the fill front reaches this stage
  const lo = Math.max(0, fillFraction - 0.06);
  const hi = Math.min(1, fillFraction + 0.14);

  // Card border/glow highlight peaks when fill arrives
  const cardHighlight = useTransform(scrollProgress, [lo, fillFraction, hi], [0, 1, 0]);

  const displayedStage = isDesktop ? stage : { ...stage, side: 'right' as const };

  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={viewportOnce}
      transition={{ duration: 0.6, delay }}
      className="absolute w-[220px] sm:w-[300px]"
      style={{
        left: `${(displayedStage.point[0] / VIEWBOX_W) * 100}%`,
        top: `${(displayedStage.point[1] / viewBoxH) * 100}%`,
        maxWidth: 'calc(100vw - 64px)',
        transform:
          displayedStage.side === 'right'
            ? 'translate(24px, -50%)'
            : 'translate(calc(-100% - 24px), -50%)',
      }}
    >
      <p className="text-[11px] font-mono mb-1" style={{ color: PALETTE.accent }}>
        {stage.index}
      </p>
      <h3 className="text-base font-medium flex items-center gap-2 mb-1.5" style={{ color: PALETTE.textPrimary }}>
        <Icon className="w-4 h-4" />
        {stage.label}
      </h3>
      <p className="text-[13px] leading-relaxed mb-3" style={{ color: PALETTE.textSecondary }}>
        {stage.description}
      </p>

      {/* Card with scroll-driven border highlight */}
      <div className="relative rounded-[10px] p-3" style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}>
        <motion.div
          className="absolute inset-0 rounded-[10px] pointer-events-none"
          style={{
            opacity: cardHighlight,
            boxShadow: '0 0 0 1px rgba(255,255,255,0.28), 0 0 22px rgba(231,232,233,0.12)',
          }}
        />
        <StageVisual id={stage.id} />
      </div>
    </motion.div>
  );
}

// ── SVG circle that pulses when the fill front passes through it ─────────────

function StageCircle({
  stage,
  fraction,
  scrollProgress,
}: {
  stage: Stage;
  fraction: number;
  scrollProgress: MotionValue<number>;
}) {
  const lo = Math.max(0, fraction - 0.06);
  const hi = Math.min(1, fraction + 0.06);

  const scale   = useTransform(scrollProgress, [lo, fraction, hi], [1,    1.25, 1]);
  const ringOp  = useTransform(scrollProgress, [lo, fraction, hi], [0,    0.75, 0]);

  const cx = stage.point[0];
  const cy = stage.point[1];

  return (
    <g style={{ transformOrigin: `${cx}px ${cy}px` }}>
      {/* Outer ring that flares as the fill arrives */}
      <motion.circle
        cx={cx} cy={cy} r={13}
        fill="none"
        stroke="white"
        strokeWidth={1.5}
        style={{
          opacity: ringOp,
          scale,
          transformOrigin: `${cx}px ${cy}px`,
        }}
      />
      {/* Core dot */}
      <motion.circle
        cx={cx} cy={cy} r={7}
        fill={PALETTE.surface}
        stroke={PALETTE.accent}
        strokeWidth={2}
        style={{
          scale,
          transformOrigin: `${cx}px ${cy}px`,
        }}
      />
    </g>
  );
}

// ── SVG path — fill driven by MotionValue, no CSS animation ─────────────────

function JourneyPath({
  d,
  pathLength,
  fractions,
  stages,
  viewBoxH,
  scrollProgress,
}: {
  d: string;
  pathLength: number;
  fractions: number[];
  stages: Stage[];
  viewBoxH: number;
  scrollProgress: MotionValue<number>;
}) {
  const show = pathLength > 0;

  // drawProgress: 0 = nothing drawn, 1 = full path drawn.
  // Framer Motion's pathLength motion value handles the strokeDasharray/offset
  // internally, measuring the actual SVG element, and always draws FORWARD from
  // the path start (stage 01) to the path end (stage 05).
  // pathOffset: 0 means drawing starts at the very beginning of the path.
  const drawProgress = useTransform(scrollProgress, [0, 1], [0, 1]);

  return (
    <svg
      viewBox={`0 0 ${VIEWBOX_W} ${viewBoxH}`}
      preserveAspectRatio="none"
      className="absolute inset-0 w-full h-full"
      aria-hidden="true"
    >
      {/* Circles before path layers — lit path draws on top, no gap at nodes */}
      {stages.map((stage, i) => (
        <StageCircle
          key={stage.id}
          stage={stage}
          fraction={fractions[i]}
          scrollProgress={scrollProgress}
        />
      ))}

      {/* Dim base rail — always visible */}
      <path d={d} fill="none" stroke="rgba(255,255,255,0.10)" strokeWidth={2} />

      {show && (
        <>
          {/* Wide soft bloom */}
          <motion.path
            d={d} fill="none"
            stroke={PALETTE.accent}
            strokeWidth={16}
            strokeLinecap="round"
            style={{ pathLength: drawProgress, pathOffset: 0, opacity: 0.13, filter: 'blur(8px)' }}
          />
          {/* Tighter bloom */}
          <motion.path
            d={d} fill="none"
            stroke={PALETTE.accent}
            strokeWidth={7}
            strokeLinecap="round"
            style={{ pathLength: drawProgress, pathOffset: 0, opacity: 0.28, filter: 'blur(3px)' }}
          />
          {/* Sharp core */}
          <motion.path
            d={d} fill="none"
            stroke={PALETTE.accentBright}
            strokeWidth={2.2}
            strokeLinecap="round"
            style={{ pathLength: drawProgress, pathOffset: 0 }}
          />
        </>
      )}
    </svg>
  );
}

// ── Root component ───────────────────────────────────────────────────────────

export const LifecycleJourney: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const [isDesktop, setIsDesktop] = useState(false);
  const prefersReducedMotion = useSafeReducedMotion();

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 900px)');
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);

  const stages = isDesktop ? desktopStages : mobileStages;
  const viewBoxH = isDesktop ? VIEWBOX_H : 90 + (mobileStages.length - 1) * 300 + 120;
  const { d, pathLength, fractions } = usePathProgress(stages);

  // Track scroll across the full section. Using a wide offset so scrollYProgress
  // gives us a stable 0→1 range that we can then remap with useTransform.
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  });

  // Remap to a tighter active window: the path starts drawing once the section
  // has scrolled slightly into view (8%) and is fully drawn well before the
  // section exits (78%). Values outside this range clamp automatically.
  // Result: the bright line feels directly attached to the scroll position —
  // it reaches stage 05 while the section is still well within the viewport,
  // not after the user has scrolled past it.
  const staticProgress = useMotionValue(1);
  const rawProgress = useTransform(scrollYProgress, [0.08, 0.78], [0, 1]);
  const activeProgress = prefersReducedMotion ? staticProgress : rawProgress;

  return (
    <section ref={sectionRef} className="relative w-full py-24 md:py-32 overflow-hidden">
      <div className="max-w-[1160px] mx-auto px-6">
        <motion.div
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={viewportOnce}
          className="max-w-2xl mb-16 space-y-3"
        >
          <p className="text-[11px] font-mono tracking-[0.12em]" style={{ color: PALETTE.accent }}>
            COMPLETE INTERVIEW LIFECYCLE
          </p>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            How InterviewAI prepares you for day one.
          </h2>
          <p className="text-sm max-w-md" style={{ color: PALETTE.textSecondary }}>
            Scroll to follow the path — the light tracks your progress through each stage.
          </p>
        </motion.div>

        <div
          className="relative w-full"
          style={{ aspectRatio: `${VIEWBOX_W} / ${viewBoxH}`, minHeight: isDesktop ? undefined : 1700 }}
        >
          <JourneyPath
            d={d}
            pathLength={pathLength}
            fractions={fractions}
            stages={stages}
            viewBoxH={viewBoxH}
            scrollProgress={activeProgress}
          />
          {stages.map((stage, i) => (
            <StageBlock
              key={stage.id}
              stage={stage}
              delay={i * 0.08}
              fillFraction={fractions[i]}
              scrollProgress={activeProgress}
              isDesktop={isDesktop}
              viewBoxH={viewBoxH}
            />
          ))}
        </div>
      </div>
    </section>
  );
};
