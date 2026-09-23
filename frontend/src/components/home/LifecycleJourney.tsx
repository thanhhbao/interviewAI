'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Mic, Code2, Users2, BarChart3, CheckCircle2 } from 'lucide-react';
import { PALETTE, fadeUp, viewportOnce, useSafeReducedMotion } from '@/lib/motion';

// Replaces the old 450vh sticky-scroll implementation entirely. This is a
// STATIC spatial journey: one predefined curved SVG path connecting five
// checkpoints, with a bright segment that travels the route on its own
// clock (CSS stroke-dashoffset animation) — completely independent of
// scroll position. The page scrolls past it normally.
//
// IMPORTANT (fixed bug): the travel animation's keyframe target must be a
// concrete pixel number, not `calc(var(--period))`. Browsers cannot reliably
// numerically interpolate a keyframe value that depends on an unregistered
// CSS custom property inside calc() — the value jumps straight from 0 to
// the final offset and then holds there, which is exactly what looked
// "stuck" instead of continuously travelling. The fix below injects the
// already-computed pixel number directly into the generated @keyframes
// rule, so there is nothing left for the browser to resolve at animation
// time — a plain number-to-number interpolation, which always animates.

interface Stage {
  id: string;
  index: string;
  label: string;
  icon: React.ElementType;
  description: string;
  point: [number, number]; // normalized 0-1000 units, matches the SVG viewBox
  side: 'left' | 'right';
}

const VIEWBOX_W = 1000;
const VIEWBOX_H = 1400;
const LOOP_DURATION = 11; // seconds — slow and deliberate, not "electricity"
const GLOW_PEAK = 0.9; // fraction of the loop where a checkpoint's highlight peaks

const desktopStages: Stage[] = [
  { id: 'resume', index: '01', label: 'Resume + JD Sync', icon: FileText, description: 'Role context is extracted before the interview begins.', point: [180, 90], side: 'right' },
  { id: 'interview', index: '02', label: 'Live AI Interview', icon: Mic, description: 'A natural voice conversation with adaptive follow-ups.', point: [800, 340], side: 'left' },
  { id: 'technical', index: '03', label: 'Technical Round', icon: Code2, description: 'Code and reasoning are evaluated together, live.', point: [210, 640], side: 'right' },
  { id: 'behavioral', index: '04', label: 'Behavioral Signals', icon: Users2, description: 'Ownership, clarity and impact are measured as you speak.', point: [780, 970], side: 'left' },
  { id: 'report', index: '05', label: 'Diagnostic Report', icon: BarChart3, description: 'Every signal converges into one actionable report.', point: [260, 1270], side: 'right' },
];

const mobileStages: Stage[] = desktopStages.map((s, i) => ({
  ...s,
  point: [i % 2 === 0 ? 130 : 170, 90 + i * 300],
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
  const svgNS = 'http://www.w3.org/2000/svg';
  const path = document.createElementNS(svgNS, 'path');
  path.setAttribute('d', d);
  return path.getTotalLength();
}

/** Delay (seconds) so a keyframe whose own peak sits at GLOW_PEAK (0-1 of
 * its cycle) lines up exactly with `fraction` of the shared loop. */
function delayFor(fraction: number): number {
  const shifted = ((fraction - GLOW_PEAK) % 1 + 1) % 1;
  return shifted * LOOP_DURATION;
}

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

function StageBlock({ stage, delay, glowDelay, animate }: { stage: Stage; delay: number; glowDelay: number; animate: boolean }) {
  const Icon = stage.icon;
  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={viewportOnce}
      transition={{ duration: 0.6, delay }}
      className="absolute w-[220px] sm:w-[300px]"
      style={{
        left: `${(stage.point[0] / VIEWBOX_W) * 100}%`,
        top: `${(stage.point[1] / VIEWBOX_H) * 100}%`,
        maxWidth: 'calc(100vw - 64px)',
        transform: stage.side === 'right' ? 'translate(24px, -50%)' : 'translate(calc(-100% - 24px), -50%)',
      }}
    >
      <p
        className="text-[11px] font-mono mb-1"
        style={animate ? { animation: `lifecycle-text-glow ${LOOP_DURATION}s ease-in-out infinite`, animationDelay: `${glowDelay}s` } : { color: PALETTE.accent }}
      >
        {stage.index}
      </p>
      <h3
        className="text-base font-medium flex items-center gap-2 mb-1.5"
        style={
          animate
            ? { color: PALETTE.textPrimary, animation: `lifecycle-text-glow-strong ${LOOP_DURATION}s ease-in-out infinite`, animationDelay: `${glowDelay}s` }
            : { color: PALETTE.textPrimary }
        }
      >
        <Icon className="w-4 h-4" />
        {stage.label}
      </h3>
      <p className="text-[13px] leading-relaxed mb-3" style={{ color: PALETTE.textSecondary }}>{stage.description}</p>
      <div
        className="rounded-[10px] p-3 transition-colors"
        style={{
          backgroundColor: PALETTE.surface,
          border: `1px solid ${PALETTE.border}`,
          ...(animate ? { animation: `lifecycle-card-glow ${LOOP_DURATION}s ease-in-out infinite`, animationDelay: `${glowDelay}s` } : {}),
        }}
      >
        <StageVisual id={stage.id} />
      </div>
    </motion.div>
  );
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
      const prefixD = buildPath(stages.slice(0, i + 1).map((s) => s.point));
      return measureLength(prefixD) / pathLength;
    });
  }, [stages, pathLength]);

  return { d, pathLength, fractions };
}

function JourneyPath({
  d,
  pathLength,
  fractions,
  stages,
  viewBoxH,
}: {
  d: string;
  pathLength: number;
  fractions: number[];
  stages: Stage[];
  viewBoxH: number;
}) {
  const prefersReducedMotion = useSafeReducedMotion();

  const brightLen = pathLength * 0.055;
  const period = brightLen + pathLength;
  const animate = pathLength > 0 && !prefersReducedMotion;

  return (
    <svg viewBox={`0 0 ${VIEWBOX_W} ${viewBoxH}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" aria-hidden="true">
      {animate && (
        <style>{`
          @keyframes lifecycle-travel { from { stroke-dashoffset: 0px; } to { stroke-dashoffset: -${period}px; } }
          @keyframes lifecycle-pulse {
            0%, 82%, 100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(120,255,231,0)); }
            90% { transform: scale(1.18); filter: drop-shadow(0 0 7px rgba(120,255,231,0.9)); }
            98% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(120,255,231,0)); }
          }
        `}</style>
      )}

      <path d={d} fill="none" stroke="rgba(160,220,210,0.12)" strokeWidth={2} />

      {animate && (
        <>
          {/* wide, very soft bloom */}
          <path
            d={d}
            fill="none"
            stroke={PALETTE.accent}
            strokeWidth={14}
            strokeLinecap="round"
            style={{ strokeDasharray: `${brightLen} ${pathLength}`, animation: `lifecycle-travel ${LOOP_DURATION}s linear infinite`, opacity: 0.15, filter: 'blur(7px)' }}
          />
          {/* tighter bloom */}
          <path
            d={d}
            fill="none"
            stroke={PALETTE.accent}
            strokeWidth={7}
            strokeLinecap="round"
            style={{ strokeDasharray: `${brightLen} ${pathLength}`, animation: `lifecycle-travel ${LOOP_DURATION}s linear infinite`, opacity: 0.3, filter: 'blur(3px)' }}
          />
          {/* sharp core */}
          <path
            d={d}
            fill="none"
            stroke={PALETTE.accentBright}
            strokeWidth={2.2}
            strokeLinecap="round"
            style={{ strokeDasharray: `${brightLen} ${pathLength}`, animation: `lifecycle-travel ${LOOP_DURATION}s linear infinite` }}
          />
        </>
      )}

      {stages.map((stage, i) => (
        <circle
          key={stage.id}
          cx={stage.point[0]}
          cy={stage.point[1]}
          r={7}
          fill={PALETTE.surface}
          stroke={PALETTE.accent}
          strokeWidth={2}
          style={
            animate
              ? {
                  transformOrigin: `${stage.point[0]}px ${stage.point[1]}px`,
                  animation: `lifecycle-pulse ${LOOP_DURATION}s ease-in-out infinite`,
                  animationDelay: `${delayFor(fractions[i])}s`,
                }
              : undefined
          }
        />
      ))}
    </svg>
  );
}

export const LifecycleJourney: React.FC = () => {
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
  const textAnimate = pathLength > 0 && !prefersReducedMotion;

  return (
    <section className="relative w-full py-24 md:py-32 overflow-hidden">
      {textAnimate && (
        <style>{`
          @keyframes lifecycle-text-glow {
            0%, 78%, 100% { color: ${PALETTE.accent}; opacity: 0.55; }
            90% { color: ${PALETTE.accentBright}; opacity: 1; }
            98% { color: ${PALETTE.accent}; opacity: 0.55; }
          }
          @keyframes lifecycle-text-glow-strong {
            0%, 78%, 100% { color: ${PALETTE.textPrimary}; text-shadow: 0 0 0 rgba(120,255,231,0); }
            90% { color: ${PALETTE.accentBright}; text-shadow: 0 0 14px rgba(120,255,231,0.45); }
            98% { color: ${PALETTE.textPrimary}; text-shadow: 0 0 0 rgba(120,255,231,0); }
          }
          @keyframes lifecycle-card-glow {
            0%, 78%, 100% { border-color: ${PALETTE.border}; box-shadow: 0 0 0 rgba(120,255,231,0); }
            90% { border-color: rgba(120,255,231,0.5); box-shadow: 0 0 24px rgba(85,229,207,0.15); }
            98% { border-color: ${PALETTE.border}; box-shadow: 0 0 0 rgba(120,255,231,0); }
          }
        `}</style>
      )}

      <div className="max-w-[1160px] mx-auto px-6">
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={viewportOnce} className="max-w-2xl mb-16 space-y-3">
          <p className="text-[11px] font-mono tracking-[0.12em]" style={{ color: PALETTE.accent }}>
            COMPLETE INTERVIEW LIFECYCLE
          </p>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            How InterviewAI prepares you for day one.
          </h2>
          <p className="text-sm max-w-md" style={{ color: PALETTE.textSecondary }}>
            One continuous session — scroll at your own pace, the route explains itself.
          </p>
        </motion.div>

        <div className="relative w-full" style={{ aspectRatio: `${VIEWBOX_W} / ${viewBoxH}`, minHeight: isDesktop ? undefined : 1700 }}>
          <JourneyPath d={d} pathLength={pathLength} fractions={fractions} stages={stages} viewBoxH={viewBoxH} />
          {stages.map((stage, i) => (
            <StageBlock
              key={stage.id}
              stage={isDesktop ? stage : { ...stage, side: 'right' }}
              delay={i * 0.08}
              glowDelay={delayFor(fractions[i])}
              animate={textAnimate}
            />
          ))}
        </div>
      </div>
    </section>
  );
};
