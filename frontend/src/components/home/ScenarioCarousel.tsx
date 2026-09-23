'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { PALETTE, fadeUp, viewportOnce, useSafeReducedMotion } from '@/lib/motion';
import { ChapterMarker } from '@/components/home/acts/ChapterMarker';
import { Grain } from '@/components/home/scenario/Grain';

// Master-detail Grainy-Carousel-style section: the carousel is a real
// photographic image strip (drag/swipe/snap via native scroll-snap for
// proven momentum + touch behavior), and a separate detail panel reads the
// active scenario off one shared `activeIndex`. Slides carry almost no
// text — only a number + category tag — everything else lives in the panel.

interface Scenario {
  id: string;
  category: string;
  company: string;
  level: string;
  duration: string;
  title: string;
  description: string;
  skills: string[];
  image: string;
  objectPosition: string;
}

const scenarios: Scenario[] = [
  {
    id: '1',
    category: 'System Design',
    company: 'Meta · Infrastructure',
    level: 'Senior / Staff',
    duration: '45 min',
    title: 'Design a Distributed Feed Cache at 500M DAU',
    description:
      'How would you architect cache invalidation, thundering herd mitigation, and multi-region replication for live news feed posts?',
    skills: ['Redis Cluster', 'Consistent Hashing', 'CDC Stream'],
    image: '/images/scenarios/system-design.webp',
    objectPosition: '35% 40%',
  },
  {
    id: '2',
    category: 'Algorithms & Code',
    company: 'Google · Cloud Engine',
    level: 'Hard · O(V+E)',
    duration: '35 min',
    title: 'Alien Dictionary Topological Sort with Cycles',
    description:
      'Given a sorted list of words in an alien language, reconstruct the exact lexicographical ordering or detect impossible cycles.',
    skills: ['DFS / Kahn Algo', 'Cycle Detection', 'DAG Traversal'],
    image: '/images/scenarios/algorithms.webp',
    objectPosition: '50% 65%',
  },
  {
    id: '3',
    category: 'Behavioral & Leadership',
    company: 'Amazon · AWS Principal',
    level: 'Bar Raiser LP',
    duration: '30 min',
    title: 'Navigating Disagree and Commit on Tech Debt',
    description:
      'Describe a time you strongly disagreed with a tech roadmap prioritization by a VP. How did you present data and commit?',
    skills: ['STAR Method', 'Executive Influence', 'Ownership'],
    image: '/images/scenarios/behavioral.webp',
    objectPosition: '50% 35%',
  },
  {
    id: '4',
    category: 'Offer & Compensation',
    company: 'Stripe · Payments Platform',
    level: 'Negotiation Room',
    duration: '25 min',
    title: 'Counter-Offering Multi-Year Equity Tranches',
    description:
      'Simulate a live phone negotiation with a senior recruiter to balance base salary vs quarterly performance RSU refreshers.',
    skills: ['Levelling Calibration', 'Counter Strategy', 'Leverage Points'],
    image: '/images/scenarios/negotiation.webp',
    objectPosition: '50% 30%',
  },
];

const CARD_WIDTH = 460;
const CARD_HEIGHT = 575; // 4:5
const CARD_GAP = 20;
const STEP = CARD_WIDTH + CARD_GAP;

function ScenarioSlide({
  scenario,
  index,
  isActive,
  prefersReducedMotion,
}: {
  scenario: Scenario;
  index: number;
  isActive: boolean;
  prefersReducedMotion: boolean;
}) {
  return (
    <div
      className="relative h-full w-full rounded-[14px] overflow-hidden select-none"
      style={{ backgroundColor: PALETTE.surface }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={scenario.image}
        alt={`${scenario.category} interview scenario`}
        className="absolute inset-0 h-full w-full object-cover pointer-events-none"
        style={{ objectPosition: scenario.objectPosition }}
        draggable={false}
        loading={index === 0 ? 'eager' : 'lazy'}
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: isActive
            ? 'linear-gradient(180deg, rgba(5,8,7,0.05) 0%, rgba(5,8,7,0.0) 40%, rgba(5,8,7,0.55) 100%)'
            : 'rgba(5,8,7,0.4)',
        }}
      />
      <Grain opacity={0.1} />
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between px-5 pt-5">
        <span className="text-[13px] font-mono tabular-nums" style={{ color: PALETTE.accentBright }}>
          {String(index + 1).padStart(2, '0')}
        </span>
        <span
          className="text-[10px] font-mono tracking-[0.14em] uppercase"
          style={{ color: 'rgba(236,237,233,0.7)' }}
        >
          {scenario.category}
        </span>
      </div>
      {!prefersReducedMotion && (
        <div
          className="pointer-events-none absolute inset-0 rounded-[14px]"
          style={{ boxShadow: isActive ? `inset 0 0 0 1px rgba(160,220,210,0.22)` : `inset 0 0 0 1px rgba(255,255,255,0.05)` }}
        />
      )}
    </div>
  );
}

function DetailPanel({ scenario, index }: { scenario: Scenario; index: number }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={scenario.id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <p className="text-[12px] font-mono tabular-nums" style={{ color: PALETTE.textSecondary }}>
          {String(index + 1).padStart(2, '0')} / {String(scenarios.length).padStart(2, '0')}
        </p>
        <p
          className="mt-4 text-[12px] font-mono tracking-[0.12em] uppercase"
          style={{ color: PALETTE.accent }}
        >
          {scenario.category}
        </p>
        <h3
          className="mt-3 text-2xl md:text-[28px] font-semibold tracking-tight leading-[1.15]"
          style={{ color: PALETTE.textPrimary }}
        >
          {scenario.title}
        </h3>
        <p className="mt-4 text-[13px]" style={{ color: PALETTE.textSecondary }}>
          {scenario.company}
        </p>
        <p className="mt-1 text-[13px]" style={{ color: PALETTE.textSecondary }}>
          {scenario.level} · {scenario.duration}
        </p>
        <p className="mt-5 text-[14px] leading-relaxed max-w-md" style={{ color: '#c7cac5' }}>
          {scenario.description}
        </p>
        <div className="mt-6 flex flex-wrap gap-x-4 gap-y-1.5">
          {scenario.skills.map((skill, i) => (
            <span key={skill} className="text-[12px]" style={{ color: '#9aa19c' }}>
              {i > 0 && <span className="mr-4" style={{ color: PALETTE.border }}>·</span>}
              {skill}
            </span>
          ))}
        </div>
        <a
          href="#"
          className="mt-8 inline-flex items-center gap-1.5 text-[13px] font-medium group"
          style={{ color: PALETTE.accentBright }}
        >
          Practice this scenario
          <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
        </a>
      </motion.div>
    </AnimatePresence>
  );
}

export const ScenarioCarousel: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const prefersReducedMotion = useSafeReducedMotion();
  const [activeIndex, setActiveIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [spacer, setSpacer] = useState(0);
  const rafRef = useRef<number | null>(null);
  const settleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function measure() {
      const el = containerRef.current;
      if (!el) return;
      setSpacer(Math.max((el.clientWidth - CARD_WIDTH) / 2, 16));
    }
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  const handleScroll = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const el = containerRef.current;
      if (!el) return;
      const p = el.scrollLeft / STEP;
      setProgress(p);

      // Detail panel updates only once the scroll gesture has settled near a
      // slide (snap complete), not continuously during the drag.
      if (settleTimer.current) clearTimeout(settleTimer.current);
      settleTimer.current = setTimeout(() => {
        setActiveIndex(Math.round(p));
      }, 90);
    });
  }, []);

  const scrollToIndex = (index: number) => {
    const clamped = Math.max(0, Math.min(index, scenarios.length - 1));
    containerRef.current?.scrollTo({ left: clamped * STEP, behavior: 'smooth' });
    setActiveIndex(clamped);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowRight') scrollToIndex(activeIndex + 1);
    if (e.key === 'ArrowLeft') scrollToIndex(activeIndex - 1);
  };

  const activeScenario = useMemo(() => scenarios[activeIndex], [activeIndex]);

  return (
    <section className="relative w-full py-24 md:py-32 overflow-hidden">
      <div className="max-w-[1240px] mx-auto px-6 mb-12 md:mb-16">
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={viewportOnce} className="max-w-xl">
          <ChapterMarker number="02" label="SIMULATE" />
          <p className="text-[11px] font-mono tracking-[0.16em] uppercase mt-4" style={{ color: PALETTE.textSecondary }}>
            Real-world interview scenarios
          </p>
          <h2 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            Practice the questions they actually ask.
          </h2>
        </motion.div>
      </div>

      <div className="max-w-[1240px] mx-auto px-6">
        <motion.div
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={viewportOnce}
          className="flex flex-col lg:flex-row lg:items-center gap-10 lg:gap-4"
        >
          {/* Carousel — ~60-65% on desktop */}
          <div className="lg:flex-[0_0_62%] min-w-0">
            <div
              ref={containerRef}
              onScroll={handleScroll}
              onKeyDown={handleKeyDown}
              tabIndex={0}
              role="listbox"
              aria-label="Interview scenario carousel"
              className="flex overflow-x-auto scrollbar-none snap-x snap-mandatory outline-none"
              style={{ gap: CARD_GAP, scrollSnapType: 'x mandatory' }}
            >
              <div style={{ flex: `0 0 ${spacer}px` }} aria-hidden="true" />
              {scenarios.map((scenario, index) => {
                const offset = progress - index;
                const abs = Math.min(Math.abs(offset), 1.6);
                const isActive = activeIndex === index;
                const scale = prefersReducedMotion ? 1 : 1 - abs * 0.09;
                const opacity = prefersReducedMotion ? 1 : Math.max(0.5, 1 - abs * 0.5);
                const brightness = prefersReducedMotion ? 1 : Math.max(0.68, 1 - abs * 0.32);
                return (
                  <div
                    key={scenario.id}
                    role="option"
                    aria-selected={isActive}
                    className="shrink-0 snap-center"
                    style={{
                      width: CARD_WIDTH,
                      height: CARD_HEIGHT,
                      transform: `scale(${scale})`,
                      opacity,
                      filter: `brightness(${brightness})`,
                      zIndex: 100 - Math.round(abs * 10),
                      transition: 'transform 0.05s linear, filter 0.05s linear',
                    }}
                  >
                    <ScenarioSlide
                      scenario={scenario}
                      index={index}
                      isActive={isActive}
                      prefersReducedMotion={prefersReducedMotion}
                    />
                  </div>
                );
              })}
              <div style={{ flex: `0 0 ${spacer}px` }} aria-hidden="true" />
            </div>

            {/* Minimal nav: numbered dashes, no bulky controls */}
            <div className="flex items-center gap-2 mt-6 px-1">
              {scenarios.map((scenario, index) => (
                <button
                  key={scenario.id}
                  onClick={() => scrollToIndex(index)}
                  aria-label={`Go to ${scenario.category} scenario`}
                  aria-current={activeIndex === index}
                  className="h-[2px] rounded-full transition-all duration-300"
                  style={{
                    width: activeIndex === index ? 28 : 14,
                    backgroundColor: activeIndex === index ? PALETTE.accentBright : PALETTE.border,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Detail panel — ~35-40%, no bordered card, editorial */}
          <div className="lg:flex-1 lg:pl-6 min-w-0">
            <DetailPanel scenario={activeScenario} index={activeIndex} />
          </div>
        </motion.div>
      </div>
    </section>
  );
};
