'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useAnimationFrame } from 'framer-motion';
import { PALETTE, useSafeReducedMotion } from '@/lib/motion';
import { industryVoices } from '@/lib/industryVoicesData';
import type { IndustryVoice } from '@/lib/types/industryVoice';

// Inspired by the shape of the Framer "Wavy Ticker" component (a
// horizontal loop where items drift along a sine path) — not a copy of its
// visuals. Everything here reuses Home's own PALETTE/surface system, and
// the motion is hand-rolled: a single rAF loop drives both the horizontal
// scroll and each card's vertical bob, rather than the CSS-keyframe
// marquee CompanyMarquee uses elsewhere, because hover needs to *ease* the
// speed down (not just pause it) and the wave needs a distinct phase per
// card — neither fits a plain CSS animation.
const SPEED_PX_S = 34;
const HOVER_SPEED_PX_S = 10;
const SPEED_LERP_RATE = 3; // higher = snappier speed transition on hover
const WAVE_AMPLITUDE_PX = 22; // within the requested 18-30px range
const WAVE_PERIOD_S = 7;
const WAVE_OMEGA = (Math.PI * 2) / WAVE_PERIOD_S;

export const IndustryVoicesTicker: React.FC = () => {
  const prefersReducedMotion = useSafeReducedMotion();
  const items = industryVoices;

  const trackRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Array<HTMLDivElement | null>>([]);
  const halfWidthRef = useRef(0);
  const xRef = useRef(0);
  const speedRef = useRef(SPEED_PX_S);
  const hoverRef = useRef(false);
  const elapsedRef = useRef(0);

  // The track holds two back-to-back copies of the same sequence with one
  // consistent gap throughout, so scrolling exactly half its total width
  // lands on a pixel-identical frame — the loop wrap is invisible.
  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;
    const measure = () => {
      halfWidthRef.current = track.scrollWidth / 2;
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(track);
    return () => ro.disconnect();
  }, []);

  useAnimationFrame((_, delta) => {
    if (prefersReducedMotion) return;
    const dt = Math.min(0.05, delta / 1000);
    elapsedRef.current += dt;

    const targetSpeed = hoverRef.current ? HOVER_SPEED_PX_S : SPEED_PX_S;
    speedRef.current += (targetSpeed - speedRef.current) * Math.min(1, dt * SPEED_LERP_RATE);
    xRef.current -= speedRef.current * dt;

    const half = halfWidthRef.current;
    if (half > 0) {
      while (xRef.current <= -half) xRef.current += half;
    }
    if (trackRef.current) {
      trackRef.current.style.transform = `translate3d(${xRef.current}px, 0, 0)`;
    }

    const phaseStep = (Math.PI * 2) / items.length;
    cardRefs.current.forEach((el, i) => {
      if (!el) return;
      const phase = (i % items.length) * phaseStep;
      const y = Math.sin(elapsedRef.current * WAVE_OMEGA + phase) * WAVE_AMPLITUDE_PX;
      el.style.transform = `translate3d(0, ${y.toFixed(2)}px, 0)`;
    });
  });

  const doubled = [...items, ...items];

  return (
    <section className="relative w-full py-24 md:py-32 overflow-hidden">
      <div className="max-w-[1240px] mx-auto px-6 mb-14 md:mb-16">
        <p
          className="text-[11px] font-mono tracking-[0.2em] uppercase mb-3"
          style={{ color: PALETTE.textSecondary }}
        >
          <span style={{ color: PALETTE.accent }}>◆</span> INDUSTRY VOICES
        </p>
        <h2
          className="text-3xl md:text-4xl font-semibold tracking-tight max-w-xl"
          style={{ color: PALETTE.textPrimary }}
        >
          Voices shaping modern work.
        </h2>
        <p className="mt-3 text-[13px] leading-relaxed max-w-md" style={{ color: PALETTE.textSecondary }}>
          Public statements from technology leaders on hiring, interviewing, and how work is
          changing — sourced and linked, never rewritten into marketing copy.
        </p>
      </div>

      <div
        className="relative w-full"
        onMouseEnter={() => {
          hoverRef.current = true;
        }}
        onMouseLeave={() => {
          hoverRef.current = false;
        }}
      >
        <div
          className="pointer-events-none absolute left-0 top-0 bottom-0 w-16 md:w-32 z-10"
          style={{ background: `linear-gradient(to right, ${PALETTE.bg}, transparent)` }}
        />
        <div
          className="pointer-events-none absolute right-0 top-0 bottom-0 w-16 md:w-32 z-10"
          style={{ background: `linear-gradient(to left, ${PALETTE.bg}, transparent)` }}
        />

        {prefersReducedMotion ? (
          <div
            className="flex gap-6 overflow-x-auto px-6 md:px-16 pb-2"
            role="list"
            aria-label="Industry voices"
          >
            {items.map((voice) => (
              <VoiceCard key={voice.id} voice={voice} />
            ))}
          </div>
        ) : (
          <div className="overflow-hidden" role="list" aria-label="Industry voices">
            <div ref={trackRef} className="flex gap-6 w-max py-6" style={{ willChange: 'transform' }}>
              {doubled.map((voice, i) => (
                <VoiceCard
                  key={`${voice.id}-${i}`}
                  voice={voice}
                  ariaHidden={i >= items.length}
                  ref={(el) => {
                    cardRefs.current[i] = el;
                  }}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

const VoiceCard = React.forwardRef<HTMLDivElement, { voice: IndustryVoice; ariaHidden?: boolean }>(
  ({ voice, ariaHidden }, ref) => {
    return (
      <div
        ref={ref}
        role="listitem"
        aria-hidden={ariaHidden || undefined}
        className="shrink-0 w-[320px]"
        style={{ willChange: 'transform' }}
      >
        <div
          className="h-full flex flex-col rounded-[14px] p-6 transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:shadow-[0_24px_48px_-28px_rgba(0,0,0,0.65)]"
          style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
        >
          <div className="flex items-center gap-2.5 mb-5">
            <CompanyLogoBadge voice={voice} />
            <span className="text-[13px] font-medium" style={{ color: PALETTE.textPrimary }}>
              {voice.company}
            </span>
          </div>

          <span
            className="self-start mb-2 text-[9px] font-mono tracking-[0.14em] uppercase px-1.5 py-0.5 rounded-[4px]"
            style={{ color: PALETTE.textSecondary, border: `1px solid ${PALETTE.border}` }}
          >
            {voice.contentType === 'quote' ? 'Direct quote' : 'Public remarks · Summary'}
          </span>

          <p className="flex-1 text-[13.5px] leading-relaxed" style={{ color: PALETTE.textPrimary }}>
            {voice.contentType === 'quote' ? `“${voice.text}”` : voice.text}
          </p>

          <div className="mt-5 pt-4" style={{ borderTop: `1px solid ${PALETTE.border}` }}>
            <p className="text-[13px] font-medium" style={{ color: PALETTE.textPrimary }}>
              {voice.person}
            </p>
            <p className="mt-0.5 text-[12px]" style={{ color: PALETTE.textSecondary }}>
              {voice.role}
            </p>

            {voice.sourceUrl ? (
              <a
                href={voice.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 inline-flex items-center gap-1 text-[12px] font-medium transition-opacity hover:opacity-70"
                style={{ color: PALETTE.accent }}
              >
                {voice.sourceLabel} ↗
              </a>
            ) : (
              <span
                className="mt-3 inline-flex items-center gap-1 text-[12px]"
                style={{ color: PALETTE.textSecondary, opacity: 0.6 }}
              >
                {voice.sourceLabel}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }
);
VoiceCard.displayName = 'VoiceCard';

// `companyLogo` is expected to be a local asset path (e.g.
// "/companies/nvidia.svg"); none of those files exist in the repo yet, so
// this renders the image optimistically and swaps to a monochrome
// text-initial badge on load failure — never a broken-image icon, and no
// build-time dependency on which logos happen to be present.
function CompanyLogoBadge({ voice }: { voice: IndustryVoice }) {
  const isImagePath = voice.companyLogo.startsWith('/');
  const [imageFailed, setImageFailed] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // A 404 on a same-origin asset can resolve (and fire the native `error`
  // event) while the browser is still parsing the server-rendered HTML —
  // before hydration has attached React's listener, so `onError` alone
  // misses it. This catches that already-failed state once mounted, in
  // addition to `onError` catching failures that happen afterward.
  useEffect(() => {
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth === 0) {
      setImageFailed(true);
    }
  }, []);

  if (isImagePath && !imageFailed) {
    return (
      <span
        className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 overflow-hidden"
        style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={voice.companyLogo}
          alt=""
          className="w-4 h-4 object-contain"
          onError={() => setImageFailed(true)}
        />
      </span>
    );
  }

  return (
    <span
      className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-mono shrink-0"
      style={{
        backgroundColor: PALETTE.surface2,
        border: `1px solid ${PALETTE.border}`,
        color: PALETTE.textPrimary,
      }}
      aria-hidden="true"
    >
      {voice.company.charAt(0)}
    </span>
  );
}
