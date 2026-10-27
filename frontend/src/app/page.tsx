'use client';

import React from 'react';
import { motion, useScroll, useSpring } from 'framer-motion';
import { InterviewTerminalHero } from '@/components/home/InterviewTerminalHero';
import { CompanyMarquee } from '@/components/home/CompanyMarquee';
import { IndustryVoicesTicker } from '@/components/home/IndustryVoicesTicker';
import { IntelligenceConsole } from '@/components/home/IntelligenceConsole';
import { ScenarioCarousel } from '@/components/home/ScenarioCarousel';
import { LifecycleJourney } from '@/components/home/LifecycleJourney';
import { CandidatePortalWindow } from '@/components/home/CandidatePortalWindow';
import { InteractiveFaq } from '@/components/home/InteractiveFaq';
import { FinalCtaSection } from '@/components/home/FinalCtaSection';
import { DandelionFooter } from '@/components/home/footer/DandelionFooter';
import { ActShell } from '@/components/home/acts/ActShell';
import { DotGrid } from '@/components/home/acts/DotGrid';
import { KineticMetalFlow } from '@/components/home/acts/KineticMetalFlow';
import { StarsDataField } from '@/components/home/acts/StarsDataField';

// ---------------------------------------------------------------------------
// PAGE STRUCTURE — one story, three acts. Hero is untouched; everything
// after it is grouped into three ActShell wrappers, each hosting one
// continuous, AUTONOMOUS background animation (not scroll-linked — every
// effect runs on its own clock and only pauses via IntersectionObserver
// when its act is off-screen):
//
//   ACT I — UNDERSTAND   (DotGrid)          Social proof -> Interview Intelligence
//   ACT II — SIMULATE    (KineticMetalFlow) Real-world scenarios -> Lifecycle
//   ACT III — IMPROVE    (StarsDataField)   Candidate Portal -> FAQ
//                                           (Final CTA + Footer keep their
//                                            own atmosphere)
//
// Section content itself uses ordinary whileInView entrance animation, not
// scroll-position-driven transforms — scrolling is plain native browser
// scrolling throughout.
// ---------------------------------------------------------------------------

export default function HomePage() {
  // Scroll Progress Bar (Framer hallmark)
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  return (
    <div className="relative flex-1 flex flex-col min-h-screen bg-[#07080a] text-neutral-100 selection:bg-teal-500/30 selection:text-teal-200">
      {/* TOP SCROLL PROGRESS BAR — the one intentional scroll-linked visual
          left on the page: a plain reading-progress indicator, not a
          content/background animation. */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-[2px] bg-[#E7E8E9] origin-left z-50 pointer-events-none"
        style={{ scaleX }}
      />

      {/* HERO — terminal computer design */}
      <InterviewTerminalHero />

      {/* ============================ ACT I — UNDERSTAND ============================ */}
      <ActShell background={<DotGrid className="w-full h-full" />}>
        <CompanyMarquee />
        <div id="explore">
          <IntelligenceConsole />
        </div>
      </ActShell>

      {/* Editorial/social-proof strip between Act I and Act II — its own
          atmosphere (no ActShell background), same as FinalCtaSection. */}
      <IndustryVoicesTicker />

      {/* ============================ ACT II — SIMULATE ============================= */}
      <ActShell background={<KineticMetalFlow className="w-full h-full" />}>
        <ScenarioCarousel />
        <LifecycleJourney />
      </ActShell>

      {/* ============================ ENDING SEQUENCE ================================
          Act III (IMPROVE) + Final CTA + Footer share ONE StarsDataField canvas so
          stars travel in a single continuous direction with no restart or seam at
          section boundaries. The canvas is absolute-inset-0 inside a relative
          wrapper whose height is the combined flow height of all three children.   */}
      <div className="relative">
        {/* Single shared star background. pointer-events-none + aria-hidden so
            it never intercepts clicks or shows up in accessibility trees.         */}
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
          <StarsDataField className="w-full h-full" />
        </div>

        {/* Top fade — dissolves the star field into Act II's background colour
            so consecutive acts blend instead of cutting hard.                     */}
        <div
          className="pointer-events-none absolute inset-x-0 top-0 h-32"
          aria-hidden="true"
          style={{ background: 'linear-gradient(to bottom, #060A0A, transparent)' }}
        />

        {/* ACT III — IMPROVE: no ActShell (avoids duplicate canvas + solid bg).
            Content sits in a relative div so it stacks above the absolute canvas. */}
        <div className="relative">
          <CandidatePortalWindow />
          <InteractiveFaq />
        </div>

        {/* FinalCtaSection background is transparent — shared stars show through */}
        <FinalCtaSection />

        {/* DandelionFooter has its own opaque gradient, naturally covers the stars */}
        <DandelionFooter />
      </div>
    </div>
  );
}
