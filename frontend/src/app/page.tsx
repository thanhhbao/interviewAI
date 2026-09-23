'use client';

import React from 'react';
import { motion, useScroll, useSpring } from 'framer-motion';
import { HeroEnergyStage } from '@/components/home/HeroEnergyStage';
import { CompanyMarquee } from '@/components/home/CompanyMarquee';
import { IntelligenceConsole } from '@/components/home/IntelligenceConsole';
import { ScenarioCarousel } from '@/components/home/ScenarioCarousel';
import { LifecycleJourney } from '@/components/home/LifecycleJourney';
import { CandidatePortalWindow } from '@/components/home/CandidatePortalWindow';
import { InteractiveFaq } from '@/components/home/InteractiveFaq';
import { FinalCtaSection } from '@/components/home/FinalCtaSection';
import { SiteFooter } from '@/components/home/SiteFooter';
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
        className="fixed top-0 left-0 right-0 h-[2px] bg-[#55E5CF] origin-left z-50 pointer-events-none"
        style={{ scaleX }}
      />

      {/* HERO — untouched */}
      <HeroEnergyStage />

      {/* ============================ ACT I — UNDERSTAND ============================ */}
      <ActShell background={<DotGrid className="w-full h-full" />}>
        <CompanyMarquee />
        <div id="explore">
          <IntelligenceConsole />
        </div>
      </ActShell>

      {/* ============================ ACT II — SIMULATE ============================= */}
      <ActShell background={<KineticMetalFlow className="w-full h-full" />}>
        <ScenarioCarousel />
        <LifecycleJourney />
      </ActShell>

      {/* ============================ ACT III — IMPROVE ============================= */}
      <ActShell background={<StarsDataField className="w-full h-full" />}>
        <CandidatePortalWindow />
        <InteractiveFaq />
      </ActShell>

      <FinalCtaSection />

      <SiteFooter />
    </div>
  );
}
