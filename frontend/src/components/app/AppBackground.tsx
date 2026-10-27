'use client';

import React from 'react';
import { PALETTE } from '@/lib/motion';
import { DotGrid } from '@/components/home/acts/DotGrid';

// Shared background for application pages (/resume-job, /setup, /report,
// /history, /settings) — the same DotGrid canvas Home uses (ACT I), so the
// dot field/cursor language carries through from the landing experience
// into the product itself instead of switching to an unrelated pattern.
// `fixed inset-0` means this is exactly the viewport, not the scrollable
// page, so the dots stay visible and consistent as the user scrolls,
// showing through the gaps around/between panels. Plus the same soft teal
// glow used on FinalCtaSection.
export function AppBackground() {
  return (
    <div className="fixed inset-0 -z-10 pointer-events-none" style={{ backgroundColor: PALETTE.bg }}>
      <DotGrid className="absolute inset-0 w-full h-full" idleIntensity={2.2} />
      <div
        className="absolute left-1/2 top-0 -translate-x-1/2 w-[900px] h-[480px]"
        style={{
          background: `radial-gradient(circle, ${PALETTE.deepTeal}29, transparent 70%)`,
        }}
      />
    </div>
  );
}
