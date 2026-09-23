'use client';

import React from 'react';
import { useSafeReducedMotion } from '@/lib/motion';

// Static-image + transparent-WebM-overlay architecture (replaces the earlier
// WebGL depth-scan shader in HeroDepthScan.tsx — kept in the tree but no
// longer mounted). The scan light is pre-rendered offline by
// scripts/generate_scan_overlay.py from the exact same graded still
// (hero-dark.webp) so the two layers are pixel-aligned; nothing here is
// computed at runtime.
interface HeroScanOverlayProps {
  imageSrc: string;
  videoSrc: string;
  alt: string;
  className?: string;
}

const LAYER_STYLE: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  objectPosition: 'center',
};

export const HeroScanOverlay: React.FC<HeroScanOverlayProps> = ({ imageSrc, videoSrc, alt, className }) => {
  // Deterministic across server/client hydration (see useSafeReducedMotion):
  // the server always assumes no preference, so mounting/unmounting the
  // <video> from the raw (synchronous, client-only) matchMedia result caused
  // "did not expect server HTML to contain a <video>" hydration errors under
  // prefers-reduced-motion. The video still ends up hidden for those users —
  // just one tick after hydration instead of on the very first paint.
  const prefersReducedMotion = useSafeReducedMotion();

  return (
    <div className={className} style={{ position: 'relative', overflow: 'hidden' }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageSrc} alt={alt} style={LAYER_STYLE} />
      {!prefersReducedMotion && (
        <video
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
          style={{ ...LAYER_STYLE, pointerEvents: 'none' }}
        >
          <source src={videoSrc} type="video/webm" />
        </video>
      )}
    </div>
  );
};
