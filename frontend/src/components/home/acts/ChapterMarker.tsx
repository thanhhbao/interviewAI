'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { PALETTE, fadeUpSmall, viewportOnce } from '@/lib/motion';

// Editorial chapter label, not a nav tab — appears once at the opening of
// each act to carry the three-part narrative (Understand / Simulate /
// Improve) without adding visual weight.
export function ChapterMarker({ number, label }: { number: string; label: string }) {
  return (
    <motion.p
      variants={fadeUpSmall}
      initial="hidden"
      whileInView="show"
      viewport={viewportOnce}
      className="text-[11px] font-mono tracking-[0.2em] mb-3"
      style={{ color: PALETTE.textSecondary }}
    >
      <span style={{ color: PALETTE.accent }}>{number}</span> / {label}
    </motion.p>
  );
}
