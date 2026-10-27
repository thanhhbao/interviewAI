'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { PALETTE, viewportOnce } from '@/lib/motion';

interface MatchScoreBlockProps {
  score: number;
}

// Resume–Job match score hierarchy (REF-D inspired), built directly on
// Home's own readiness-bar pattern from CandidatePortalWindow: large
// numeral + short label + thin animated progress fill. No circular gauge,
// no chart — a single honest number plus the annotation that this is
// alignment, not a hiring probability.
export function MatchScoreBlock({ score }: MatchScoreBlockProps) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline gap-2.5">
        <span className="text-3xl font-semibold tabular-nums" style={{ color: PALETTE.accentBright }}>
          {score}%
        </span>
        <span className="text-xs font-medium" style={{ color: PALETTE.textSecondary }}>
          Estimated Resume–Job Match
        </span>
      </div>
      <div className="h-1 rounded-full overflow-hidden max-w-xs" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: PALETTE.accent }}
          initial={{ width: 0 }}
          whileInView={{ width: `${score}%` }}
          viewport={viewportOnce}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <p className="text-[11px] max-w-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
        Alignment between your provided resume and target job description — not a hiring probability.
      </p>
    </div>
  );
}
