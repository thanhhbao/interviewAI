'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowRight, LucideIcon } from 'lucide-react';
import { PALETTE } from '@/lib/motion';

interface WorkflowGuardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  ctaLabel: string;
  ctaHref: string;
}

// Shared empty/blocked state for pages that require workflow context the
// user hasn't reached yet (e.g. /setup without a resume, or /report without
// a completed session). Uses the same dark PALETTE tokens as the rest of
// the application pages (/resume-job, /setup) rather than a light card.
export const WorkflowGuard: React.FC<WorkflowGuardProps> = ({
  icon: Icon,
  title,
  description,
  ctaLabel,
  ctaHref,
}) => {
  return (
    <div className="relative flex-1 flex items-center justify-center px-8 py-16">
      <div
        className="w-full max-w-sm text-center space-y-4 p-8 rounded-[16px]"
        style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
      >
        <div
          className="w-11 h-11 rounded-[10px] flex items-center justify-center mx-auto"
          style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textSecondary }}
        >
          <Icon className="w-5 h-5" />
        </div>
        <div className="space-y-1.5">
          <h2 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>{title}</h2>
          <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>{description}</p>
        </div>
        <Link
          href={ctaHref}
          className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-[10px] text-xs font-semibold transition-opacity hover:opacity-90"
          style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
        >
          <span>{ctaLabel}</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    </div>
  );
};
