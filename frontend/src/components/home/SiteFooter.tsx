'use client';

import React from 'react';
import Link from 'next/link';
import { PALETTE } from '@/lib/motion';

const productLinks = [
  { label: 'Questions', href: '/setup' },
  { label: 'Interview Room', href: '/interview' },
  { label: 'Resume Match', href: '/resume-job' },
  { label: 'Reports', href: '/report' },
];

export const SiteFooter: React.FC = () => {
  return (
    <footer className="relative w-full py-10" style={{ backgroundColor: '#050706', borderTop: `1px solid ${PALETTE.border}` }}>
      <div className="max-w-[1160px] mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6 text-xs">
        <div className="flex items-center gap-3" style={{ color: PALETTE.textSecondary }}>
          <span className="font-medium" style={{ color: PALETTE.textPrimary }}>InterviewPal</span>
          <span>Practice interviews that feel real.</span>
        </div>

        <nav className="flex items-center gap-5">
          {productLinks.map((l) => (
            <Link key={l.label} href={l.href} className="transition-colors hover:text-white" style={{ color: PALETTE.textSecondary }}>
              {l.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-4" style={{ color: PALETTE.textSecondary }}>
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accent }} />
            All systems operational
          </span>
          <span>© 2026 InterviewAI</span>
        </div>
      </div>
    </footer>
  );
};
