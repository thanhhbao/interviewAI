'use client';

import React from 'react';
import Link from 'next/link';
import { Twitter, Linkedin, Github } from 'lucide-react';
import { PALETTE } from '@/lib/motion';

const FOOTER_COLS = [
  {
    title: 'Product',
    links: [
      { label: 'Interview Room',  href: '/interview'  },
      { label: 'Resume & Job',    href: '/resume-job' },
      { label: 'Reports',         href: '/report'     },
      { label: 'History',         href: '/history'    },
      { label: 'Settings',        href: '/settings'   },
    ],
  },
  {
    title: 'Practice',
    links: [
      { label: 'Quick Warm-up',       href: '/setup' },
      { label: 'Behavioral',          href: '/setup' },
      { label: 'Technical Design',    href: '/setup' },
      { label: 'Coding Interview',    href: '/setup' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { label: 'Help Center',     href: '#' },
      { label: 'Interview Tips',  href: '#' },
      { label: 'Resume Guide',    href: '#' },
      { label: 'FAQ',             href: '#' },
    ],
  },
  {
    title: 'Legal',
    links: [
      { label: 'Privacy Policy',   href: '#' },
      { label: 'Terms of Service', href: '#' },
      { label: 'Cookie Policy',    href: '#' },
    ],
  },
];

export function AppFooter() {
  return (
    <footer
      className="relative mt-auto px-6 py-10"
      style={{ borderTop: `1px solid ${PALETTE.border}` }}
    >
      <div className="max-w-5xl mx-auto space-y-8">

        {/* 4-column grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {FOOTER_COLS.map((col) => (
            <div key={col.title} className="space-y-3">
              <p
                className="text-[10px] font-semibold uppercase tracking-[0.14em]"
                style={{ color: PALETTE.textSecondary }}
              >
                {col.title}
              </p>
              <ul className="space-y-2">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <Link
                      href={l.href}
                      className="text-[12px] transition-colors hover:text-white"
                      style={{ color: 'rgba(133,142,138,0.75)' }}
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div
          className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-6 text-[11px]"
          style={{ borderTop: `1px solid ${PALETTE.border}`, color: 'rgba(133,142,138,0.6)' }}
        >
          <div className="flex items-center gap-4">
            <span className="font-mono tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
              InterviewPal
            </span>
            <span>© 2026. All rights reserved.</span>
            <span className="hidden sm:flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accent }} />
              All systems operational
            </span>
          </div>
          <div className="flex items-center gap-3">
            <a href="#" aria-label="Twitter / X" className="transition-colors hover:text-white">
              <Twitter className="w-3.5 h-3.5" />
            </a>
            <a href="#" aria-label="LinkedIn" className="transition-colors hover:text-white">
              <Linkedin className="w-3.5 h-3.5" />
            </a>
            <a href="#" aria-label="GitHub" className="transition-colors hover:text-white">
              <Github className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>

      </div>
    </footer>
  );
}
