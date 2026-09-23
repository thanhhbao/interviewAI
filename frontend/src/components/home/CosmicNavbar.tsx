'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, useReducedMotion } from 'framer-motion';
import { MessageSquare, Menu, X } from 'lucide-react';
import { PALETTE } from '@/lib/motion';

// Same teal accent used everywhere below the hero (Intelligence Console,
// Scenario, Lifecycle, Candidate Portal, FAQ, Final CTA) — was previously a
// leftover indigo from an earlier design pass, which read as a different
// product on scroll. Text on the bright teal fill goes dark (matches
// FinalCtaSection's CTA), not white — white fails contrast on #55E5CF.
const ACCENT = PALETTE.accent;
const ACCENT_TEXT = '#07080a';

export const CosmicNavbar: React.FC = () => {
  const pathname = usePathname();
  const prefersReducedMotion = useReducedMotion();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [hoveredNav, setHoveredNav] = useState<string | null>(null);

  // `useReducedMotion()` reads `matchMedia` synchronously on the client but
  // the server has no `window` and always assumes "no preference" — so under
  // an OS/browser reduced-motion setting, the client's first render (used
  // for the hydration diff) already picks the `initial={false}` branch while
  // the server rendered `initial={{opacity:0}}`, a markup/style mismatch.
  // `hasMounted` starts `false` identically on both server and client's
  // first render, so the two stay in sync through hydration; motion only
  // adapts to the real reduced-motion preference in the effect that follows.
  const [hasMounted, setHasMounted] = useState(false);
  useEffect(() => {
    setHasMounted(true);
  }, []);
  const reduceMotion = hasMounted && !!prefersReducedMotion;

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 30);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Labels/routes match the old per-page AppSidebar exactly (Home, Interview,
  // Resume & Job, Reports, History) — only the position moved to this shared
  // top bar. "Interview" still points to /setup (the setup flow), same as
  // the old sidebar; /interview itself is only ever reached from within that
  // flow, never as a direct nav item.
  const navLinks = [
    { name: 'Home', href: '/' },
    { name: 'Interview', href: '/setup' },
    { name: 'Resume & Job', href: '/resume-job' },
    { name: 'Reports', href: '/report' },
    { name: 'History', href: '/history' },
  ];

  // This is now the one nav shell for every route (rendered from the root
  // layout, not just the homepage) — the fixed header used to only exist on
  // "/", so leaving any page via its links landed you on a completely
  // different nav system (the old AppSidebar) with different labels and no
  // way back to the other 4 links. Only "/interview" opts out, since that
  // page is a focused live-session view with its own self-contained header.
  // Away from the hero there's no full-bleed image for a transparent bar to
  // sit over, so it stays solid immediately instead of waiting for scroll.
  const isHome = pathname === '/';
  if (pathname === '/interview') return null;
  const showSolid = isScrolled || !isHome;

  return (
    <header className="fixed top-0 inset-x-0 z-50 select-none">
      <motion.div
        initial={reduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className={`flex items-center justify-between px-4 sm:px-6 h-14 border-b transition-colors duration-300 ${
          showSolid ? 'bg-[#07080a]/85 border-white/[0.08] backdrop-blur-md' : 'bg-transparent border-transparent'
        }`}
      >
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2 group">
          <div
            className="w-6 h-6 rounded-md flex items-center justify-center transition-opacity group-hover:opacity-85"
            style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
          >
            <MessageSquare className="w-3.5 h-3.5" />
          </div>
          <span className="font-semibold text-sm tracking-tight text-[#F2F0EA]">InterviewPal</span>
        </Link>

        {/* Center nav links */}
        <nav className="hidden lg:flex items-center gap-1" onMouseLeave={() => setHoveredNav(null)}>
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.name}
                href={link.href}
                onMouseEnter={() => setHoveredNav(link.name)}
                className={`relative px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors ${
                  isActive ? 'text-[#F2F0EA]' : 'text-[#97958F] hover:text-[#F2F0EA]'
                }`}
              >
                {hoveredNav === link.name && (
                  <motion.div
                    layoutId="navbarHoverHighlight"
                    className="absolute inset-0 rounded-md bg-white/[0.05] -z-10"
                    transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                  />
                )}
                {link.name}
              </Link>
            );
          })}
        </nav>

        {/* Right actions */}
        <div className="hidden lg:flex items-center gap-4">
          <Link
            href="/settings"
            className="text-[13px] font-medium text-[#97958F] hover:text-[#F2F0EA] transition-colors"
          >
            Settings
          </Link>

          <Link
            href="/setup"
            className="px-3.5 py-1.5 rounded-md text-[13px] font-semibold transition-opacity hover:opacity-90 active:opacity-80"
            style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
          >
            Start free
          </Link>
        </div>

        {/* Mobile toggle */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-1.5 rounded-md text-[#97958F] hover:text-[#F2F0EA] lg:hidden"
          aria-label="Toggle navigation"
          aria-expanded={mobileMenuOpen}
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </motion.div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className="lg:hidden px-4 pb-4 bg-[#07080a]/95 border-b border-white/[0.08] backdrop-blur-md flex flex-col gap-1"
        >
          {navLinks.map((link) => (
            <Link
              key={link.name}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className="px-3 py-2 rounded-md text-sm font-medium text-[#97958F] hover:bg-white/[0.04] hover:text-[#F2F0EA] transition-colors"
            >
              {link.name}
            </Link>
          ))}
          <div className="pt-2 mt-1 border-t border-white/[0.08] flex flex-col gap-2">
            <Link
              href="/setup"
              onClick={() => setMobileMenuOpen(false)}
              className="w-full py-2.5 rounded-md text-center text-sm font-semibold"
              style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
            >
              Start AI interview
            </Link>
          </div>
        </motion.div>
      )}
    </header>
  );
};
