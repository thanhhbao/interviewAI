'use client';

import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, useReducedMotion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Menu, X, LogOut, User, Settings, ChevronDown } from 'lucide-react';
import { PALETTE } from '@/lib/motion';
import { useAuth } from '@/context/AuthContext';

// Same teal accent used everywhere below the hero (Intelligence Console,
// Scenario, Lifecycle, Candidate Portal, FAQ, Final CTA) — was previously a
// leftover indigo from an earlier design pass, which read as a different
// product on scroll. Text on the bright teal fill goes dark (matches
// FinalCtaSection's CTA), not white — white fails contrast on #E7E8E9.
const ACCENT = PALETTE.accent;
const ACCENT_TEXT = '#07080a';

// Ported from the actual "Nova Glow Navigation" Framer component
// (framer.com/marketplace/components/nova-glow-navigation). Values below
// are measured directly off the live reference's rendered DOM (computed
// styles + getBoundingClientRect on a 798x60 "Glass Pill"), not guessed —
// see the geometry note on ROTOR_WIDTH_RATIO below for the one figure that
// had to be re-derived because our pill is a different size than theirs.
//
// Confirmed DOM tree (measured, not inferred):
//   Border Glow Wrap (relative, overflow:hidden, padding:1px, radius 31px)
//    ├─ Border Rotor   (absolute, z-index:0, 1400x1662px, centered,
//    │                  rotate 0->360, 5s, linear, continuous)
//    └─ Glass Pill      (relative, z-index:1, overflow:hidden, radius 30px,
//                        fills the 1px-padding content box exactly —
//                        that 1px gap is what exposes a thin ring of the
//                        RAW rotor around the edge; the softer bloom seen
//                        through the pill elsewhere is the same rotor
//                        diffusing through backdrop-filter:blur(24px))
//        ├─ Top Hairline   (absolute, left 7.5%, width 70% of pill — not
//        │                  edge-to-edge, stops short before the CTA)
//        ├─ Shimmer Streak (absolute, width ~15% of pill, sweeps from
//        │                  ~-20% to ~+108% of pill width, 2.2s sweep +
//        │                  2.5s repeatDelay)
//        └─ Top Row (actual content)
//
// Full 5s-cycle capture against the live reference confirmed the highlight
// genuinely travels the WHOLE perimeter — top (one direction) -> corner ->
// bottom (opposite direction) -> corner -> back to top — not just the top
// edge. What makes that work is Border Rotor's size relative to the pill:
// 1400x1662 against their 798-wide pill is ~1.75x/2.08x the pill's WIDTH.
// Our previous 1400x1400 square was only ~1.2x our (wider) pill's width —
// nowhere near large/tall enough relative to the pill for the angle-to-
// edge mapping to read as even travel, which is what produced the uneven
// "blob on top, thin at the sides" look. ROTOR_WIDTH_RATIO/ROTOR_HEIGHT_RATIO
// reproduce their measured ratios against whatever pill width we pass in.
const ROTOR_WIDTH_RATIO = 1400 / 798; // ~1.754
const ROTOR_HEIGHT_RATIO = 1662 / 798; // ~2.083

// DEBUG_LAYERING: temporary high-contrast mode to verify the Border Rotor
// is only ever exposed through the thin outer ring (never as a band
// crossing the glass body). Solid black pill, no blur, no shadow, no
// shimmer, red/white rotor. Must be false before shipping.
const DEBUG_LAYERING = false;

const BORDER_ROTOR_GRADIENT = DEBUG_LAYERING
  ? 'conic-gradient(from 0deg at 50% 50%, transparent 0deg, transparent 60deg, red 78deg, white 90deg, red 100deg, transparent 101deg)'
  : 'conic-gradient(from 0deg at 50% 50%, rgba(231, 232, 233,0) 0deg, rgba(231, 232, 233,0) 60deg, rgba(255, 255, 255,0.7) 78deg, rgb(255, 255, 255) 86deg, rgb(255,255,255) 90deg, rgba(231, 232, 233,0) 100deg)';
const HAIRLINE_GRADIENT =
  'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0) 100%)';
const SHIMMER_GRADIENT =
  'linear-gradient(105deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.06) 45%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0.06) 55%, rgba(255,255,255,0) 100%)';

function NovaGlassBar({
  radius,
  pillWidth,
  solid,
  reduceMotion,
  className = '',
  children,
}: {
  radius: number;
  // Reference width to scale the rotor off of — pass the bar's own
  // (approximate) rendered width; the reference's rotor was a fixed size
  // authored for their fixed 798px pill, not computed responsively, so a
  // size derived once from each usage's typical width (not a live
  // ResizeObserver) is the faithful equivalent for our two usages
  // (desktop bar vs. mobile menu panel).
  pillWidth: number;
  solid: boolean;
  reduceMotion: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  const rotorWidth = Math.round(pillWidth * ROTOR_WIDTH_RATIO);
  const rotorHeight = Math.round(pillWidth * ROTOR_HEIGHT_RATIO);
  return (
    <div className={`relative overflow-hidden ${className}`} style={{ borderRadius: radius }}>
      {/* Border Rotor — centering (CSS class, static) and rotation (Framer
          Motion, dynamic) are deliberately on TWO nested elements, not
          combined on one. Framer Motion writes the full `transform`
          property via inline style once it animates a motion value like
          `rotate`, which silently overwrites — not composes with — a
          Tailwind translate utility class also targeting `transform`. That
          previously dropped the -50%/-50% centering the instant the
          rotation animation started, flinging the (huge) rotor far outside
          the clipped ring with nothing visible left inside it. */}
      <div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
        style={{ width: rotorWidth, height: rotorHeight }}
      >
        <motion.div
          className="w-full h-full"
          style={{ background: BORDER_ROTOR_GRADIENT }}
          animate={reduceMotion ? undefined : { rotate: 360 }}
          transition={reduceMotion ? undefined : { duration: 5, ease: [0, 0, 1, 1], repeat: Infinity }}
        />
      </div>

      {/* Glass Pill — inset by 1px from the outer clip (measured value)
          so a thin ring of the raw Border Rotor stays directly exposed
          (not diffused through the glass's own blur+opacity) right at the
          edge. That's the part that reads as "a bright border traveling
          around the outside"; the softer bloom seen through the glass
          elsewhere is a second, distinct effect on top of it.
          The pill's own base is intentionally neutral (no teal) — every
          bit of color here comes from the rotor showing through/around it,
          never from a static tinted border or shadow, so the eye can only
          read color where the light is actually moving. */}
      <div
        className="absolute inset-[1px] overflow-hidden"
        style={
          DEBUG_LAYERING
            ? { borderRadius: Math.max(0, radius - 1), background: '#000' }
            : {
                borderRadius: Math.max(0, radius - 1),
                backdropFilter: 'blur(24px)',
                WebkitBackdropFilter: 'blur(24px)',
                // High opacity is the actual fix for light bleed, not the
                // ring geometry (already verified edge-only with a solid
                // black pill in DEBUG_LAYERING). The Border Rotor is a
                // giant square sitting behind this ENTIRE surface, not just
                // behind the 1px ring — at the old 0.5-0.55 opacity (the
                // `!solid` / over-hero state) nearly half of that giant
                // rotor showed straight through across the whole interior
                // whenever its bright wedge swept past, not just at the
                // edge. Near-opaque fill blocks that; the exposed 1px ring
                // plus this glass's own backdrop-blur(24px) softening ~24px
                // inward from any bright boundary is what still gives a
                // small "local reflection" near the edge — not a second
                // layer, just blur radius acting on a mostly-blocked rotor.
                background: solid
                  ? 'linear-gradient(180deg, rgba(18,26,25,0.99) 0%, rgba(7,8,10,0.98) 100%)'
                  : 'linear-gradient(180deg, rgba(18,26,25,0.92) 0%, rgba(7,8,10,0.9) 100%)',
                boxShadow:
                  '0px 30px 80px 0px rgba(0,0,0,0.65), inset 0px 1px 0px 0px rgba(255,255,255,0.10)',
              }
        }
      >
        {/* Top Hairline — inset, not edge-to-edge (measured: starts ~7.5%
            in, spans ~70% of the pill, stopping short of the CTA area).
            Neutral white, not teal — a static base edge, not a second
            competing "animation". */}
        {!DEBUG_LAYERING && (
          <div
            className="absolute top-0 left-[7.5%] right-[22.5%] h-px pointer-events-none"
            style={{ background: HAIRLINE_GRADIENT }}
          />
        )}

        {/* Shimmer Streak — TEMPORARILY DISABLED. It was reading as the
            primary animation (a bright band sweeping the glass body),
            competing with/masking the Border Rotor's own motion at the
            edge. Re-enable only once the rotor's border movement is
            confirmed correct on its own, and much weaker than the rotor
            when it comes back. */}
        {false && (
          <motion.div
            className="absolute -inset-y-6 w-[15%] pointer-events-none"
            style={{ background: SHIMMER_GRADIENT }}
            animate={reduceMotion ? undefined : { x: ['-133%', '720%'] }}
            transition={
              reduceMotion
                ? undefined
                : { duration: 2.2, ease: [0.45, 0, 0.55, 1], repeat: Infinity, repeatDelay: 2.5 }
            }
          />
        )}
      </div>

      {/* Actual content — normal flow, on top of both layers above; this is
          what gives the whole bar its real height (both layers above are
          `absolute` and contribute none), same as before. */}
      <div className="relative">{children}</div>
    </div>
  );
}

function ProfileDropdown({ user, signOut }: { user: { email?: string | null; user_metadata?: Record<string, string> }, signOut: () => void }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, right: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (btnRef.current && !btnRef.current.contains(e.target as Node)) {
        const target = e.target as Element;
        if (!target.closest('[data-profile-menu]')) setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function handleToggle() {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPos({ top: rect.bottom + 8, right: window.innerWidth - rect.right });
    }
    setOpen(v => !v);
  }

  const initials = (user.user_metadata?.full_name || user.email || '?')
    .split(/[\s@]/).filter(Boolean).slice(0, 2)
    .map((s: string) => s[0].toUpperCase()).join('');
  const displayName = user.user_metadata?.full_name || user.email?.split('@')[0] || 'User';

  const menu = (
    <AnimatePresence>
      {open && (
        <motion.div
          data-profile-menu
          initial={{ opacity: 0, y: -6, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -6, scale: 0.97 }}
          transition={{ duration: 0.14, ease: [0.16, 1, 0.3, 1] }}
          style={{ position: 'fixed', top: pos.top, right: pos.right, zIndex: 9999, width: 220 }}
          className="rounded-xl border border-white/[0.08] bg-[#0A1110] shadow-2xl"
        >
          {/* User info */}
          <div className="px-3.5 py-3 border-b border-white/[0.06]">
            <p className="text-[13px] font-medium text-[#ECEDE9] truncate">{displayName}</p>
            <p className="text-[11px] text-[#4a5250] truncate mt-0.5">{user.email}</p>
          </div>

          {/* Menu items */}
          <div className="p-1.5">
            <Link
              href="/profile"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-[13px] text-[#97958F] hover:text-[#ECEDE9] hover:bg-white/[0.04] transition-colors"
            >
              <User className="w-3.5 h-3.5" />
              Profile
            </Link>
            <Link
              href="/settings"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-[13px] text-[#97958F] hover:text-[#ECEDE9] hover:bg-white/[0.04] transition-colors"
            >
              <Settings className="w-3.5 h-3.5" />
              Settings
            </Link>
          </div>

          {/* Sign out */}
          <div className="p-1.5 border-t border-white/[0.06]">
            <button
              onClick={() => { setOpen(false); signOut(); }}
              className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-[13px] text-[#97958F] hover:text-red-400 hover:bg-red-950/20 transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <>
      <button
        ref={btnRef}
        onClick={handleToggle}
        className="flex items-center gap-2 pl-1 pr-2.5 py-1 rounded-full hover:bg-white/[0.05] transition-colors group"
      >
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-semibold flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #1a2e2a 0%, #0d1f1c 100%)', color: '#E7E8E9', border: '1px solid rgba(255,255,255,0.1)' }}
        >
          {initials}
        </div>
        <span className="text-[13px] text-[#97958F] group-hover:text-[#F2F0EA] transition-colors max-w-[120px] truncate">
          {displayName}
        </span>
        <ChevronDown className={`w-3 h-3 text-[#97958F] transition-transform duration-150 ${open ? 'rotate-180' : ''}`} />
      </button>

      {mounted && createPortal(menu, document.body)}
    </>
  );
}

export const CosmicNavbar: React.FC = () => {
  const pathname = usePathname();
  const prefersReducedMotion = useReducedMotion();
  const { user, signOut, openLoginModal } = useAuth();
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

  // One coordinated entrance: the bar fades/scales in first, then brand ->
  // links -> actions stagger in behind it. This is the single orchestrated
  // moment for the nav (mount only) — everything after is ordinary
  // hover/active transitions or the Nova glass bar's own continuous glow.
  const barVariants = {
    hidden: { opacity: 0, y: -10, scale: 0.98 },
    show: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { duration: 0.5, ease: 'easeOut', staggerChildren: 0.05, delayChildren: 0.08 },
    },
  };
  const itemVariants = {
    hidden: { opacity: 0, y: -6 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  return (
    <header className="fixed top-0 inset-x-0 z-50 select-none p-3">
      <motion.div
        initial={reduceMotion ? false : 'hidden'}
        animate="show"
        variants={barVariants}
        className="relative mx-auto max-w-6xl"
      >
        <NovaGlassBar radius={9999} pillWidth={1152} solid={showSolid} reduceMotion={reduceMotion}>
          <div className="relative flex items-center justify-between px-4 sm:px-6 h-14">
            {/* Brand */}
            <motion.div variants={itemVariants}>
              <Link href="/" className="flex items-center gap-2 group">
                <div
                  className="w-6 h-6 rounded-md flex items-center justify-center transition-opacity group-hover:opacity-85"
                  style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                </div>
                <span className="font-semibold text-sm tracking-tight text-[#F2F0EA]">InterviewAI</span>
              </Link>
            </motion.div>

            {/* Center nav links */}
            <motion.nav
              variants={itemVariants}
              className="hidden lg:flex items-center gap-1"
              onMouseLeave={() => setHoveredNav(null)}
            >
              {navLinks.map((link) => {
                const isActive = pathname === link.href;
                return (
                  <Link
                    key={link.name}
                    href={link.href}
                    onMouseEnter={() => setHoveredNav(link.name)}
                    className={`relative px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors ${
                      isActive ? 'text-[#F2F0EA]' : 'text-[#97958F] hover:text-[#F2F0EA]'
                    }`}
                  >
                    {hoveredNav === link.name && (
                      <motion.div
                        layoutId="navbarHoverHighlight"
                        className="absolute inset-0 rounded-full bg-white/[0.05] -z-10"
                        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                      />
                    )}
                    {link.name}
                  </Link>
                );
              })}
            </motion.nav>

            {/* Right actions */}
            <motion.div variants={itemVariants} className="hidden lg:flex items-center gap-4">
              {user ? (
                <>
                  <ProfileDropdown user={user} signOut={signOut} />
                  <Link
                    href="/setup"
                    className="px-3.5 py-1.5 rounded-full text-[13px] font-semibold transition-opacity hover:opacity-90 active:opacity-80"
                    style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
                  >
                    Start interview
                  </Link>
                </>
              ) : (
                <>
                  <button
                    onClick={openLoginModal}
                    className="text-[13px] font-medium text-[#97958F] hover:text-[#F2F0EA] transition-colors"
                  >
                    Sign in
                  </button>
                  <button
                    onClick={openLoginModal}
                    className="px-3.5 py-1.5 rounded-full text-[13px] font-semibold transition-opacity hover:opacity-90 active:opacity-80"
                    style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
                  >
                    Get started
                  </button>
                </>
              )}
            </motion.div>

            {/* Mobile toggle */}
            <motion.button
              variants={itemVariants}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-1.5 rounded-full text-[#97958F] hover:text-[#F2F0EA] lg:hidden"
              aria-label="Toggle navigation"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </motion.button>
          </div>
        </NovaGlassBar>
      </motion.div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className="relative mx-auto max-w-6xl mt-2 lg:hidden"
        >
          <NovaGlassBar radius={24} pillWidth={420} solid reduceMotion={reduceMotion}>
            <div className="relative px-4 py-3 flex flex-col gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-3 py-2 rounded-lg text-sm font-medium text-[#97958F] hover:bg-white/[0.04] hover:text-[#F2F0EA] transition-colors"
                >
                  {link.name}
                </Link>
              ))}
              <div className="pt-2 mt-1 border-t border-white/[0.08] flex flex-col gap-2">
                {user ? (
                  <>
                    <span className="px-3 py-1 text-xs text-[#97958F] truncate">{user.email}</span>
                    <button
                      onClick={() => { setMobileMenuOpen(false); signOut(); }}
                      className="w-full py-2.5 rounded-full text-center text-sm font-medium text-[#97958F] border border-white/[0.08]"
                    >
                      Sign out
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => { setMobileMenuOpen(false); openLoginModal(); }}
                    className="w-full py-2.5 rounded-full text-center text-sm font-semibold"
                    style={{ backgroundColor: ACCENT, color: ACCENT_TEXT }}
                  >
                    Sign in
                  </button>
                )}
              </div>
            </div>
          </NovaGlassBar>
        </motion.div>
      )}
    </header>
  );
};
