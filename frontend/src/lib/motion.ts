// Shared motion language for the post-hero half of the homepage (Scenario →
// Footer). Keeping these in one place is what makes the section feel like
// one coherent system instead of per-component animation choices.
// `useSafeReducedMotion` below is general-purpose and also used by the hero
// (HeroScanOverlay.tsx) to fix a hydration mismatch.
import { useEffect, useState } from 'react';
import { useReducedMotion, type Variants } from 'framer-motion';

// `useReducedMotion()` reads `matchMedia` synchronously on the client, but
// the server has no `window` and always assumes "no preference". Using its
// raw value to change initial motion props/DOM structure/motion-value ranges
// therefore diverges between the server-rendered markup and the client's
// first (pre-effect) render used for the hydration diff — a classic
// server/client mismatch. Gating on a `hasMounted` flag (false identically
// on both the server and the client's first render, flipped only inside a
// post-commit effect) keeps hydration deterministic; real reduced-motion
// behavior applies one tick later, which is imperceptible.
export function useSafeReducedMotion(): boolean {
  const prefersReducedMotion = useReducedMotion();
  const [hasMounted, setHasMounted] = useState(false);
  useEffect(() => {
    setHasMounted(true);
  }, []);
  return hasMounted && !!prefersReducedMotion;
}

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: EASE_OUT } },
};

export const fadeUpSmall: Variants = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE_OUT } },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.98 },
  show: { opacity: 1, scale: 1, transition: { duration: 0.6, ease: EASE_OUT } },
};

export const SPRING_SOFT = { type: 'spring' as const, stiffness: 260, damping: 30 };
export const SPRING_SNAPPY = { type: 'spring' as const, stiffness: 420, damping: 34 };

export const viewportOnce = { once: true, margin: '-80px' } as const;

// Palette constants shared by every redesigned section below the fold.
export const PALETTE = {
  bg: '#07080a',
  bg2: '#070C0C',
  surface: '#0A1110',
  surface2: '#0C1413',
  surfaceFaint: 'rgba(255,255,255,0.025)',
  border: 'rgba(255,255,255,0.07)',
  borderTeal: 'rgba(160,220,210,0.08)',
  textPrimary: '#ECEDE9',
  textSecondary: '#858E8A',
  accent: '#55E5CF',
  accentBright: '#78FFE7',
  deepTeal: '#14A98F',
} as const;
