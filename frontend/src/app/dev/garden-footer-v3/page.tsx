import { notFound } from 'next/navigation';
import { GardenFooterV3 } from '@/components/home/garden-v3/GardenFooterV3';

export const metadata = {
  title: 'Garden Footer V3 (static) — dev preview',
};

// Phase 1 static-only prototype for a precomposed-texture meadow footer —
// see the task report for why this replaces the V2 per-flower-instance
// approach. No scroll, no camera travel, no animation of any kind; this
// route exists purely to validate one still frame. Isolated at
// /dev/garden-footer-v3 — 404s outright in production, not linked from
// any nav, and does not touch the V2 prototype at /dev/dandelion-garden-v2
// or production Home.
export default function GardenFooterV3PreviewPage() {
  if (process.env.NODE_ENV === 'production') {
    notFound();
  }

  return (
    <div className="min-h-screen w-full" style={{ backgroundColor: '#020302' }}>
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-10 text-center">
        <p className="text-[11px] font-mono tracking-[0.2em] uppercase text-neutral-500 mb-3">
          DEV PREVIEW — NOT A PRODUCTION ROUTE — STATIC ONLY, NO ANIMATION
        </p>
        <h1 className="text-2xl font-semibold text-neutral-100">Garden Footer V3 (static)</h1>
        <p className="mt-3 text-sm text-neutral-500">
          Precomposed-texture meadow, phase 1: one static frame, no scroll or camera motion.
          Compare against the instanced-flower approach at{' '}
          <a href="/dev/dandelion-garden-v2" className="underline">
            /dev/dandelion-garden-v2
          </a>
          .
        </p>
      </div>

      <GardenFooterV3 />

      <div className="h-[40vh] flex items-center justify-center text-neutral-600 text-sm">
        (page content would continue here in production)
      </div>
    </div>
  );
}
