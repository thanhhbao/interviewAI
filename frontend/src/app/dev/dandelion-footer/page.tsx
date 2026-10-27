import { notFound } from 'next/navigation';
import { DandelionFooter } from '@/components/home/footer/DandelionFooter';

export const metadata = {
  title: 'Dandelion Footer — dev preview',
};

// Isolated preview for the DandelionFooter prototype (see PR/task notes).
// Not linked from any nav — reachable only by typing the URL, and 404s
// outright in production so it never ships as a real route.
export default function DandelionFooterPreviewPage() {
  if (process.env.NODE_ENV === 'production') {
    notFound();
  }

  return (
    <div className="min-h-screen w-full" style={{ backgroundColor: '#07080a' }}>
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-24 text-center">
        <p className="text-[11px] font-mono tracking-[0.2em] uppercase text-neutral-500 mb-3">
          DEV PREVIEW — NOT A PRODUCTION ROUTE
        </p>
        <h1 className="text-2xl font-semibold text-neutral-100">DandelionFooter prototype</h1>
        <p className="mt-3 text-sm text-neutral-500">
          Scroll down to trigger the entrance animation. DandelionFooter is now the production
          Home footer (see src/app/page.tsx) — this page exists for isolated visual regression
          checks, without the rest of Home around it.
        </p>
      </div>

      {/* Filler so the footer starts below the fold and its scroll-entrance
          can actually be observed, without turning this into a long
          scroll-scrubbed experience itself. */}
      <div className="h-[70vh]" />

      <DandelionFooter />
    </div>
  );
}
