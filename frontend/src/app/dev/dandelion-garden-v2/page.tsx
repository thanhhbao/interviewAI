import { notFound } from 'next/navigation';
import { DandelionGarden } from '@/components/home/garden/DandelionGarden';

export const metadata = {
  title: 'Dandelion Garden V2 — dev preview',
};

// Isolated preview for the V2 "glide through a forest" scene — a real
// scroll-driven PerspectiveCamera moving through a 3D field of instanced
// dandelions (see src/components/home/garden/). Not linked from any nav;
// 404s outright in production so it never ships as a real route. The V1
// footer prototype at /dev/dandelion-footer is left in place for
// side-by-side comparison — do not remove it as part of this task.
export default function DandelionGardenV2PreviewPage() {
  if (process.env.NODE_ENV === 'production') {
    notFound();
  }

  return (
    <div className="min-h-screen w-full" style={{ backgroundColor: '#020302' }}>
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-10 text-center">
        <p className="text-[11px] font-mono tracking-[0.2em] uppercase text-neutral-500 mb-3">
          DEV PREVIEW — NOT A PRODUCTION ROUTE
        </p>
        <h1 className="text-2xl font-semibold text-neutral-100">Dandelion Garden V2 prototype</h1>
        <p className="mt-3 text-sm text-neutral-500">
          Scroll down through the 200vh section below to travel through the field. Move the mouse
          for subtle parallax. Compare against the V1 approach at{' '}
          <a href="/dev/dandelion-footer" className="underline">
            /dev/dandelion-footer
          </a>
          .
        </p>
      </div>

      <DandelionGarden />

      <div className="h-[40vh] flex items-center justify-center text-neutral-600 text-sm">
        (page content would continue here in production)
      </div>
    </div>
  );
}
