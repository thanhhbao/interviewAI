'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { CosmicNavbar } from '@/components/home/CosmicNavbar';
import { AppFooter } from '@/components/app/AppFooter';

// Single nav+footer shell for the whole app.
// CosmicNavbar hides itself on /interview; this wrapper uses pathname to:
//   • reserve pt-20 for pages that need the fixed-bar offset (non-home, non-interview)
//   • inject AppFooter on app routes but NOT on / (has DandelionFooter) or
//     /interview (full-screen live session — footer would interfere)
//
// The flex structure here guarantees footer sticks to the bottom on short pages:
//   outer (min-h-screen flex-col)
//     CosmicNavbar   (fixed, no flow space)
//     inner (flex-1 flex-col pt-20?)
//       content-wrapper (flex-1 flex-col) ← expands to push footer down
//         {children}
//       AppFooter
export function SiteChrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const needsOffset = pathname !== '/' && pathname !== '/interview';
  const showFooter  = pathname !== '/' && pathname !== '/interview';

  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-screen">
      <CosmicNavbar />
      <div className={`flex-1 flex flex-col ${needsOffset ? 'pt-20' : ''}`}>
        {/* flex-1 pushes AppFooter to the natural bottom of the viewport on
            short pages; on long pages it flows below the scrollable content */}
        <div className="flex-1 flex flex-col">
          {children}
        </div>
        {showFooter && <AppFooter />}
      </div>
    </div>
  );
}
