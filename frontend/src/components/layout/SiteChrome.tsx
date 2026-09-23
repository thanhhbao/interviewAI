'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { CosmicNavbar } from '@/components/home/CosmicNavbar';

// Single nav shell for the whole app (replaces the old split between
// CosmicNavbar — homepage only — and AppSidebar — every other route with a
// different link set). CosmicNavbar itself decides whether to render (it
// hides on /interview); this wrapper only needs to know whether to reserve
// space for it. The homepage doesn't need the offset: its hero is designed
// to sit full-bleed under the fixed, initially-transparent bar. Every other
// page needs `pt-14` so the now-solid bar doesn't cover their content.
export function SiteChrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const needsOffset = pathname !== '/' && pathname !== '/interview';

  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-screen">
      <CosmicNavbar />
      <div className={`flex-1 flex flex-col min-h-screen ${needsOffset ? 'pt-14' : ''}`}>{children}</div>
    </div>
  );
}
