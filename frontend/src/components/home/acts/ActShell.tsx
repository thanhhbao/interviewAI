'use client';

import React from 'react';

// Wraps every section belonging to one act under a single shared dark base
// and one continuous background field, so section boundaries inside an act
// stop reading as "card / gap / card" and start reading as one environment.
// The field fades at its own top/bottom edges (mask, not extra DOM) so
// consecutive acts blend into each other across a soft ~15-20vh seam
// instead of cutting hard.
export function ActShell({
  background,
  className = '',
  children,
}: {
  background: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`relative ${className}`} style={{ backgroundColor: '#060A0A' }}>
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          maskImage: 'linear-gradient(to bottom, transparent, black 16vh, black calc(100% - 16vh), transparent)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 16vh, black calc(100% - 16vh), transparent)',
        }}
      >
        {background}
      </div>
      <div className="relative">{children}</div>
    </div>
  );
}
