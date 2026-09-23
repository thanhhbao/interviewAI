'use client';

import React from 'react';

const companies = [
  { name: 'Google', symbol: 'G' },
  { name: 'Meta', symbol: 'M' },
  { name: 'Apple', symbol: '' },
  { name: 'OpenAI', symbol: '✦' },
  { name: 'Amazon', symbol: 'A' },
  { name: 'Stripe', symbol: 'S' },
  { name: 'Netflix', symbol: 'N' },
  { name: 'Microsoft', symbol: '⊞' },
  { name: 'Uber', symbol: 'U' },
  { name: 'Airbnb', symbol: '▲' },
];

// Quiet transitional strip between the cinematic hero and the product
// content — credibility evidence, not a feature section, so it recedes:
// monochrome text at partial opacity, no card chrome, no color.
export const CompanyMarquee: React.FC = () => {
  return (
    <div className="relative w-full overflow-hidden py-7 border-t border-white/[0.06]">
      <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-20 md:w-40 bg-gradient-to-r from-[#060A0A] to-transparent z-10" />
      <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-20 md:w-40 bg-gradient-to-l from-[#060A0A] to-transparent z-10" />

      <p className="text-center text-[11px] tracking-wide text-[#6f6d68] mb-5">
        Interview practice built for roles at
      </p>

      <div className="flex select-none group">
        {[0, 1].map((dup) => (
          <div
            key={dup}
            aria-hidden={dup === 1}
            className="flex shrink-0 items-center gap-10 pr-10 group-hover:[animation-play-state:paused] animate-marquee"
          >
            {companies.concat(companies).map((c, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-[#97958F]/45 hover:text-[#97958F]/80 transition-colors cursor-default"
              >
                <span className="text-xs font-mono">{c.symbol}</span>
                <span className="text-sm font-medium tracking-tight">{c.name}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
