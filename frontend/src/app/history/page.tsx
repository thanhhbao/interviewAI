'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { TrendingUp, RotateCcw, Calendar, Clock } from 'lucide-react';
import { samplePastInterviews } from '@/lib/mockData';
import { AppBackground } from '@/components/app/AppBackground';
import { PALETTE } from '@/lib/motion';

const panelStyle = { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` };

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
      {children}
    </h2>
  );
}

function StatTile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="p-5 rounded-[16px] space-y-2" style={panelStyle}>
      <span className="text-[11px] font-medium block" style={{ color: PALETTE.textSecondary }}>
        {label}
      </span>
      <span className="text-2xl font-bold" style={{ color: PALETTE.textPrimary }}>
        {value}
      </span>
      <span className="text-[10px] block" style={{ color: PALETTE.accent }}>
        {note}
      </span>
    </div>
  );
}

export default function HistoryPage() {
  const router = useRouter();
  const [selectedFilter, setSelectedFilter] = useState('all');

  // Route through Setup rather than starting a session directly — the user
  // should get a chance to review/change configuration before practicing
  // again, and Setup is the only place a session gets created.
  const handlePracticeAgain = () => {
    router.push('/setup');
  };

  const filteredInterviews =
    selectedFilter === 'all'
      ? samplePastInterviews
      : samplePastInterviews.filter((item) =>
          item.type.toLowerCase().includes(selectedFilter.toLowerCase())
        );

  const filters: Array<{ id: string; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'technical', label: 'Technical' },
    { id: 'coding', label: 'Coding' },
  ];

  return (
    <div className="relative flex-1 flex flex-col pb-16">
      <AppBackground />

      <main className="relative flex-1 max-w-5xl w-full mx-auto px-6 md:px-8 py-10 md:py-14 space-y-10">
        {/* PAGE INTRO */}
        <div className="max-w-2xl space-y-2.5">
          <p className="text-[11px] font-mono tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
            Interview preparation / History
          </p>
          <h1 className="text-2xl md:text-[28px] font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            History &amp; Progress
          </h1>
          <p className="text-sm leading-relaxed" style={{ color: PALETTE.textSecondary }}>
            Track your interview performance trends and review past sessions.
          </p>
        </div>

        {/* LIGHTWEIGHT PROGRESS ANALYTICS TILES */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <SectionLabel>Performance Trend</SectionLabel>
            <span className="text-[11px] font-medium flex items-center gap-1" style={{ color: PALETTE.accent }}>
              <TrendingUp className="w-3.5 h-3.5" />
              +7 points improvement over last 3 sessions
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatTile label="Average Overall Score" value="84.3" note="Target: 85+ (Strong Hire)" />
            <StatTile label="Coding & Algorithms" value="92%" note="Top 5% candidate pool" />
            <StatTile label="System Design Depth" value="88%" note="High architectural rigor" />
            <StatTile label="STAR Communication" value="86%" note="Consistent structure" />
          </div>

          {/* Simple Visual Score Progression */}
          <div className="p-6 rounded-[16px] space-y-4" style={panelStyle}>
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                Score Progression Over Time
              </span>
              <span className="font-mono text-[11px]" style={{ color: PALETTE.textSecondary }}>Last 3 Interviews</span>
            </div>

            {/* Visual Bar Chart */}
            <div className="space-y-3 pt-2">
              {[
                { label: 'Vercel (Aug 29)', score: 81, alpha: '55' },
                { label: 'Stripe (Sep 14)', score: 84, alpha: '85' },
                { label: 'Nova Technologies (Sep 21)', score: 88, alpha: 'FF' },
              ].map((b) => (
                <div key={b.label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span style={{ color: PALETTE.textSecondary }}>{b.label}</span>
                    <span className="font-bold" style={{ color: PALETTE.textPrimary }}>
                      {b.score}/100
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${b.score}%`, backgroundColor: `${PALETTE.accent}${b.alpha}` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* INTERVIEW SESSIONS LIST */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <SectionLabel>Completed Interview Sessions</SectionLabel>

            {/* Simple Filter Pills */}
            <div className="flex items-center gap-2">
              {filters.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setSelectedFilter(f.id)}
                  className="px-3 py-1 rounded-[8px] text-xs font-medium transition-colors"
                  style={
                    selectedFilter === f.id
                      ? { backgroundColor: `${PALETTE.accent}1F`, color: PALETTE.accent }
                      : { color: PALETTE.textSecondary }
                  }
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            {filteredInterviews.map((item) => (
              <div
                key={item.id}
                className="p-5 rounded-[16px] flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors"
                style={panelStyle}
              >
                <div className="space-y-1.5 min-w-0">
                  <div className="flex items-center gap-2.5">
                    <span className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
                      {item.role}
                    </span>
                    <span className="text-xs" style={{ color: PALETTE.textSecondary }}>· {item.company}</span>
                  </div>

                  <div className="flex items-center gap-3 text-xs font-medium" style={{ color: PALETTE.textSecondary }}>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" style={{ color: PALETTE.textSecondary }} />
                      {item.date}
                    </span>
                    <span>•</span>
                    <span>{item.type}</span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" style={{ color: PALETTE.textSecondary }} />
                      {item.duration}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                  <div className="text-right">
                    <span className="text-base font-bold" style={{ color: PALETTE.textPrimary }}>
                      {item.score}
                    </span>
                    <span className="text-xs font-normal" style={{ color: PALETTE.textSecondary }}> / 100</span>
                    <span className="block text-[10px] font-semibold" style={{ color: PALETTE.accent }}>
                      {item.verdict}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Link
                      href="/report"
                      className="px-4 py-2 rounded-[8px] text-xs font-semibold transition-colors hover:bg-white/[0.05]"
                      style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textPrimary }}
                    >
                      Open Report
                    </Link>

                    <button
                      onClick={handlePracticeAgain}
                      className="p-2 rounded-[8px] transition-colors hover:bg-white/[0.05]"
                      style={{ color: PALETTE.textSecondary }}
                      title="Practice Again"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
