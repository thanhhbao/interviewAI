'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  History,
  TrendingUp,
  RotateCcw,
  BarChart3,
  Calendar,
  Clock,
  CheckCircle2,
  Filter,
  ArrowUpRight,
  Sparkles,
} from 'lucide-react';
import { samplePastInterviews } from '@/lib/mockData';
import { TopHeader } from '@/components/layout/TopHeader';
import { useInterview } from '@/context/InterviewContext';

export default function HistoryPage() {
  const router = useRouter();
  const { startInterviewSession } = useInterview();
  const [selectedFilter, setSelectedFilter] = useState('all');

  const handlePracticeAgain = (type: string) => {
    startInterviewSession('full');
    router.push('/interview');
  };

  const filteredInterviews =
    selectedFilter === 'all'
      ? samplePastInterviews
      : samplePastInterviews.filter((item) =>
          item.type.toLowerCase().includes(selectedFilter.toLowerCase())
        );

  return (
    <div className="flex-1 flex flex-col min-h-screen pb-16">
      <TopHeader
        title="History & Progress"
        subtitle="Track your interview performance trends and review past sessions"
      />

      <main className="flex-1 max-w-5xl w-full mx-auto px-8 py-8 space-y-10">
        {/* LIGHTWEIGHT PROGRESS ANALYTICS TILES */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
              Performance Trend
            </h2>
            <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1">
              <TrendingUp className="w-3.5 h-3.5" />
              +7 points improvement over last 3 sessions
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-2">
              <span className="text-[11px] text-neutral-400 block font-medium">
                Average Overall Score
              </span>
              <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                84.3
              </span>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 block">
                Target: 85+ (Strong Hire)
              </span>
            </div>

            <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-2">
              <span className="text-[11px] text-neutral-400 block font-medium">
                Coding & Algorithms
              </span>
              <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                92%
              </span>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 block">
                Top 5% candidate pool
              </span>
            </div>

            <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-2">
              <span className="text-[11px] text-neutral-400 block font-medium">
                System Design Depth
              </span>
              <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                88%
              </span>
              <span className="text-[10px] text-indigo-600 dark:text-indigo-400 block">
                High architectural rigor
              </span>
            </div>

            <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-2">
              <span className="text-[11px] text-neutral-400 block font-medium">
                STAR Communication
              </span>
              <span className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                86%
              </span>
              <span className="text-[10px] text-neutral-500 block">Consistent structure</span>
            </div>
          </div>

          {/* Simple Visual Score Progression */}
          <div className="p-6 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                Score Progression Over Time
              </span>
              <span className="text-neutral-400 font-mono text-[11px]">Last 3 Interviews</span>
            </div>

            {/* Visual Bar Chart */}
            <div className="space-y-3 pt-2">
              {[
                { label: 'Vercel (Aug 29)', score: 81, color: 'bg-neutral-400' },
                { label: 'Stripe (Sep 14)', score: 84, color: 'bg-indigo-400' },
                { label: 'Nova Technologies (Sep 21)', score: 88, color: 'bg-indigo-600' },
              ].map((b) => (
                <div key={b.label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400">{b.label}</span>
                    <span className="font-bold text-neutral-900 dark:text-neutral-100">
                      {b.score}/100
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
                    <div
                      className={`h-full ${b.color} rounded-full transition-all duration-500`}
                      style={{ width: `${b.score}%` }}
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
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
              Completed Interview Sessions
            </h2>

            {/* Simple Filter Pills */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSelectedFilter('all')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  selectedFilter === 'all'
                    ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950'
                    : 'text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setSelectedFilter('technical')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  selectedFilter === 'technical'
                    ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950'
                    : 'text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                Technical
              </button>
              <button
                onClick={() => setSelectedFilter('coding')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  selectedFilter === 'coding'
                    ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950'
                    : 'text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                Coding
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {filteredInterviews.map((item) => (
              <div
                key={item.id}
                className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:border-neutral-300 dark:hover:border-neutral-700 transition-colors shadow-xs"
              >
                <div className="space-y-1.5 min-w-0">
                  <div className="flex items-center gap-2.5">
                    <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                      {item.role}
                    </span>
                    <span className="text-xs text-neutral-400">· {item.company}</span>
                  </div>

                  <div className="flex items-center gap-3 text-xs text-neutral-400 font-medium">
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-neutral-400" />
                      {item.date}
                    </span>
                    <span>•</span>
                    <span>{item.type}</span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-neutral-400" />
                      {item.duration}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                  <div className="text-right">
                    <span className="text-base font-bold text-neutral-900 dark:text-neutral-100">
                      {item.score}
                    </span>
                    <span className="text-xs text-neutral-400 font-normal"> / 100</span>
                    <span className="block text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                      {item.verdict}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Link
                      href="/report"
                      className="px-4 py-2 rounded-xl bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 text-xs font-semibold text-neutral-800 dark:text-neutral-200 transition-colors"
                    >
                      Open Report
                    </Link>

                    <button
                      onClick={() => handlePracticeAgain(item.type)}
                      className="p-2 rounded-xl text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
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
