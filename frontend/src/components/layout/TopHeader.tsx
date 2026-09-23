'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ArrowLeft, Play, Sparkles } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';

interface TopHeaderProps {
  title?: string;
  subtitle?: string;
  showBack?: boolean;
  backHref?: string;
  actionButton?: React.ReactNode;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  title,
  subtitle,
  showBack = false,
  backHref = '/',
  actionButton,
}) => {
  const { targetJob } = useInterview();

  return (
    <header className="px-8 py-5 border-b border-neutral-200/60 dark:border-neutral-800/60 bg-white/70 dark:bg-[#0A0A0C]/70 backdrop-blur-md sticky top-14 z-20 flex items-center justify-between">
      <div className="flex items-center gap-3">
        {showBack && (
          <Link
            href={backHref}
            className="p-1.5 -ml-2 rounded-lg text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100 hover:bg-neutral-100 dark:hover:bg-neutral-800/60 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
        )}
        <div>
          {title && (
            <h1 className="text-base font-semibold text-neutral-900 dark:text-neutral-100 tracking-tight">
              {title}
            </h1>
          )}
          {subtitle && (
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {actionButton || (
          <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              {targetJob.company} · {targetJob.title}
            </span>
          </div>
        )}
      </div>
    </header>
  );
};
