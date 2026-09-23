'use client';

import React, { useState } from 'react';
import {
  Sparkles,
  X,
  Check,
  TrendingUp,
  AlertCircle,
  FileCheck,
  ArrowRight,
  Plus,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';

interface ResumeOptimizerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResumeOptimizerModal: React.FC<ResumeOptimizerModalProps> = ({ isOpen, onClose }) => {
  const { candidate, suggestions, applySuggestion, dismissSuggestion, targetJob } = useInterview();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [customDraft, setCustomDraft] = useState<string>('');

  if (!isOpen) return null;

  const pendingCount = suggestions.filter((s) => s.status === 'pending').length;

  const handleStartEdit = (id: string, currentText: string) => {
    setEditingId(id);
    setCustomDraft(currentText);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-5xl h-[85vh] bg-white dark:bg-[#111114] border border-neutral-200/80 dark:border-neutral-800/80 rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-scale-up">
        {/* Header */}
        <div className="px-6 py-4 border-b border-neutral-200/80 dark:border-neutral-800/80 flex items-center justify-between bg-neutral-50/50 dark:bg-neutral-900/30">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                Resume Optimization
                <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-300">
                  {pendingCount} suggestions
                </span>
              </h3>
              <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
                Tailored for {targetJob.title} at {targetJob.company}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-xs text-neutral-500 dark:text-neutral-400 flex items-center gap-1.5 font-medium">
              <span>Match Score:</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">
                {targetJob.matchScore}%
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Side-by-side Layout */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Column: Resume Highlights */}
          <div className="w-1/2 border-r border-neutral-200/80 dark:border-neutral-800/80 p-6 overflow-y-auto space-y-6">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
                Resume Content
              </h4>
              <span className="text-[11px] text-neutral-400 font-mono">
                {candidate.resumeFileName}
              </span>
            </div>

            {/* Candidate Summary */}
            <div className="p-4 rounded-2xl bg-neutral-50 dark:bg-neutral-900/40 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1.5">
              <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wide">
                Professional Summary
              </span>
              <p className="text-xs text-neutral-700 dark:text-neutral-300 leading-relaxed">
                {candidate.summary}
              </p>
            </div>

            {/* Work Experience */}
            <div className="space-y-4">
              <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wide block">
                Work Experience
              </span>
              {candidate.experience.map((exp) => (
                <div
                  key={exp.id}
                  className="p-4 rounded-2xl border border-neutral-200/60 dark:border-neutral-800/60 space-y-2 bg-white dark:bg-[#131316]"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                      {exp.role}
                    </span>
                    <span className="text-[11px] text-neutral-400">{exp.company}</span>
                  </div>
                  <ul className="space-y-2 text-xs text-neutral-600 dark:text-neutral-300">
                    {exp.highlights.map((bullet, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-neutral-300 dark:bg-neutral-600 mt-1.5 shrink-0"></span>
                        <span className="leading-relaxed">{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: AI Optimization Suggestions */}
          <div className="w-1/2 p-6 overflow-y-auto space-y-4 bg-neutral-50/40 dark:bg-neutral-950/20">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
                AI Optimization Suggestions
              </h4>
              <span className="text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
                High Impact
              </span>
            </div>

            {suggestions.map((sug) => {
              const isApplied = sug.status === 'applied';
              const isDismissed = sug.status === 'dismissed';

              return (
                <div
                  key={sug.id}
                  className={`p-5 rounded-2xl border transition-all ${
                    isApplied
                      ? 'bg-emerald-50/30 border-emerald-200 dark:border-emerald-950/60'
                      : isDismissed
                      ? 'opacity-40 bg-neutral-100 dark:bg-neutral-900 border-transparent'
                      : 'bg-white dark:bg-[#141417] border-neutral-200/80 dark:border-neutral-800/80 shadow-xs'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <span className="inline-block text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400">
                        {sug.category.replace('_', ' ')}
                      </span>
                      <h5 className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                        {sug.title}
                      </h5>
                    </div>

                    {isApplied && (
                      <span className="flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
                        <Check className="w-3.5 h-3.5" />
                        Applied (+4% match)
                      </span>
                    )}
                  </div>

                  <p className="text-[11px] text-neutral-500 dark:text-neutral-400 mt-2 leading-relaxed">
                    {sug.explanation}
                  </p>

                  {/* Before vs After */}
                  <div className="mt-3 space-y-2 text-xs">
                    <div className="p-2.5 rounded-xl bg-neutral-100/60 dark:bg-neutral-900/60 text-neutral-500 line-through">
                      {sug.currentText}
                    </div>
                    <div className="p-2.5 rounded-xl bg-indigo-50/70 dark:bg-indigo-950/40 text-neutral-900 dark:text-neutral-100 border border-indigo-100 dark:border-indigo-900/50">
                      <span className="font-semibold text-indigo-600 dark:text-indigo-400 block text-[10px] uppercase tracking-wider mb-0.5">
                        Proposed Replacement:
                      </span>
                      {sug.suggestedText}
                    </div>
                  </div>

                  {/* Actions */}
                  {!isApplied && !isDismissed && (
                    <div className="flex items-center gap-2 mt-4 pt-3 border-t border-neutral-100 dark:border-neutral-800">
                      <button
                        onClick={() => applySuggestion(sug.id)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium shadow-xs transition-colors"
                      >
                        <Check className="w-3.5 h-3.5" />
                        <span>Apply Suggestion</span>
                      </button>

                      <button
                        onClick={() => dismissSuggestion(sug.id)}
                        className="px-3 py-1.5 rounded-xl text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-xs transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-neutral-200/80 dark:border-neutral-800/80 bg-neutral-50/50 dark:bg-neutral-900/30 flex items-center justify-between">
          <p className="text-xs text-neutral-500">
            Applying suggestions updates your resume context for upcoming interviews.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-neutral-900 dark:bg-white text-white dark:text-neutral-950 text-xs font-semibold"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
