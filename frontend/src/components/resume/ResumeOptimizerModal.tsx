'use client';

import React, { useState } from 'react';
import { Sparkles, Check, Pencil } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { ModalShell } from '@/components/app/ModalShell';
import { PALETTE } from '@/lib/motion';

interface ResumeOptimizerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResumeOptimizerModal: React.FC<ResumeOptimizerModalProps> = ({ isOpen, onClose }) => {
  const { candidate, suggestions, applySuggestion, applyEditedSuggestion, dismissSuggestion, targetJob, matchAnalysis, matchStatus } =
    useInterview();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [customDraft, setCustomDraft] = useState<string>('');

  if (!isOpen) return null;

  const pendingCount = suggestions.filter((s) => s.status === 'pending').length;

  const handleStartEdit = (id: string, suggestedText: string) => {
    setEditingId(id);
    setCustomDraft(suggestedText);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setCustomDraft('');
  };

  const handleApplyEdited = (id: string) => {
    applyEditedSuggestion(id, customDraft);
    setEditingId(null);
    setCustomDraft('');
  };

  return (
    <ModalShell
      title="Resume Optimization"
      onClose={onClose}
      maxWidthClassName="max-w-5xl"
      headerExtra={
        <div className="flex items-center gap-4">
          <span
            className="text-[11px] font-medium px-2 py-0.5 rounded-full"
            style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.accent }}
          >
            {pendingCount} suggestions
          </span>
          <div className="text-xs flex items-center gap-1.5 font-medium" style={{ color: PALETTE.textSecondary }}>
            <span>Estimated Match:</span>
            <span className="font-bold" style={{ color: PALETTE.accentBright }}>
              {matchStatus === 'ready' ? `${matchAnalysis.score}%` : '—'}
            </span>
          </div>
        </div>
      }
      footer={
        <>
          <p className="text-xs mr-auto" style={{ color: PALETTE.textSecondary }}>
            Applying suggestions updates your resume context for upcoming interviews.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-[8px] text-xs font-semibold"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Done
          </button>
        </>
      }
    >
        <p className="px-6 pt-1 text-[11px]" style={{ color: PALETTE.textSecondary }}>
          Tailored for {targetJob.title} at {targetJob.company}
        </p>

        {/* Side-by-side Layout */}
        <div className="flex overflow-hidden" style={{ height: 'calc(85vh - 140px)' }}>
          {/* Left Column: Resume Highlights */}
          <div
            className="w-1/2 p-6 overflow-y-auto space-y-6"
            style={{ borderRight: `1px solid ${PALETTE.border}` }}
          >
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
                Resume Content
              </h4>
              <span className="text-[11px] font-mono" style={{ color: PALETTE.textSecondary }}>
                {candidate.resumeFileName}
              </span>
            </div>

            {/* Candidate Summary */}
            <div
              className="p-4 rounded-[10px] space-y-1.5"
              style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
            >
              <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: PALETTE.textSecondary }}>
                Professional Summary
              </span>
              <p className="text-xs leading-relaxed" style={{ color: PALETTE.textPrimary }}>
                {candidate.summary}
              </p>
            </div>

            {/* Work Experience */}
            <div className="space-y-4">
              <span className="text-[11px] font-semibold uppercase tracking-wide block" style={{ color: PALETTE.textSecondary }}>
                Work Experience
              </span>
              {candidate.experience.map((exp) => (
                <div
                  key={exp.id}
                  className="p-4 rounded-[10px] space-y-2"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                      {exp.role}
                    </span>
                    <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>{exp.company}</span>
                  </div>
                  <ul className="space-y-2 text-xs" style={{ color: PALETTE.textSecondary }}>
                    {exp.highlights.map((bullet, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: PALETTE.border }}></span>
                        <span className="leading-relaxed">{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: AI Optimization Suggestions */}
          <div className="w-1/2 p-6 overflow-y-auto space-y-4" style={{ backgroundColor: PALETTE.bg2 }}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
                AI Optimization Suggestions
              </h4>
              <span className="text-[11px] font-medium" style={{ color: PALETTE.accent }}>
                High Impact
              </span>
            </div>

            {suggestions.map((sug) => {
              const isApplied = sug.status === 'applied';
              const isDismissed = sug.status === 'dismissed';
              const isEditing = editingId === sug.id;

              const cardStyle = isApplied
                ? { backgroundColor: `${PALETTE.accent}14`, border: `1px solid ${PALETTE.accent}40` }
                : isDismissed
                ? { backgroundColor: PALETTE.surfaceFaint, border: '1px solid transparent', opacity: 0.4 }
                : { backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` };

              return (
                <div key={sug.id} className="p-5 rounded-[10px] transition-all" style={cardStyle}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <span
                        className="inline-block text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full"
                        style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textSecondary }}
                      >
                        {sug.category.replace('_', ' ')}
                      </span>
                      <h5 className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                        {sug.title}
                      </h5>
                    </div>

                    {isApplied && (
                      <span className="flex items-center gap-1 text-[11px] font-medium" style={{ color: PALETTE.accent }}>
                        <Check className="w-3.5 h-3.5" />
                        Applied
                      </span>
                    )}
                  </div>

                  <p className="text-[11px] mt-2 leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                    {sug.explanation}
                  </p>

                  {/* Before vs After — this is an AI-suggested rewrite, not a
                      verified fact; nothing here is applied to the resume
                      until the user explicitly clicks Apply (or edits it
                      first), never automatically. */}
                  <div className="mt-3 space-y-2 text-xs">
                    <div
                      className="p-2.5 rounded-[8px] line-through"
                      style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textSecondary }}
                    >
                      {sug.currentText}
                    </div>
                    {isEditing ? (
                      <textarea
                        rows={3}
                        autoFocus
                        value={customDraft}
                        onChange={(e) => setCustomDraft(e.target.value)}
                        className="w-full p-2.5 rounded-[8px] outline-none leading-relaxed"
                        style={{ backgroundColor: `${PALETTE.accent}0F`, color: PALETTE.textPrimary, border: `1px solid ${PALETTE.accent}60` }}
                      />
                    ) : (
                      <div
                        className="p-2.5 rounded-[8px]"
                        style={{ backgroundColor: `${PALETTE.accent}0F`, color: PALETTE.textPrimary, border: `1px solid ${PALETTE.accent}30` }}
                      >
                        <span className="font-semibold block text-[10px] uppercase tracking-wider mb-0.5" style={{ color: PALETTE.accent }}>
                          Proposed Replacement:
                        </span>
                        {sug.suggestedText}
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  {!isApplied && !isDismissed && (
                    <div className="flex items-center gap-2 mt-4 pt-3" style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                      {isEditing ? (
                        <>
                          <button
                            onClick={() => handleApplyEdited(sug.id)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs font-medium transition-colors"
                            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                          >
                            <Check className="w-3.5 h-3.5" />
                            <span>Apply Edited Version</span>
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="px-3 py-1.5 rounded-[8px] text-xs transition-colors hover:bg-white/[0.05]"
                            style={{ color: PALETTE.textSecondary }}
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => applySuggestion(sug.id)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs font-medium transition-colors"
                            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                          >
                            <Check className="w-3.5 h-3.5" />
                            <span>Apply</span>
                          </button>
                          <button
                            onClick={() => handleStartEdit(sug.id, sug.suggestedText)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.05]"
                            style={{ color: PALETTE.textSecondary }}
                          >
                            <Pencil className="w-3.5 h-3.5" />
                            <span>Edit</span>
                          </button>
                          <button
                            onClick={() => dismissSuggestion(sug.id)}
                            className="px-3 py-1.5 rounded-[8px] text-xs transition-colors hover:bg-white/[0.05]"
                            style={{ color: PALETTE.textSecondary }}
                          >
                            Dismiss
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
    </ModalShell>
  );
};
