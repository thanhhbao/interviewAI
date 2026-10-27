'use client';

import React, { useEffect, useState } from 'react';
import { Upload, FileText } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { JobInput } from '@/lib/types/interview';
import { ModalShell } from '@/components/app/ModalShell';
import { PALETTE } from '@/lib/motion';

interface JobEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  // 'add'/'replace' start from a blank form; 'edit' prefills from the
  // current target job. All three submit through the same analyze flow —
  // the only difference is what the form starts with.
  mode: 'add' | 'edit' | 'replace';
}

const EMPTY_INPUT: JobInput = { title: '', company: '', location: '', description: '' };

export const JobEditorModal: React.FC<JobEditorModalProps> = ({ isOpen, onClose, mode }) => {
  const { targetJob, jobStatus, submitJob } = useInterview();
  const [draft, setDraft] = useState<JobInput>(EMPTY_INPUT);
  const [jdFileName, setJdFileName] = useState<string | null>(null);
  const [errors, setErrors] = useState<{ title?: string; description?: string }>({});

  useEffect(() => {
    if (isOpen) {
      setDraft(
        mode === 'edit' && jobStatus === 'ready'
          ? {
              title: targetJob.title,
              company: targetJob.company,
              location: targetJob.location,
              description: targetJob.description,
            }
          : EMPTY_INPUT
      );
      setJdFileName(null);
      setErrors({});
    }
  }, [isOpen, mode, targetJob, jobStatus]);

  if (!isOpen) return null;

  const handleJdFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setJdFileName(file.name);
    // Real JD file parsing is out of scope for this phase — the file name is
    // captured as a UI affordance only; the pasted description text below
    // is what actually gets analyzed.
  };

  const handleAnalyze = () => {
    const nextErrors: { title?: string; description?: string } = {};
    if (!draft.title.trim()) nextErrors.title = 'Please add a job title.';
    if (!draft.description.trim()) nextErrors.description = 'Please paste a job description.';
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      return;
    }
    submitJob(draft);
    onClose();
  };

  const inputClass = 'w-full rounded-[8px] px-3 py-2 outline-none transition-colors text-xs';
  const inputStyle = {
    backgroundColor: PALETTE.surface2,
    border: `1px solid ${PALETTE.border}`,
    color: PALETTE.textPrimary,
  };
  const labelStyle = { color: PALETTE.textSecondary };

  const title = mode === 'edit' ? 'Edit Job Description' : mode === 'replace' ? 'Replace Job Description' : 'Add Job Description';

  return (
    <ModalShell
      title={title}
      onClose={onClose}
      footer={
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.05]"
            style={{ color: PALETTE.textSecondary }}
          >
            Cancel
          </button>
          <button
            onClick={handleAnalyze}
            className="px-4 py-2 rounded-[8px] text-xs font-semibold"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Analyze Job
          </button>
        </>
      }
    >
        <div className="p-6 space-y-4 text-xs">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="block" style={labelStyle}>Job Title</label>
              <input
                className={inputClass}
                style={inputStyle}
                value={draft.title}
                onChange={(e) => setDraft((p) => ({ ...p, title: e.target.value }))}
                placeholder="e.g. Senior Backend Engineer"
              />
              {errors.title && <p className="text-red-400 text-[11px]">{errors.title}</p>}
            </div>
            <div className="space-y-1">
              <label className="block" style={labelStyle}>Company</label>
              <input
                className={inputClass}
                style={inputStyle}
                value={draft.company}
                onChange={(e) => setDraft((p) => ({ ...p, company: e.target.value }))}
                placeholder="Optional"
              />
            </div>
          </div>

          <div className="space-y-1">
            <label className="block" style={labelStyle}>Location (optional)</label>
            <input
              className={inputClass}
              style={inputStyle}
              value={draft.location}
              onChange={(e) => setDraft((p) => ({ ...p, location: e.target.value }))}
              placeholder="e.g. Remote, San Francisco CA"
            />
          </div>

          <div className="space-y-1">
            <label className="block" style={labelStyle}>Job Description</label>
            <textarea
              rows={8}
              className={`${inputClass} leading-relaxed`}
              style={inputStyle}
              value={draft.description}
              onChange={(e) => setDraft((p) => ({ ...p, description: e.target.value }))}
              placeholder="Paste the job description here..."
            />
            {errors.description && <p className="text-red-400 text-[11px]">{errors.description}</p>}
          </div>

          <div className="space-y-1">
            <label className="block" style={labelStyle}>Upload JD file (optional)</label>
            <label
              className="flex items-center gap-2 px-3 py-2 rounded-[8px] border border-dashed cursor-pointer transition-colors w-fit"
              style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{jdFileName || 'Choose file'}</span>
              <input type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={handleJdFile} />
            </label>
            {jdFileName && (
              <p className="text-[11px] flex items-center gap-1 mt-1" style={{ color: PALETTE.textSecondary }}>
                <FileText className="w-3 h-3" /> The pasted description above is what gets analyzed.
              </p>
            )}
          </div>
        </div>
    </ModalShell>
  );
};
