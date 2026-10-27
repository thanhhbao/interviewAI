'use client';

import React, { useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  FileText,
  Upload,
  Sparkles,
  Edit3,
  Briefcase,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Plus,
  RefreshCw,
  Loader2,
  MinusCircle,
  X,
  Lock,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInterview } from '@/context/InterviewContext';
import { AppBackground } from '@/components/app/AppBackground';
import { FileRow } from '@/components/resume/FileRow';
import { MatchScoreBlock } from '@/components/resume/MatchScoreBlock';
import { ResumeOptimizerModal } from '@/components/resume/ResumeOptimizerModal';
import { EditProfileModal } from '@/components/resume/EditProfileModal';
import { JobEditorModal } from '@/components/resume/JobEditorModal';
import { PALETTE } from '@/lib/motion';

const RESUME_PARSE_LABEL: Record<string, string> = {
  uploading: 'Uploading resume...',
  reading: 'Reading resume...',
  extracting: 'Extracting candidate profile...',
};

const JOB_ANALYSIS_LABEL: Record<string, string> = {
  analyzing: 'Analyzing job description...',
  extracting_required: 'Extracting required skills...',
  extracting_preferred: 'Identifying preferred skills...',
};

const panelStyle = { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` };
const dividerStyle = { borderColor: PALETTE.borderTeal };

// Focus area → representative sample interview question
const FOCUS_TO_QUESTION: Record<string, string> = {
  'System Design': 'How would you design a distributed cache that handles 100k reads/sec?',
  'Distributed Systems': 'Tell me about a system you scaled under heavy traffic.',
  'Backend Architecture': 'Walk me through a microservices migration you led or contributed to.',
  'Behavioral': 'Describe a technical decision you disagreed with and how you handled it.',
  'Coding': 'How do you approach debugging a latency regression in a production service?',
  'Technical Leadership': 'How have you influenced engineering decisions without direct authority?',
  'API Design': 'What tradeoffs do you consider when designing a public-facing REST API?',
  'Database Engineering': 'How would you approach schema design for a write-heavy workload?',
  'Cloud Infrastructure': 'Describe how you have used infrastructure-as-code in a production environment.',
  'Testing & CI/CD': 'What does a healthy deployment pipeline look like to you?',
};

function sampleQuestions(focusAreas: string[]): string[] {
  const qs: string[] = [];
  for (const area of focusAreas) {
    const q = FOCUS_TO_QUESTION[area];
    if (q) qs.push(q);
    if (qs.length >= 3) break;
  }
  // Fallback generic questions if no matches found
  if (qs.length === 0) {
    return [
      'Tell me about a technically challenging project you led.',
      'Describe a time you had to make a difficult trade-off.',
      'How do you approach debugging a production incident?',
    ];
  }
  return qs;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
      {children}
    </span>
  );
}

function Pill({ children, variant = 'default' }: { children: React.ReactNode; variant?: 'required' | 'optional' | 'default' }) {
  const colors =
    variant === 'required'
      ? { color: '#d6b278', bg: 'rgba(214,178,120,0.08)', border: 'rgba(214,178,120,0.18)' }
      : variant === 'optional'
      ? { color: PALETTE.textSecondary, bg: 'rgba(255,255,255,0.04)', border: PALETTE.border }
      : { color: PALETTE.textSecondary, bg: 'rgba(255,255,255,0.04)', border: PALETTE.border };
  return (
    <span
      className="text-[10px] font-mono tracking-[0.1em] px-2 py-0.5 rounded-[5px]"
      style={{ color: colors.color, backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      {children}
    </span>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="px-2.5 py-1 rounded-[8px] text-[11px] font-medium"
      style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textPrimary, border: `1px solid ${PALETTE.border}` }}
    >
      {children}
    </span>
  );
}

export default function ResumeJobPage() {
  const router = useRouter();
  const {
    candidate,
    resumeStatus,
    resumeError,
    resumeParseStage,
    resumeAnalysis,
    uploadResume,
    targetJob,
    jobStatus,
    jobError,
    jobAnalysisStage,
    clearTargetJob,
    matchAnalysis,
    matchStatus,
    matchError,
    retryMatchAnalysis,
    hasInterviewContext,
  } = useInterview();

  const [isOptimizerOpen, setIsOptimizerOpen] = useState(false);
  const [isEditProfileOpen, setIsEditProfileOpen] = useState(false);
  const [jobModal, setJobModal] = useState<'add' | 'edit' | 'replace' | null>(null);
  const [isDraggingResume, setIsDraggingResume] = useState(false);

  const resumeInputRef = useRef<HTMLInputElement>(null);

  const handleResumeFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    uploadResume(file);
  };

  const triggerResumePicker = () => resumeInputRef.current?.click();

  const handleResumeDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingResume(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadResume(file);
  };

  const optimizerAvailable = resumeStatus === 'ready' && jobStatus === 'ready';
  const resumeReady = resumeStatus === 'ready';
  const jobReady = jobStatus === 'ready';

  const handleBuildInterview = () => {
    router.push('/setup');
  };

  const questions = resumeReady ? sampleQuestions(resumeAnalysis.inferredFocusAreas) : [];
  const focusAreas = jobReady
    ? matchAnalysis.focusAreas.slice(0, 4)
    : resumeReady
    ? resumeAnalysis.inferredFocusAreas.slice(0, 4)
    : [];

  return (
    <div className="relative flex-1 flex flex-col">
      <AppBackground />

      <input
        ref={resumeInputRef}
        type="file"
        accept=".pdf,.docx"
        className="hidden"
        onChange={handleResumeFileChange}
      />

      <main className="relative flex-1 max-w-6xl w-full mx-auto px-6 md:px-8 py-10 md:py-14 space-y-12">

        {/* ── PAGE HEADER ─────────────────────────────────────────────────── */}
        <div className="max-w-2xl space-y-3">
          <p className="text-[10px] font-mono tracking-[0.18em]" style={{ color: PALETTE.textSecondary }}>
            INTERVIEW PREPARATION
          </p>
          <h1 className="text-2xl md:text-[30px] font-semibold tracking-tight leading-[1.2]" style={{ color: PALETTE.textPrimary }}>
            Build your interview.
          </h1>
          <p className="text-sm leading-relaxed max-w-md" style={{ color: PALETTE.textSecondary }}>
            Give InterviewPal your experience and target role. We'll use them to shape
            the questions, difficulty, and focus of your interview.
          </p>
        </div>

        {/* ── TWO-COLUMN INPUT ────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-[55fr_45fr] gap-6 items-start">

          {/* LEFT: YOUR PROFILE ─────────────────────────────────────── */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <SectionLabel>Your Profile</SectionLabel>
              <Pill variant="required">REQUIRED</Pill>
            </div>

            <div className="rounded-[16px] overflow-hidden" style={panelStyle}>
              <AnimatePresence mode="wait" initial={false}>
                {resumeStatus === 'empty' || resumeStatus === 'error' ? (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="p-6 md:p-7 space-y-5"
                  >
                    <div>
                      <h2 className="text-sm font-semibold mb-1" style={{ color: PALETTE.textPrimary }}>Resume</h2>
                      <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                        Upload your resume so InterviewPal can understand your experience,
                        seniority, technical background, and skills.
                      </p>
                    </div>

                    <div
                      role="button"
                      tabIndex={0}
                      onClick={triggerResumePicker}
                      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && triggerResumePicker()}
                      onDragOver={(e) => { e.preventDefault(); setIsDraggingResume(true); }}
                      onDragLeave={() => setIsDraggingResume(false)}
                      onDrop={handleResumeDrop}
                      className="min-h-[160px] rounded-[10px] border border-dashed flex flex-col items-center justify-center text-center gap-3 px-6 cursor-pointer transition-colors"
                      style={{
                        borderColor: isDraggingResume ? PALETTE.accent : PALETTE.border,
                        backgroundColor: isDraggingResume ? `${PALETTE.accent}0D` : PALETTE.surface2,
                      }}
                    >
                      <div
                        className="w-9 h-9 rounded-[8px] flex items-center justify-center"
                        style={{ backgroundColor: PALETTE.surfaceFaint, color: isDraggingResume ? PALETTE.accent : PALETTE.textSecondary }}
                      >
                        <Upload className="w-4 h-4" />
                      </div>
                      <p className="text-xs font-medium" style={{ color: PALETTE.textPrimary }}>
                        {resumeStatus === 'error' ? 'Try again' : 'Drop your resume, or click to browse'}
                      </p>
                      <span className="text-[11px] font-mono" style={{ color: PALETTE.textSecondary }}>PDF / DOCX</span>
                      {resumeStatus === 'error' && (
                        <p className="text-[11px] leading-relaxed max-w-xs" style={{ color: '#e8a29c' }}>{resumeError}</p>
                      )}
                    </div>

                    <p className="text-[11px]" style={{ color: PALETTE.textSecondary }}>
                      Used only to build your interview context — never shared outside this session.
                    </p>
                  </motion.div>
                ) : resumeStatus === 'parsing' ? (
                  <motion.div
                    key="parsing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="p-6 md:p-7 flex flex-col items-center justify-center text-center space-y-3 py-16"
                  >
                    <Loader2 className="w-5 h-5 animate-spin" style={{ color: PALETTE.textSecondary }} />
                    <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                      {resumeParseStage ? RESUME_PARSE_LABEL[resumeParseStage] : 'Parsing resume...'}
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="p-6 md:p-7 space-y-6"
                  >
                    {/* File + status */}
                    <div className="flex items-center justify-between">
                      <FileRow fileName={candidate.resumeFileName} onReplace={triggerResumePicker} replaceLabel="Replace" />
                      <span className="text-[11px] font-mono shrink-0 ml-4" style={{ color: '#7ecba1' }}>Resume ready ✓</span>
                    </div>

                    {/* Candidate identity */}
                    <div>
                      <h3 className="text-base font-semibold" style={{ color: PALETTE.textPrimary }}>{candidate.name}</h3>
                      <p className="text-xs mt-0.5" style={{ color: PALETTE.textSecondary }}>{candidate.title}</p>
                    </div>

                    {/* Role + seniority */}
                    <div
                      className="p-3.5 rounded-[10px] grid grid-cols-2 gap-4"
                      style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.borderTeal}` }}
                    >
                      <div className="space-y-1">
                        <SectionLabel>Detected Role</SectionLabel>
                        <p className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>{resumeAnalysis.inferredRole}</p>
                      </div>
                      {resumeAnalysis.inferredSeniority && (
                        <div className="space-y-1">
                          <SectionLabel>Seniority</SectionLabel>
                          <p className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>{resumeAnalysis.inferredSeniority}</p>
                        </div>
                      )}
                    </div>

                    {/* Key signals */}
                    <div className="space-y-2">
                      <SectionLabel>Key signals</SectionLabel>
                      <div className="flex flex-wrap gap-1.5">
                        {candidate.skills.slice(0, 6).map((skill) => (
                          <Chip key={skill}>{skill}</Chip>
                        ))}
                      </div>
                    </div>

                    {/* Recent Experience */}
                    <div className="space-y-2.5">
                      <SectionLabel>Experience</SectionLabel>
                      <div className="space-y-2">
                        {candidate.experience.slice(0, 2).map((exp) => (
                          <div
                            key={exp.id}
                            className="p-3 rounded-[10px] text-xs"
                            style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                          >
                            <div className="flex items-center justify-between mb-0.5">
                              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>{exp.role}</span>
                              <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>{exp.duration}</span>
                            </div>
                            <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>{exp.company}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    <div
                      className="flex flex-wrap items-center gap-2 pt-4 text-xs"
                      style={{ borderTop: `1px solid ${dividerStyle.borderColor}` }}
                    >
                      <button
                        onClick={() => setIsEditProfileOpen(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] transition-colors hover:bg-white/[0.04]"
                        style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                        <span>Edit Profile</span>
                      </button>
                      <button
                        onClick={triggerResumePicker}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] transition-colors hover:bg-white/[0.04]"
                        style={{ color: PALETTE.textSecondary }}
                      >
                        <Upload className="w-3.5 h-3.5" />
                        <span>Replace Resume</span>
                      </button>
                      <button
                        onClick={() => setIsOptimizerOpen(true)}
                        disabled={!optimizerAvailable}
                        title={optimizerAvailable ? undefined : 'Add a target job first to receive tailored resume suggestions.'}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] font-medium transition-colors ml-auto disabled:opacity-40 disabled:pointer-events-none hover:bg-white/[0.04]"
                        style={{
                          border: `1px solid ${optimizerAvailable ? `${PALETTE.accent}50` : PALETTE.border}`,
                          color: optimizerAvailable ? PALETTE.accent : PALETTE.textSecondary,
                        }}
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Optimize Resume</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* RIGHT: TARGET INTERVIEW ─────────────────────────────────── */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <SectionLabel>Target Interview</SectionLabel>
              <Pill variant="optional">OPTIONAL</Pill>
            </div>

            <div className="rounded-[16px] overflow-hidden" style={panelStyle}>
              <AnimatePresence mode="wait" initial={false}>
                {jobStatus === 'empty' || jobStatus === 'error' ? (
                  <motion.div
                    key="job-empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="p-6 md:p-7 flex flex-col items-center justify-center text-center space-y-4 py-14"
                  >
                    <div
                      className="w-9 h-9 rounded-[8px] flex items-center justify-center"
                      style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textSecondary }}
                    >
                      <Briefcase className="w-4 h-4" />
                    </div>
                    <div className="space-y-1.5 max-w-xs">
                      <h2 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>Target role</h2>
                      {jobStatus === 'error' ? (
                        <p className="text-xs leading-relaxed" style={{ color: '#e8a29c' }}>{jobError}</p>
                      ) : (
                        <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                          Add a target role to make the interview more specific — tailored
                          questions, skill gaps, and role-focused difficulty.
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => setJobModal(jobStatus === 'error' ? 'replace' : 'add')}
                      className="inline-flex items-center gap-1.5 px-4 py-2 rounded-[8px] text-xs font-semibold transition-colors hover:bg-white/[0.04]"
                      style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                    >
                      <Plus className="w-3.5 h-3.5" />
                      <span>{jobStatus === 'error' ? 'Try again' : 'Add target role'}</span>
                    </button>
                  </motion.div>
                ) : jobStatus === 'analyzing' ? (
                  <motion.div
                    key="job-analyzing"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="p-6 md:p-7 flex flex-col items-center justify-center text-center space-y-3 py-16"
                  >
                    <Loader2 className="w-5 h-5 animate-spin" style={{ color: PALETTE.textSecondary }} />
                    <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                      {jobAnalysisStage ? JOB_ANALYSIS_LABEL[jobAnalysisStage] : 'Analyzing job description...'}
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="job-ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="p-6 md:p-7 space-y-5"
                  >
                    {/* Role identity */}
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h2 className="text-sm font-semibold leading-tight" style={{ color: PALETTE.textPrimary }}>
                          {targetJob.title}
                        </h2>
                        <p className="text-[12px] mt-0.5" style={{ color: PALETTE.textSecondary }}>
                          {targetJob.company}{targetJob.location ? ` · ${targetJob.location}` : ''}
                        </p>
                      </div>
                      <span
                        className="px-2.5 py-1 rounded-full text-[10px] font-semibold shrink-0"
                        style={{ backgroundColor: `${PALETTE.accent}1F`, color: PALETTE.accent }}
                      >
                        Active
                      </span>
                    </div>

                    {/* Interview focus signals from this role */}
                    <div className="space-y-2">
                      <SectionLabel>Interview focus</SectionLabel>
                      <ul className="space-y-1.5">
                        {targetJob.requiredSkills.slice(0, 4).map((req, i) => (
                          <li key={i} className="flex items-center gap-2 text-xs" style={{ color: PALETTE.textPrimary }}>
                            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: PALETTE.accent }} />
                            <span>{req}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Preferred (condensed) */}
                    {targetJob.preferredSkills.length > 0 && (
                      <div className="space-y-2">
                        <SectionLabel>Nice-to-have</SectionLabel>
                        <div className="flex flex-wrap gap-1.5">
                          {targetJob.preferredSkills.slice(0, 4).map((s, i) => (
                            <Chip key={i}>{s}</Chip>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-2 pt-4 text-xs" style={{ borderTop: `1px solid ${dividerStyle.borderColor}` }}>
                      <button
                        onClick={() => setJobModal('edit')}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] transition-colors hover:bg-white/[0.04]"
                        style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                        <span>Edit</span>
                      </button>
                      <button
                        onClick={() => setJobModal('replace')}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] transition-colors hover:bg-white/[0.04]"
                        style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>Replace</span>
                      </button>
                      <button
                        onClick={clearTargetJob}
                        title="Remove target role and switch back to a resume-based interview"
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] transition-colors ml-auto hover:text-red-400 hover:bg-red-500/10"
                        style={{ color: PALETTE.textSecondary }}
                      >
                        <X className="w-3.5 h-3.5" />
                        <span>Remove</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* ── INTERVIEW PREVIEW ────────────────────────────────────────────── */}
        <AnimatePresence mode="wait" initial={false}>
          {!resumeReady ? (
            /* Locked / waiting state */
            <motion.section
              key="preview-locked"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3 }}
              className="rounded-[16px] px-8 py-10 flex flex-col items-center justify-center text-center space-y-3"
              style={{ border: `1px dashed ${PALETTE.border}` }}
            >
              <Lock className="w-4 h-4" style={{ color: PALETTE.textSecondary, opacity: 0.5 }} />
              <p className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>
                Add your resume to generate a personalized interview plan.
              </p>
              <p className="text-xs max-w-sm" style={{ color: PALETTE.textSecondary }}>
                Your resume lets InterviewPal tailor the question set, difficulty, and focus
                to match your background.
              </p>
            </motion.section>

          ) : jobReady && matchStatus !== 'idle' ? (
            /* Resume + Job: Match analysis */
            <motion.section
              key="preview-match"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3 }}
              className="rounded-[16px] space-y-6 p-7 md:p-8"
              style={panelStyle}
            >
              <div className="pb-4 space-y-1" style={{ borderBottom: `1px solid ${dividerStyle.borderColor}` }}>
                <h3 className="text-lg font-semibold" style={{ color: PALETTE.textPrimary }}>
                  Your interview is taking shape.
                </h3>
                <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                  Resume + target role aligned — interview questions will reflect this context.
                </p>
              </div>

              {matchStatus === 'analyzing' ? (
                <div className="flex flex-col items-center justify-center text-center space-y-3 py-10">
                  <Loader2 className="w-5 h-5 animate-spin" style={{ color: PALETTE.textSecondary }} />
                  <p className="text-xs" style={{ color: PALETTE.textSecondary }}>Aligning resume with target role...</p>
                </div>
              ) : matchStatus === 'error' ? (
                <div className="flex flex-col items-center justify-center text-center space-y-3 py-10">
                  <AlertTriangle className="w-5 h-5" style={{ color: '#e8a29c' }} />
                  <p className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>Match analysis unavailable</p>
                  <p className="text-xs max-w-sm leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                    You can retry the analysis or continue with your resume-based interview.
                  </p>
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={retryMatchAnalysis}
                      className="px-4 py-2 rounded-[8px] text-xs font-semibold"
                      style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                    >
                      Retry Match
                    </button>
                    <button
                      onClick={handleBuildInterview}
                      className="px-4 py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.04]"
                      style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                    >
                      Continue with Resume
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <MatchScoreBlock score={matchAnalysis.score} />

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                        Matched ({matchAnalysis.matchedSkills.length})
                      </h4>
                      <div className="flex flex-wrap gap-1.5">
                        {matchAnalysis.matchedSkills.length === 0 ? (
                          <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>None yet</span>
                        ) : (
                          matchAnalysis.matchedSkills.map((s) => (
                            <span
                              key={s}
                              className="px-2.5 py-1 rounded-[8px] text-xs font-medium"
                              style={{ backgroundColor: `${PALETTE.accent}1A`, color: PALETTE.accent }}
                            >
                              {s}
                            </span>
                          ))
                        )}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                        Partially matched ({matchAnalysis.partiallyMatchedSkills.length})
                      </h4>
                      <div className="flex flex-wrap gap-1.5">
                        {matchAnalysis.partiallyMatchedSkills.length === 0 ? (
                          <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>None</span>
                        ) : (
                          matchAnalysis.partiallyMatchedSkills.map((s) => (
                            <span
                              key={s}
                              className="px-2.5 py-1 rounded-[8px] text-xs font-medium"
                              style={{ backgroundColor: 'rgba(214,178,120,0.12)', color: '#d6b278' }}
                            >
                              {s}
                            </span>
                          ))
                        )}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center gap-1.5">
                        <MinusCircle className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
                        <h4 className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                          Growth areas ({matchAnalysis.missingSkills.length})
                        </h4>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {matchAnalysis.missingSkills.length === 0 ? (
                          <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>None</span>
                        ) : (
                          matchAnalysis.missingSkills.map((s) => (
                            <span
                              key={s}
                              className="px-2.5 py-1 rounded-[8px] text-xs font-medium"
                              style={{ backgroundColor: 'rgba(224,150,120,0.12)', color: '#e09678' }}
                            >
                              {s}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Focus areas — feed directly into interview generation */}
                  <div className="pt-6 grid grid-cols-1 md:grid-cols-2 gap-6" style={{ borderTop: `1px solid ${dividerStyle.borderColor}` }}>
                    <div className="space-y-2.5">
                      <SectionLabel>Interview focus</SectionLabel>
                      <ul className="space-y-1.5 text-sm">
                        {matchAnalysis.focusAreas.map((area, i) => (
                          <li key={i} className="flex items-start gap-2" style={{ color: PALETTE.textPrimary }}>
                            <span className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: PALETTE.accent }} />
                            <span>{area}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="space-y-2.5">
                      <SectionLabel>Potential questions</SectionLabel>
                      <ul className="space-y-2">
                        {sampleQuestions(matchAnalysis.focusAreas).map((q, i) => (
                          <li
                            key={i}
                            className="text-xs leading-relaxed pl-3"
                            style={{ color: PALETTE.textSecondary, borderLeft: `2px solid ${PALETTE.border}` }}
                          >
                            &ldquo;{q}&rdquo;
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </>
              )}
            </motion.section>

          ) : (
            /* Resume only: live interview preview */
            <motion.section
              key="preview-resume"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3 }}
              className="rounded-[16px] space-y-6 p-7 md:p-8"
              style={panelStyle}
            >
              <div className="pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3" style={{ borderBottom: `1px solid ${dividerStyle.borderColor}` }}>
                <div className="space-y-0.5">
                  <h3 className="text-lg font-semibold" style={{ color: PALETTE.textPrimary }}>
                    Your interview is taking shape.
                  </h3>
                  <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                    Resume-based context · Add a target role to focus further
                  </p>
                </div>
                <button
                  onClick={() => setJobModal('add')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs font-medium shrink-0 transition-colors hover:bg-white/[0.04]"
                  style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add target role
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Detected role / seniority */}
                <div className="space-y-1.5">
                  <SectionLabel>Detected role</SectionLabel>
                  <p className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>{resumeAnalysis.inferredRole}</p>
                  {resumeAnalysis.inferredSeniority && (
                    <p className="text-xs" style={{ color: PALETTE.textSecondary }}>{resumeAnalysis.inferredSeniority} level</p>
                  )}
                </div>

                {/* Interview focus */}
                <div className="space-y-2">
                  <SectionLabel>Interview focus</SectionLabel>
                  <ul className="space-y-1.5">
                    {focusAreas.map((area, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs" style={{ color: PALETTE.textPrimary }}>
                        <span className="w-1 h-1 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: PALETTE.accent }} />
                        <span>{area}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Core skills */}
                <div className="space-y-2">
                  <SectionLabel>Core strengths</SectionLabel>
                  <div className="flex flex-wrap gap-1.5">
                    {resumeAnalysis.coreSkills.map((s) => <Chip key={s}>{s}</Chip>)}
                  </div>
                </div>
              </div>

              {/* Potential questions */}
              <div className="pt-4 space-y-3" style={{ borderTop: `1px solid ${dividerStyle.borderColor}` }}>
                <SectionLabel>Potential questions</SectionLabel>
                <ul className="space-y-2">
                  {questions.map((q, i) => (
                    <li
                      key={i}
                      className="text-xs leading-relaxed pl-3"
                      style={{ color: PALETTE.textSecondary, borderLeft: `2px solid ${PALETTE.border}` }}
                    >
                      &ldquo;{q}&rdquo;
                    </li>
                  ))}
                </ul>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* ── PRIMARY CTA ──────────────────────────────────────────────────── */}
        <section className="pt-2 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div className="space-y-1">
            <p className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
              {hasInterviewContext ? 'Ready to build your interview.' : 'Add your resume to continue.'}
            </p>
            <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
              Resume required · Target role optional
            </p>
          </div>
          <button
            onClick={handleBuildInterview}
            disabled={!hasInterviewContext}
            className="px-6 py-3 rounded-[10px] text-sm font-semibold shrink-0 flex items-center gap-2 transition-opacity hover:opacity-90 active:opacity-80 disabled:opacity-30 disabled:pointer-events-none"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            <span>Build my interview</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </section>

      </main>

      {/* Modals — unchanged */}
      <ResumeOptimizerModal isOpen={isOptimizerOpen} onClose={() => setIsOptimizerOpen(false)} />
      <EditProfileModal isOpen={isEditProfileOpen} onClose={() => setIsEditProfileOpen(false)} />
      <JobEditorModal
        isOpen={jobModal !== null}
        mode={jobModal || 'add'}
        onClose={() => setJobModal(null)}
      />
    </div>
  );
}
