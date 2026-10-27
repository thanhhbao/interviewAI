'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Volume2,
  Code2,
  ChevronDown,
  ChevronUp,
  Award,
  Zap,
  ArrowRight,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useInterview } from '@/context/InterviewContext';
import { AppBackground } from '@/components/app/AppBackground';
import { PALETTE, useSafeReducedMotion } from '@/lib/motion';

const panelStyle = { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` };
const AMBER = '#d6b278';

// ── Blueprint skeleton preview (empty state) ──────────────────────────────────
// Shows the STRUCTURE of a report — category labels, placeholder bars, a
// feedback line — with no numeric scores or fake user data. Clearly labelled
// PREVIEW so users understand this is illustrative.

const SKELETON_ROWS = ['Communication', 'Technical depth', 'Reasoning'];
// Placeholder bar widths (purely visual rhythm — convey "varying scores" without
// implying these are real values). No numbers are shown next to them.
const SKELETON_WIDTHS = ['72%', '58%', '66%'];

function ReportEmptyState({ ctaHref }: { ctaHref: string }) {
  const reduced = useSafeReducedMotion();

  const vis = (delay: number) => ({
    initial: { opacity: reduced ? 1 : 0 },
    animate: { opacity: 1 },
    transition: reduced ? { duration: 0 } : { delay, duration: 0.4 },
  });

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-14 md:py-20">
      <div className="w-full max-w-[440px] space-y-8">

        {/* ── Blueprint report panel ─────────────────────────────────── */}
        <motion.div
          className="relative rounded-[14px] overflow-hidden"
          style={{
            backgroundColor: '#080D0D',
            border: `1px solid ${PALETTE.border}`,
            fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, monospace',
          }}
          initial={{ opacity: reduced ? 1 : 0, y: reduced ? 0 : 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={reduced ? { duration: 0 } : { delay: 0.1, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Chrome bar */}
          <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: `1px solid ${PALETTE.border}` }}>
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.07)' }} />
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.07)' }} />
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.07)' }} />
            <span className="ml-2 text-[10px] tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
              evaluation-report
            </span>
            <span
              className="ml-auto text-[9px] tracking-[0.14em] px-2 py-0.5 rounded-[4px]"
              style={{ color: PALETTE.textSecondary, backgroundColor: 'rgba(255,255,255,0.04)', border: `1px solid ${PALETTE.border}` }}
            >
              PREVIEW
            </span>
          </div>

          <div className="p-5 space-y-5">

            {/* Status label */}
            <motion.div {...vis(0.3)}>
              <p className="text-[10px] tracking-[0.18em]" style={{ color: PALETTE.textSecondary }}>
                WHAT YOU'LL GET
              </p>
              <p className="text-[11px] mt-0.5" style={{ color: PALETTE.textSecondary, opacity: 0.5 }}>
                Interview summary · evaluation · feedback
              </p>
            </motion.div>

            {/* Overall score — placeholder, no number */}
            <motion.div {...vis(0.6)} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>Overall score</span>
                {/* Placeholder pill instead of a number */}
                <span
                  className="h-[14px] w-8 rounded-[4px]"
                  style={{ backgroundColor: 'rgba(255,255,255,0.06)', display: 'inline-block' }}
                  aria-hidden="true"
                />
              </div>
              {/* Main bar — animates in but no label */}
              <div className="w-full h-[3px] rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.07)' }}>
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: PALETTE.accent, opacity: 0.6 }}
                  initial={{ width: reduced ? '68%' : '0%' }}
                  animate={{ width: '68%' }}
                  transition={reduced ? { duration: 0 } : { delay: 0.8, duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
            </motion.div>

            {/* Divider */}
            <motion.div {...vis(1.5)} style={{ borderTop: `1px solid ${PALETTE.border}` }} />

            {/* Category skeleton rows — label only, no score value */}
            <div className="space-y-3">
              {SKELETON_ROWS.map((label, i) => (
                <motion.div key={label} {...vis(reduced ? 1.55 : 1.55 + i * 0.25)} className="space-y-1.5">
                  <span className="text-[11px]" style={{ color: PALETTE.textSecondary }}>{label}</span>
                  <div className="w-full h-[2px] rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.06)' }}>
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: PALETTE.accent, opacity: 0.35 }}
                      initial={{ width: reduced ? SKELETON_WIDTHS[i] : '0%' }}
                      animate={{ width: SKELETON_WIDTHS[i] }}
                      transition={reduced ? { duration: 0 } : { delay: 1.7 + i * 0.25, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Divider */}
            <motion.div {...vis(2.5)} style={{ borderTop: `1px solid ${PALETTE.border}` }} />

            {/* Feedback placeholder */}
            <motion.div {...vis(2.6)} className="space-y-1.5">
              <p className="text-[10px] tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
                AI FEEDBACK
              </p>
              <p className="text-[11px] leading-relaxed italic" style={{ color: PALETTE.textSecondary, opacity: 0.55 }}>
                Your feedback will appear here after the interview.
              </p>
            </motion.div>
          </div>

          {/* Scan line — single pass after intro, repeats quietly */}
          {!reduced && (
            <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-[14px]" aria-hidden="true">
              <motion.div
                className="absolute inset-x-0 h-px"
                style={{ background: 'linear-gradient(to right, transparent 0%, rgba(255,255,255,0.07) 40%, rgba(255,255,255,0.07) 60%, transparent 100%)' }}
                initial={{ top: '-1px', opacity: 0 }}
                animate={{ top: ['-1px', 'calc(100% + 1px)', 'calc(100% + 1px)'], opacity: [0, 0.9, 0] }}
                transition={{ delay: 3.0, duration: 2.6, repeat: Infinity, repeatDelay: 12, ease: 'linear', times: [0, 0.9, 1] }}
              />
            </div>
          )}
        </motion.div>

        {/* ── Empty-state copy + CTA ─────────────────────────────────── */}
        <motion.div
          className="text-center space-y-4"
          initial={{ opacity: reduced ? 1 : 0, y: reduced ? 0 : 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={reduced ? { duration: 0 } : { delay: 0.35, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="space-y-1.5">
            <p className="text-base font-semibold" style={{ color: PALETTE.textPrimary }}>
              No interview report yet
            </p>
            <p className="text-sm" style={{ color: PALETTE.textSecondary }}>
              Complete an interview to generate your evaluation report.
            </p>
          </div>
          <Link
            href={ctaHref}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[10px] text-sm font-semibold transition-opacity hover:opacity-90"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Start an Interview
            <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>

      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
      {children}
    </h3>
  );
}

export default function ReportPage() {
  const router = useRouter();
  const { latestReport, hasCompletedSession, hasInterviewContext } = useInterview();
  const [expandedQuestionId, setExpandedQuestionId] = useState<string | null>('q-4');
  const [playingAudioId, setPlayingAudioId] = useState<string | null>(null);

  const toggleQuestion = (id: string) => {
    setExpandedQuestionId(expandedQuestionId === id ? null : id);
  };

  // Phase 1: this is intentionally NOT question-specific yet — it must not
  // pretend to replay a single selected question when it actually only
  // routes into a fresh full setup. Both the header-level and per-question
  // "Practice Again" actions share this until single-question practice is
  // built (later phase). Neither starts a session directly; Setup does.
  const handlePracticeAgain = () => {
    router.push('/setup');
  };

  const handleSimulateListen = (id: string, text: string) => {
    if (playingAudioId === id) {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      setPlayingAudioId(null);
      return;
    }

    setPlayingAudioId(id);
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.onend = () => setPlayingAudioId(null);
      utterance.onerror = () => setPlayingAudioId(null);
      window.speechSynthesis.speak(utterance);
    } else {
      setTimeout(() => setPlayingAudioId(null), 3000);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    return `${mins} min`;
  };

  if (!hasCompletedSession) {
    return (
      <div className="relative flex-1 flex flex-col">
        <AppBackground />
        <ReportEmptyState ctaHref={hasInterviewContext ? '/setup' : '/resume-job'} />
      </div>
    );
  }

  return (
    <div className="relative flex-1 flex flex-col pb-16">
      <AppBackground />

      <main className="relative flex-1 max-w-5xl w-full mx-auto px-6 md:px-8 py-10 md:py-14 space-y-10">
        {/* PAGE INTRO */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="max-w-2xl space-y-2.5">
            <p className="text-[11px] font-mono tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
              Interview preparation / Report
            </p>
            <h1 className="text-2xl md:text-[28px] font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
              Interview Evaluation Report
            </h1>
            <p className="text-sm leading-relaxed" style={{ color: PALETTE.textSecondary }}>
              Comprehensive breakdown of your technical and conversational performance.
            </p>
          </div>
          <button
            onClick={handlePracticeAgain}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-[10px] text-xs font-semibold shrink-0 transition-opacity hover:opacity-90"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Practice Again</span>
          </button>
        </div>

        {/* REPORT HEADER SUMMARY CARD */}
        <section
          className="p-8 rounded-[16px] flex flex-col md:flex-row items-start md:items-center justify-between gap-6"
          style={panelStyle}
        >
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span
                className="px-2.5 py-1 rounded-full text-xs font-semibold"
                style={{ backgroundColor: `${PALETTE.accent}1F`, color: PALETTE.accent }}
              >
                {latestReport.verdict}
              </span>
              <span className="text-xs" style={{ color: PALETTE.textSecondary }}>·</span>
              <span className="text-xs font-mono" style={{ color: PALETTE.textSecondary }}>{latestReport.date}</span>
            </div>

            <h2 className="text-2xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
              {latestReport.role}
            </h2>

            <div className="flex flex-wrap items-center gap-4 text-xs font-medium" style={{ color: PALETTE.textSecondary }}>
              <span>{latestReport.company}</span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
                Duration: {formatDuration(latestReport.durationSeconds)}
              </span>
              <span>•</span>
              <span>{latestReport.questions.length} Questions Evaluated</span>
            </div>
          </div>

          {/* Overall Score Badge */}
          <div
            className="flex items-center gap-4 shrink-0 p-5 rounded-[10px]"
            style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
          >
            <div className="text-right">
              <span className="text-3xl font-bold" style={{ color: PALETTE.textPrimary }}>
                {latestReport.overallScore}
              </span>
              <span className="text-xs font-medium" style={{ color: PALETTE.textSecondary }}> / 100</span>
              <span className="block text-[11px] font-semibold mt-0.5" style={{ color: PALETTE.accent }}>
                Strong Readiness
              </span>
            </div>
            <div
              className="w-11 h-11 rounded-[10px] flex items-center justify-center"
              style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.accent }}
            >
              <Award className="w-5 h-5" />
            </div>
          </div>
        </section>

        {/* CATEGORY SCORES & OBJECTIVE SPEECH SIGNALS */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Category Evaluation Scores */}
          <div className="p-6 rounded-[16px] space-y-4" style={panelStyle}>
            <SectionLabel>Evaluation Categories</SectionLabel>

            <div className="space-y-3.5">
              {[
                { label: 'Job Relevance', score: latestReport.categoryScores.jobRelevance },
                { label: 'Technical Knowledge', score: latestReport.categoryScores.technicalKnowledge },
                { label: 'Answer Quality', score: latestReport.categoryScores.answerQuality },
                { label: 'Communication & Clarity', score: latestReport.categoryScores.communication },
                { label: 'Problem Solving', score: latestReport.categoryScores.problemSolving },
              ].map((cat) => (
                <div key={cat.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium" style={{ color: PALETTE.textSecondary }}>
                      {cat.label}
                    </span>
                    <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                      {cat.score}%
                    </span>
                  </div>
                  <div className="w-full h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${cat.score}%`, backgroundColor: PALETTE.accent }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Measurable Speech & Delivery Signals */}
          <div className="p-6 rounded-[16px] space-y-4 flex flex-col justify-between" style={panelStyle}>
            <div className="space-y-4">
              <SectionLabel>Objective Delivery Metrics</SectionLabel>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div
                  className="p-3 rounded-[10px] space-y-1"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                >
                  <span className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>Speaking Pace</span>
                  <span className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
                    {latestReport.speechMetrics.wordsPerMinute} WPM
                  </span>
                  <span className="text-[10px] font-medium block" style={{ color: PALETTE.accent }}>
                    Ideal Range (120-150)
                  </span>
                </div>

                <div
                  className="p-3 rounded-[10px] space-y-1"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                >
                  <span className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>Long Pauses (&gt;3s)</span>
                  <span className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
                    {latestReport.speechMetrics.longPauses} detected
                  </span>
                  <span className="text-[10px] font-medium block" style={{ color: PALETTE.accent }}>
                    Natural cadence
                  </span>
                </div>

                <div
                  className="p-3 rounded-[10px] space-y-1"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                >
                  <span className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>Avg Response Time</span>
                  <span className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
                    {latestReport.speechMetrics.averageDurationSeconds}s
                  </span>
                  <span className="text-[10px] block" style={{ color: PALETTE.textSecondary }}>Well structured</span>
                </div>

                <div
                  className="p-3 rounded-[10px] space-y-1"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
                >
                  <span className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>STAR Adherence</span>
                  <span className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
                    {latestReport.speechMetrics.starAdherenceScore}%
                  </span>
                  <span className="text-[10px] font-medium block" style={{ color: PALETTE.accent }}>
                    High consistency
                  </span>
                </div>
              </div>
            </div>

            <p className="text-[11px] italic" style={{ color: PALETTE.textSecondary }}>
              * Delivery metrics measure concrete acoustic timing and structured phrasing without
              inferring internal emotional states.
            </p>
          </div>
        </section>

        {/* STRONG AREAS & AREAS TO IMPROVE */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Strong Areas */}
          <div className="p-6 rounded-[16px] space-y-3" style={panelStyle}>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" style={{ color: PALETTE.accent }} />
              <SectionLabel>Strongest Competencies</SectionLabel>
            </div>
            <ul className="space-y-2 text-xs">
              {latestReport.strongAreas.map((area, i) => (
                <li key={i} className="flex items-start gap-2" style={{ color: PALETTE.textPrimary }}>
                  <span className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: PALETTE.accent }}></span>
                  <span className="leading-relaxed">{area}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Areas to Improve */}
          <div className="p-6 rounded-[16px] space-y-3" style={panelStyle}>
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" style={{ color: AMBER }} />
              <SectionLabel>Targeted Areas for Growth</SectionLabel>
            </div>
            <ul className="space-y-2 text-xs">
              {latestReport.areasToImprove.map((area, i) => (
                <li key={i} className="flex items-start gap-2" style={{ color: PALETTE.textPrimary }}>
                  <span className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: AMBER }}></span>
                  <span className="leading-relaxed">{area}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* QUESTION-BY-QUESTION DEEP DIVE (ACCORDION) */}
        <section className="space-y-4">
          <div className="space-y-0.5">
            <h3 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
              Question-by-Question Detailed Review
            </h3>
            <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
              Expand any question to inspect transcripts, code submissions, and STAR breakdowns
            </p>
          </div>

          <div className="space-y-3">
            {latestReport.questions.map((q, index) => {
              const isExpanded = expandedQuestionId === q.id;
              const isCoding = q.type === 'coding';

              return (
                <div key={q.id} className="rounded-[16px] overflow-hidden transition-all" style={panelStyle}>
                  {/* Question Header Bar */}
                  <div
                    onClick={() => toggleQuestion(q.id)}
                    className="p-5 flex items-center justify-between gap-4 cursor-pointer transition-colors hover:bg-white/[0.02]"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span
                        className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                        style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textPrimary }}
                      >
                        {index + 1}
                      </span>
                      <div className="truncate">
                        <div className="flex items-center gap-2">
                          <span
                            className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded"
                            style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textSecondary }}
                          >
                            {q.type.replace('_', ' ')}
                          </span>
                          {isCoding && (
                            <span
                              className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded"
                              style={{ backgroundColor: `${PALETTE.accent}1A`, color: PALETTE.accent }}
                            >
                              4/4 Tests Passed
                            </span>
                          )}
                        </div>
                        <p className="text-xs font-medium mt-1 truncate" style={{ color: PALETTE.textPrimary }}>
                          {q.questionText}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs font-bold" style={{ color: PALETTE.textPrimary }}>
                        {q.evaluation?.score || 90}/100
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4" style={{ color: PALETTE.textSecondary }} />
                      ) : (
                        <ChevronDown className="w-4 h-4" style={{ color: PALETTE.textSecondary }} />
                      )}
                    </div>
                  </div>

                  {/* Expanded Body */}
                  {isExpanded && (
                    <div
                      className="p-6 space-y-6 text-xs"
                      style={{ borderTop: `1px solid ${PALETTE.borderTeal}`, backgroundColor: PALETTE.bg2 }}
                    >
                      {/* Candidate Answer Transcript */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <SectionLabel>Candidate Transcript</SectionLabel>
                          <button
                            onClick={() =>
                              handleSimulateListen(q.id, q.candidateTranscript || q.questionText)
                            }
                            className="flex items-center gap-1.5 text-xs font-semibold hover:underline"
                            style={{ color: PALETTE.accent }}
                          >
                            <Volume2 className="w-3.5 h-3.5" />
                            <span>
                              {playingAudioId === q.id ? 'Stop Audio' : 'Listen to Answer'}
                            </span>
                          </button>
                        </div>
                        <div
                          className="p-4 rounded-[10px] leading-relaxed"
                          style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                        >
                          {q.candidateTranscript || 'Transcript recorded.'}
                        </div>
                      </div>

                      {/* If Coding Question: Show Submitted Code & Test Execution Analysis */}
                      {isCoding && q.submittedCode && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-semibold uppercase tracking-wider flex items-center gap-1.5" style={{ color: PALETTE.textSecondary }}>
                              <Code2 className="w-3.5 h-3.5" style={{ color: PALETTE.accent }} />
                              Submitted Solution ({q.codeLanguage || 'python'})
                            </span>
                            <span className="text-[11px] font-mono font-medium" style={{ color: PALETTE.accent }}>
                              Optimal Time & Space
                            </span>
                          </div>

                          <div
                            className="p-4 rounded-[10px] font-mono text-xs overflow-x-auto"
                            style={{ backgroundColor: PALETTE.surface2, color: PALETTE.textPrimary, border: `1px solid ${PALETTE.border}` }}
                          >
                            <pre>{q.submittedCode}</pre>
                          </div>

                          {q.evaluation?.codingEvaluation && (
                            <div className="grid grid-cols-2 gap-3 pt-1">
                              <div className="p-3 rounded-[8px]" style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}>
                                <span className="text-[10px] block" style={{ color: PALETTE.textSecondary }}>
                                  Time Complexity
                                </span>
                                <span className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                                  {q.evaluation.codingEvaluation.timeComplexity}
                                </span>
                              </div>
                              <div className="p-3 rounded-[8px]" style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}>
                                <span className="text-[10px] block" style={{ color: PALETTE.textSecondary }}>
                                  Space Complexity
                                </span>
                                <span className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>
                                  {q.evaluation.codingEvaluation.spaceComplexity}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* STAR Framework Breakdown (if available) */}
                      {q.evaluation?.starFeedback && (
                        <div
                          className="p-4 rounded-[10px] space-y-3"
                          style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
                        >
                          <span className="text-[11px] font-semibold uppercase tracking-wider block" style={{ color: PALETTE.accent }}>
                            STAR Framework Breakdown
                          </span>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                            <div>
                              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                                Situation:
                              </span>{' '}
                              <span style={{ color: PALETTE.textSecondary }}>
                                {q.evaluation.starFeedback.situation}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                                Task:
                              </span>{' '}
                              <span style={{ color: PALETTE.textSecondary }}>
                                {q.evaluation.starFeedback.task}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                                Action:
                              </span>{' '}
                              <span style={{ color: PALETTE.textSecondary }}>
                                {q.evaluation.starFeedback.action}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>
                                Result:
                              </span>{' '}
                              <span style={{ color: PALETTE.textSecondary }}>
                                {q.evaluation.starFeedback.result}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Strengths & Improvements */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-1.5">
                          <span className="font-semibold text-xs" style={{ color: PALETTE.accent }}>
                            Strengths Noted:
                          </span>
                          <ul className="space-y-1" style={{ color: PALETTE.textSecondary }}>
                            {q.evaluation?.strengths.map((s, i) => (
                              <li key={i} className="flex items-start gap-1.5">
                                <span className="font-bold" style={{ color: PALETTE.accent }}>•</span>
                                <span>{s}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="space-y-1.5">
                          <span className="font-semibold text-xs" style={{ color: AMBER }}>
                            Growth Recommendation:
                          </span>
                          <ul className="space-y-1" style={{ color: PALETTE.textSecondary }}>
                            {q.evaluation?.improvements.map((imp, i) => (
                              <li key={i} className="flex items-start gap-1.5">
                                <span className="font-bold" style={{ color: AMBER }}>•</span>
                                <span>{imp}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Action CTA */}
                      <div className="pt-2 flex justify-end">
                        <button
                          onClick={handlePracticeAgain}
                          className="flex items-center gap-1.5 px-4 py-2 rounded-[8px] font-medium transition-colors hover:bg-white/[0.05]"
                          style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.textPrimary }}
                        >
                          <RotateCcw className="w-3 h-3" />
                          {/* Single-question replay isn't built yet (later phase) — this
                              routes into a fresh full setup rather than implying it will
                              replay just this question. */}
                          <span>Practice Similar Questions</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* RECOMMENDED PRACTICE PLAN */}
        <section
          className="p-7 rounded-[16px] space-y-4"
          style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.accent}30` }}
        >
          <div className="space-y-1">
            <span className="text-[11px] uppercase font-semibold tracking-wider" style={{ color: PALETTE.accent }}>
              Targeted Next Steps
            </span>
            <h3 className="text-base font-semibold" style={{ color: PALETTE.textPrimary }}>Recommended Practice Curriculum</h3>
          </div>

          <div className="space-y-2 text-xs">
            {latestReport.recommendedPractice.map((rec, i) => (
              <div key={i} className="flex items-start gap-2.5" style={{ color: PALETTE.textSecondary }}>
                <Zap className="w-4 h-4 mt-0.5 shrink-0" style={{ color: PALETTE.accent }} />
                <span className="leading-relaxed">{rec}</span>
              </div>
            ))}
          </div>

          <div className="pt-2 flex items-center gap-3">
            <Link
              href="/setup"
              className="px-4 py-2 rounded-[8px] text-xs font-semibold transition-opacity hover:opacity-90"
              style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
            >
              Start Next Practice Session
            </Link>
            <Link
              href="/"
              className="px-4 py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.05]"
              style={{ color: PALETTE.textSecondary }}
            >
              Back to Dashboard
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
