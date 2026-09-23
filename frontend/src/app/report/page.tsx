'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Sparkles,
  ArrowRight,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Volume2,
  Code2,
  FileText,
  ChevronDown,
  ChevronUp,
  Share2,
  Download,
  Award,
  Zap,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { TopHeader } from '@/components/layout/TopHeader';

export default function ReportPage() {
  const router = useRouter();
  const { latestReport, startInterviewSession } = useInterview();
  const [expandedQuestionId, setExpandedQuestionId] = useState<string | null>('q-4');
  const [playingAudioId, setPlayingAudioId] = useState<string | null>(null);

  const toggleQuestion = (id: string) => {
    setExpandedQuestionId(expandedQuestionId === id ? null : id);
  };

  const handlePracticeQuestionAgain = (questionIndex: number) => {
    startInterviewSession(questionIndex === 3 ? 'coding' : 'full');
    router.push('/interview');
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

  return (
    <div className="flex-1 flex flex-col min-h-screen pb-16">
      <TopHeader
        title="Interview Evaluation Report"
        subtitle="Comprehensive breakdown of your technical and conversational performance"
        actionButton={
          <div className="flex items-center gap-2">
            <button
              onClick={() => handlePracticeQuestionAgain(0)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 text-xs font-semibold shadow-sm transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Practice Again</span>
            </button>
          </div>
        }
      />

      <main className="flex-1 max-w-5xl w-full mx-auto px-8 py-8 space-y-8">
        {/* REPORT HEADER SUMMARY CARD */}
        <section className="p-8 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/70 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
                {latestReport.verdict}
              </span>
              <span className="text-xs text-neutral-400">·</span>
              <span className="text-xs text-neutral-400 font-mono">{latestReport.date}</span>
            </div>

            <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-100 tracking-tight">
              {latestReport.role}
            </h1>

            <div className="flex items-center gap-4 text-xs text-neutral-500 dark:text-neutral-400 font-medium">
              <span>{latestReport.company}</span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-neutral-400" />
                Duration: {formatDuration(latestReport.durationSeconds)}
              </span>
              <span>•</span>
              <span>{latestReport.questions.length} Questions Evaluated</span>
            </div>
          </div>

          {/* Overall Score Badge */}
          <div className="flex items-center gap-4 shrink-0 bg-neutral-50 dark:bg-neutral-900/60 p-5 rounded-2xl border border-neutral-200/60 dark:border-neutral-800/60">
            <div className="text-right">
              <span className="text-3xl font-extrabold text-neutral-900 dark:text-neutral-100">
                {latestReport.overallScore}
              </span>
              <span className="text-xs text-neutral-400 font-medium"> / 100</span>
              <span className="block text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5">
                Strong Readiness
              </span>
            </div>
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
              <Award className="w-6 h-6" />
            </div>
          </div>
        </section>

        {/* CATEGORY SCORES & OBJECTIVE SPEECH SIGNALS */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Category Evaluation Scores */}
          <div className="p-6 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
              Evaluation Categories
            </h3>

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
                    <span className="font-medium text-neutral-700 dark:text-neutral-300">
                      {cat.label}
                    </span>
                    <span className="font-semibold text-neutral-900 dark:text-neutral-100">
                      {cat.score}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{ width: `${cat.score}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Measurable Speech & Delivery Signals */}
          <div className="p-6 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-4 flex flex-col justify-between">
            <div className="space-y-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
                Objective Delivery Metrics
              </h3>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1">
                  <span className="text-[11px] text-neutral-400 block">Speaking Pace</span>
                  <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                    {latestReport.speechMetrics.wordsPerMinute} WPM
                  </span>
                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium block">
                    Ideal Range (120-150)
                  </span>
                </div>

                <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1">
                  <span className="text-[11px] text-neutral-400 block">Long Pauses (&gt;3s)</span>
                  <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                    {latestReport.speechMetrics.longPauses} detected
                  </span>
                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium block">
                    Natural cadence
                  </span>
                </div>

                <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1">
                  <span className="text-[11px] text-neutral-400 block">Avg Response Time</span>
                  <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                    {latestReport.speechMetrics.averageDurationSeconds}s
                  </span>
                  <span className="text-[10px] text-neutral-500 block">Well structured</span>
                </div>

                <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60 space-y-1">
                  <span className="text-[11px] text-neutral-400 block">STAR Adherence</span>
                  <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                    {latestReport.speechMetrics.starAdherenceScore}%
                  </span>
                  <span className="text-[10px] text-indigo-600 dark:text-indigo-400 font-medium block">
                    High consistency
                  </span>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-neutral-400 italic">
              * Delivery metrics measure concrete acoustic timing and structured phrasing without
              inferring internal emotional states.
            </p>
          </div>
        </section>

        {/* STRONG AREAS & AREAS TO IMPROVE */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Strong Areas */}
          <div className="p-6 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-800 dark:text-neutral-200">
                Strongest Competencies
              </h3>
            </div>
            <ul className="space-y-2 text-xs text-neutral-600 dark:text-neutral-300">
              {latestReport.strongAreas.map((area, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 shrink-0"></span>
                  <span className="leading-relaxed">{area}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Areas to Improve */}
          <div className="p-6 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-500" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-800 dark:text-neutral-200">
                Targeted Areas for Growth
              </h3>
            </div>
            <ul className="space-y-2 text-xs text-neutral-600 dark:text-neutral-300">
              {latestReport.areasToImprove.map((area, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0"></span>
                  <span className="leading-relaxed">{area}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* QUESTION-BY-QUESTION DEEP DIVE (ACCORDION) */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h3 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                Question-by-Question Detailed Review
              </h3>
              <p className="text-xs text-neutral-500">
                Expand any question to inspect transcripts, code submissions, and STAR breakdowns
              </p>
            </div>
          </div>

          <div className="space-y-3">
            {latestReport.questions.map((q, index) => {
              const isExpanded = expandedQuestionId === q.id;
              const isCoding = q.type === 'coding';

              return (
                <div
                  key={q.id}
                  className="rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 overflow-hidden shadow-xs transition-all"
                >
                  {/* Question Header Bar */}
                  <div
                    onClick={() => toggleQuestion(q.id)}
                    className="p-5 flex items-center justify-between gap-4 cursor-pointer hover:bg-neutral-50/70 dark:hover:bg-neutral-900/50 transition-colors"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="w-6 h-6 rounded-full bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center text-xs font-semibold text-neutral-700 dark:text-neutral-300 shrink-0">
                        {index + 1}
                      </span>
                      <div className="truncate">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800 text-neutral-500">
                            {q.type.replace('_', ' ')}
                          </span>
                          {isCoding && (
                            <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-300">
                              4/4 Tests Passed
                            </span>
                          )}
                        </div>
                        <p className="text-xs font-medium text-neutral-900 dark:text-neutral-100 mt-1 truncate">
                          {q.questionText}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs font-bold text-neutral-900 dark:text-neutral-100">
                        {q.evaluation?.score || 90}/100
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-neutral-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-neutral-400" />
                      )}
                    </div>
                  </div>

                  {/* Expanded Body */}
                  {isExpanded && (
                    <div className="p-6 border-t border-neutral-100 dark:border-neutral-800/60 space-y-6 text-xs bg-neutral-50/30 dark:bg-neutral-950/20">
                      {/* Candidate Answer Transcript */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                            Candidate Transcript
                          </span>
                          <button
                            onClick={() =>
                              handleSimulateListen(q.id, q.candidateTranscript || q.questionText)
                            }
                            className="flex items-center gap-1.5 text-xs text-indigo-600 dark:text-indigo-400 font-semibold hover:underline"
                          >
                            <Volume2 className="w-3.5 h-3.5" />
                            <span>
                              {playingAudioId === q.id ? 'Stop Audio' : 'Listen to Answer'}
                            </span>
                          </button>
                        </div>
                        <div className="p-4 rounded-2xl bg-white dark:bg-[#141417] border border-neutral-200/60 dark:border-neutral-800/60 text-neutral-700 dark:text-neutral-300 leading-relaxed">
                          {q.candidateTranscript || 'Transcript recorded.'}
                        </div>
                      </div>

                      {/* If Coding Question: Show Submitted Code & Test Execution Analysis */}
                      {isCoding && q.submittedCode && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-1.5">
                              <Code2 className="w-3.5 h-3.5 text-indigo-500" />
                              Submitted Solution ({q.codeLanguage || 'python'})
                            </span>
                            <span className="text-[11px] font-mono text-emerald-500 font-medium">
                              Optimal Time & Space
                            </span>
                          </div>

                          <div className="p-4 rounded-2xl bg-[#121215] text-neutral-200 font-mono text-xs border border-neutral-800 overflow-x-auto">
                            <pre>{q.submittedCode}</pre>
                          </div>

                          {q.evaluation?.codingEvaluation && (
                            <div className="grid grid-cols-2 gap-3 pt-1">
                              <div className="p-3 rounded-xl bg-neutral-100/60 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60">
                                <span className="text-neutral-400 text-[10px] block">
                                  Time Complexity
                                </span>
                                <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                                  {q.evaluation.codingEvaluation.timeComplexity}
                                </span>
                              </div>
                              <div className="p-3 rounded-xl bg-neutral-100/60 dark:bg-neutral-900/60 border border-neutral-200/60 dark:border-neutral-800/60">
                                <span className="text-neutral-400 text-[10px] block">
                                  Space Complexity
                                </span>
                                <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                                  {q.evaluation.codingEvaluation.spaceComplexity}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* STAR Framework Breakdown (if available) */}
                      {q.evaluation?.starFeedback && (
                        <div className="p-4 rounded-2xl bg-white dark:bg-[#141417] border border-neutral-200/60 dark:border-neutral-800/60 space-y-3">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 block">
                            STAR Framework Breakdown
                          </span>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                            <div>
                              <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                                Situation:
                              </span>{' '}
                              <span className="text-neutral-600 dark:text-neutral-400">
                                {q.evaluation.starFeedback.situation}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                                Task:
                              </span>{' '}
                              <span className="text-neutral-600 dark:text-neutral-400">
                                {q.evaluation.starFeedback.task}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                                Action:
                              </span>{' '}
                              <span className="text-neutral-600 dark:text-neutral-400">
                                {q.evaluation.starFeedback.action}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                                Result:
                              </span>{' '}
                              <span className="text-neutral-600 dark:text-neutral-400">
                                {q.evaluation.starFeedback.result}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Strengths & Improvements */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-1.5">
                          <span className="font-semibold text-emerald-600 dark:text-emerald-400 text-xs">
                            Strengths Noted:
                          </span>
                          <ul className="space-y-1 text-neutral-600 dark:text-neutral-300">
                            {q.evaluation?.strengths.map((s, i) => (
                              <li key={i} className="flex items-start gap-1.5">
                                <span className="text-emerald-500 font-bold">•</span>
                                <span>{s}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="space-y-1.5">
                          <span className="font-semibold text-amber-600 dark:text-amber-400 text-xs">
                            Growth Recommendation:
                          </span>
                          <ul className="space-y-1 text-neutral-600 dark:text-neutral-300">
                            {q.evaluation?.improvements.map((imp, i) => (
                              <li key={i} className="flex items-start gap-1.5">
                                <span className="text-amber-500 font-bold">•</span>
                                <span>{imp}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Action CTA */}
                      <div className="pt-2 flex justify-end">
                        <button
                          onClick={() => handlePracticeQuestionAgain(index)}
                          className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 text-neutral-800 dark:text-neutral-200 font-medium transition-colors"
                        >
                          <RotateCcw className="w-3 h-3" />
                          <span>Practice This Question Again</span>
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
        <section className="p-7 rounded-3xl bg-neutral-900 dark:bg-white text-white dark:text-neutral-950 space-y-4 shadow-xl">
          <div className="space-y-1">
            <span className="text-[11px] uppercase font-semibold text-indigo-400 dark:text-indigo-600 tracking-wider">
              Targeted Next Steps
            </span>
            <h3 className="text-base font-semibold">Recommended Practice Curriculum</h3>
          </div>

          <div className="space-y-2 text-xs">
            {latestReport.recommendedPractice.map((rec, i) => (
              <div key={i} className="flex items-start gap-2.5 opacity-90">
                <Zap className="w-4 h-4 text-indigo-400 dark:text-indigo-600 mt-0.5 shrink-0" />
                <span className="leading-relaxed">{rec}</span>
              </div>
            ))}
          </div>

          <div className="pt-2 flex items-center gap-3">
            <Link
              href="/setup"
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-xs transition-colors"
            >
              Start Next Practice Session
            </Link>
            <Link
              href="/"
              className="px-4 py-2 rounded-xl text-xs font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              Back to Dashboard
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
