'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  MessageSquare,
  PhoneOff,
  ArrowRight,
  CheckCircle,
  Clock,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';
import { InterviewAvatar3D } from './InterviewAvatar3D';
import { VoiceControls } from './VoiceControls';
import { TranscriptDrawer } from './TranscriptDrawer';
import { SelfViewCamera } from './SelfViewCamera';
import { CodingWorkspace } from './coding/CodingWorkspace';

export const InterviewRoom: React.FC = () => {
  const router = useRouter();
  const {
    targetJob,
    currentQuestionIndex,
    questions,
    currentQuestion,
    interviewState,
    avatarState,
    activeSubtitle,
    elapsedSeconds,
    isFollowUpActive,
    endInterviewSession,
    startInterviewSession,
  } = useInterview();

  // Auto-start interview if idle upon loading room
  useEffect(() => {
    if (interviewState === 'idle') {
      startInterviewSession();
    }
  }, [interviewState, startInterviewSession]);

  const totalQuestions = questions.length;
  const progressPercent = Math.round(((currentQuestionIndex + 1) / totalQuestions) * 100);

  const formatElapsed = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  const isCodingMode =
    currentQuestion?.type === 'coding' &&
    (interviewState === 'coding' ||
      interviewState === 'running_code' ||
      interviewState === 'reviewing_code');

  return (
    <div
      className="relative w-screen h-screen flex flex-col overflow-hidden select-none"
      style={{ backgroundColor: PALETTE.bg, color: PALETTE.textPrimary }}
    >
      {/* TOP STATUS BAR */}
      <header
        className="h-16 px-6 flex items-center justify-between z-20 shrink-0 backdrop-blur-md"
        style={{ borderBottom: `1px solid ${PALETTE.border}`, backgroundColor: 'rgba(7,8,10,0.75)' }}
      >
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2">
            <div
              className="w-7 h-7 rounded-md flex items-center justify-center"
              style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
            >
              <MessageSquare className="w-3.5 h-3.5" />
            </div>
            <span className="font-semibold text-xs tracking-tight" style={{ color: PALETTE.textPrimary }}>
              InterviewPal
            </span>
          </Link>

          <div className="w-px h-4" style={{ backgroundColor: PALETTE.border }} />

          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium" style={{ color: PALETTE.textPrimary }}>
              {targetJob.title}
            </span>
            <span style={{ color: PALETTE.textSecondary }}>·</span>
            <span style={{ color: PALETTE.textSecondary }}>{targetJob.company}</span>
          </div>
        </div>

        {/* Center Progress & Timer */}
        <div className="flex items-center gap-5 text-xs">
          <div className="flex items-center gap-2">
            <span style={{ color: PALETTE.textSecondary }}>
              Question {currentQuestionIndex + 1} of {totalQuestions}
            </span>
            <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
              <div
                className="h-full transition-all duration-300 rounded-full"
                style={{ width: `${progressPercent}%`, backgroundColor: PALETTE.accent }}
              />
            </div>
          </div>

          <div
            className="flex items-center gap-1.5 font-mono text-[11px] px-2.5 py-1 rounded-full"
            style={{ color: PALETTE.textSecondary, backgroundColor: 'rgba(255,255,255,0.05)' }}
          >
            <Clock className="w-3 h-3" style={{ color: PALETTE.textSecondary }} />
            <span>{formatElapsed(elapsedSeconds)}</span>
          </div>

          <div className="flex items-center gap-1.5 text-[11px]" style={{ color: PALETTE.accentBright }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: PALETTE.accentBright }} />
            <span>Live Session</span>
          </div>
        </div>

        {/* Right End Call */}
        <div className="flex items-center gap-3">
          <button
            onClick={endInterviewSession}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors hover:bg-white/[0.04]"
            style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }}
          >
            <PhoneOff className="w-3.5 h-3.5 text-red-400" />
            <span>End Interview</span>
          </button>
        </div>
      </header>

      {/* Optional Self-View video preview */}
      <SelfViewCamera />

      {/* Synchronized Transcript Drawer */}
      <TranscriptDrawer />

      {/* MAIN INTERVIEW VIEWPORT */}
      {isCodingMode ? (
        <CodingWorkspace />
      ) : (
        <main className="flex-1 flex flex-col justify-between items-center p-8 max-w-5xl mx-auto w-full relative">
          {/* Follow-up Question Indicator Banner */}
          {isFollowUpActive && (
            <div
              className="px-4 py-1.5 rounded-full text-xs font-medium flex items-center gap-2"
              style={{ backgroundColor: 'rgba(160,220,210,0.08)', border: `1px solid ${PALETTE.borderTeal}`, color: PALETTE.accentBright }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accentBright }} />
              <span>Conversational follow-up based on your response</span>
            </div>
          )}

          {/* 3D AI Interviewer Stage */}
          <div className="w-full max-w-md h-72 flex items-center justify-center my-auto relative">
            <InterviewAvatar3D state={avatarState} isCompact={false} />

            {/* Subtle state pill below 3D character */}
            <div
              className="absolute -bottom-2 px-3 py-1 rounded-full backdrop-blur-md text-[11px] font-medium flex items-center gap-1.5"
              style={{ backgroundColor: 'rgba(10,17,16,0.8)', border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor:
                    avatarState === 'speaking'
                      ? PALETTE.accentBright
                      : avatarState === 'listening'
                      ? '#f87171'
                      : avatarState === 'thinking'
                      ? '#f5b544'
                      : PALETTE.accent,
                  animation: avatarState === 'speaking' || avatarState === 'listening' ? 'pulse 2s infinite' : undefined,
                }}
              />
              <span className="capitalize">
                {interviewState === 'ai_speaking'
                  ? 'AI is speaking'
                  : interviewState === 'recording'
                  ? 'Listening to answer...'
                  : interviewState === 'processing'
                  ? 'Processing response...'
                  : 'Your turn to speak'}
              </span>
            </div>
          </div>

          {/* Conversational Question Subtitle */}
          <div className="w-full max-w-2xl text-center space-y-2 mb-6">
            <div
              className="p-5 rounded-2xl backdrop-blur-md"
              style={{ backgroundColor: PALETTE.surfaceFaint, border: `1px solid ${PALETTE.border}` }}
            >
              <p className="text-sm md:text-base font-normal leading-relaxed" style={{ color: PALETTE.textPrimary }}>
                &ldquo;{activeSubtitle || currentQuestion.questionText}&rdquo;
              </p>
            </div>
          </div>

          {/* Controlled Push-to-Talk Recording Controls */}
          <div className="w-full pb-4">
            <VoiceControls />
          </div>
        </main>
      )}

      {/* INTERVIEW COMPLETION MODAL */}
      {interviewState === 'completed' && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4">
          <div
            className="w-full max-w-md rounded-3xl p-8 shadow-2xl text-center space-y-6"
            style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
          >
            <div
              className="w-14 h-14 rounded-2xl mx-auto flex items-center justify-center"
              style={{ backgroundColor: 'rgba(85,229,207,0.12)', color: PALETTE.accentBright }}
            >
              <CheckCircle className="w-7 h-7" />
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
                Interview Complete
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                Great job! You answered all {totalQuestions} questions including behavioral,
                system design, and algorithmic coding challenges.
              </p>
            </div>

            <div
              className="p-4 rounded-2xl flex items-center justify-around text-xs"
              style={{ backgroundColor: 'rgba(255,255,255,0.03)', border: `1px solid ${PALETTE.border}` }}
            >
              <div>
                <span className="block text-[11px]" style={{ color: PALETTE.textSecondary }}>Questions</span>
                <span className="font-semibold text-sm" style={{ color: PALETTE.textPrimary }}>{totalQuestions}</span>
              </div>
              <div className="w-px h-6" style={{ backgroundColor: PALETTE.border }} />
              <div>
                <span className="block text-[11px]" style={{ color: PALETTE.textSecondary }}>Duration</span>
                <span className="font-semibold text-sm" style={{ color: PALETTE.textPrimary }}>
                  {formatElapsed(elapsedSeconds)}
                </span>
              </div>
              <div className="w-px h-6" style={{ backgroundColor: PALETTE.border }} />
              <div>
                <span className="block text-[11px]" style={{ color: PALETTE.textSecondary }}>Status</span>
                <span className="font-semibold text-sm" style={{ color: PALETTE.accentBright }}>Ready</span>
              </div>
            </div>

            <div className="space-y-2.5 pt-2">
              <button
                onClick={() => router.push('/report')}
                className="w-full py-3 px-4 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-opacity hover:opacity-90"
                style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
              >
                <span>View Interview Report</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={() => router.push('/')}
                className="w-full py-2.5 px-4 rounded-xl text-xs font-medium transition-colors hover:text-white"
                style={{ color: PALETTE.textSecondary }}
              >
                Return Home
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
