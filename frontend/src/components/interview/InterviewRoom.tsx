'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  MessageSquare,
  PhoneOff,
  ArrowRight,
  CheckCircle,
  Clock,
  AlertTriangle,
  Code2,
} from 'lucide-react';

import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';
import { SiriOrb } from './SiriOrb';
import { useSimulatedAmplitude, type AIState } from './ai-core';
import { VoiceControls } from './VoiceControls';
import { TranscriptDrawer } from './TranscriptDrawer';
import { SelfViewCamera } from './SelfViewCamera';
import { CodingWorkspace } from './coding/CodingWorkspace';
import { InterviewReadinessScreen } from './InterviewReadinessScreen';

// The orb visualizes the AI's own voice ('streaming' = avatarState
// 'speaking'), not the candidate's mic input — 'listening' here means the
// AI is waiting/attentive while the candidate answers, not that the orb is
// reacting to the candidate's voice.
function toOrbAIState(state: string): AIState {
  switch (state) {
    case 'speaking':
      return 'streaming';
    case 'listening':
      return 'listening';
    case 'thinking':
      return 'thinking';
    default:
      return 'idle';
  }
}

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
    hasActiveSession,
    hasCompletedSession,
  } = useInterview();

  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const orbState = toOrbAIState(avatarState);
  const orbAmplitude = useSimulatedAmplitude(orbState);

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

  // Whether THIS question has a code editor to offer at all — independent
  // of whether it's currently showing. Drives the right-edge tab's
  // visibility; `showCodeEditor` is the user's own open/closed choice.
  const hasCodingTask = Boolean(currentQuestion?.codingTask);
  const [showCodeEditor, setShowCodeEditor] = useState(false);

  // Keeps the previous behavior (coding questions open the editor
  // automatically) as the default, while still letting the candidate close
  // it and reopen via the tab — the click-to-open bar is additive, not a
  // replacement for the existing auto-open.
  useEffect(() => {
    if (isCodingMode) setShowCodeEditor(true);
  }, [isCodingMode, currentQuestionIndex]);

  const handleEndInterviewClick = () => {
    if (interviewState === 'completed' || interviewState === 'ended_early') return;
    setShowEndConfirm(true);
  };

  const handleConfirmEndEarly = () => {
    setShowEndConfirm(false);
    endInterviewSession('ended_early');
  };

  // This room only ever consumes an already-prepared session — it does not
  // create one itself. A direct/stale visit (no session started from Setup,
  // or one from a previous browser session left idle) shows the readiness
  // pre-flight screen instead of silently auto-starting a brand-new interview.
  if (!hasActiveSession && !hasCompletedSession) {
    return <InterviewReadinessScreen />;
  }

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
            onClick={handleEndInterviewClick}
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
      {hasCodingTask && showCodeEditor ? (
        <CodingWorkspace onClose={() => setShowCodeEditor(false)} />
      ) : (
        <main className="flex-1 flex flex-col justify-between items-center p-8 max-w-5xl mx-auto w-full relative">
          {/* Right-edge tab — always visible, not gated on question data.
              Opens the editor if this question actually has one
              (`hasCodingTask`); otherwise the click is a no-op, so there's
              nothing to keep in sync as questions change. */}
          {!showCodeEditor && (
            <button
              onClick={() => hasCodingTask && setShowCodeEditor(true)}
              className="fixed right-0 top-1/2 -translate-y-1/2 z-30 flex flex-col items-center gap-1.5 pl-2.5 pr-2 py-3 rounded-l-xl text-[11px] font-medium transition-colors hover:bg-white/[0.06] disabled:opacity-40 disabled:pointer-events-none"
              disabled={!hasCodingTask}
              style={{
                backgroundColor: 'rgba(10,17,16,0.9)',
                border: `1px solid ${PALETTE.border}`,
                borderRight: 'none',
                color: PALETTE.textSecondary,
              }}
              title={hasCodingTask ? 'Open code editor' : 'This question has no code editor'}
            >
              <Code2 className="w-4 h-4" style={{ color: hasCodingTask ? PALETTE.accent : PALETTE.textSecondary }} />
              <span style={{ writingMode: 'vertical-rl' }}>Code</span>
            </button>
          )}

          {/* Follow-up Question Indicator Banner */}
          {isFollowUpActive && (
            <div
              className="px-4 py-1.5 rounded-full text-xs font-medium flex items-center gap-2"
              style={{ backgroundColor: 'rgba(255, 255, 255,0.08)', border: `1px solid ${PALETTE.borderTeal}`, color: PALETTE.accentBright }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PALETTE.accentBright }} />
              <span>Conversational follow-up based on your response</span>
            </div>
          )}

          {/* 3D AI Interviewer Stage */}
          <div className="w-full max-w-md h-72 flex items-center justify-center my-auto relative">
            <SiriOrb state={orbState} amplitude={orbAmplitude} size="240px" />

            {/* Subtle state pill below the orb — centered under it regardless
                of its own text width (absolute positioning drops it out of
                the flex layout that used to center it, so it needs its own
                left-1/2 + -translate-x-1/2 rather than relying on that). */}
            <div
              className="absolute -bottom-3 left-1/2 -translate-x-1/2 px-3.5 py-1.5 rounded-full backdrop-blur-md text-[11px] font-medium flex items-center gap-2 shadow-[0_8px_24px_-8px_rgba(0,0,0,0.6)] whitespace-nowrap"
              style={{ backgroundColor: 'rgba(10,17,16,0.85)', border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }}
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
            <VoiceControls onEndInterviewClick={handleEndInterviewClick} />
          </div>
        </main>
      )}

      {/* END-INTERVIEW CONFIRMATION — only shown for a manual early end, not
          the natural full-completion path, which never touches this modal. */}
      {showEndConfirm && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4">
          <div
            className="w-full max-w-sm rounded-3xl p-7 shadow-2xl text-center space-y-5"
            style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
          >
            <div
              className="w-12 h-12 rounded-2xl mx-auto flex items-center justify-center"
              style={{ backgroundColor: 'rgba(245,181,68,0.12)', color: '#f5b544' }}
            >
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="space-y-2">
              <h3 className="text-base font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
                End interview?
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                You have completed {currentQuestionIndex} of {totalQuestions} questions. You can
                continue practicing, or end now and review a partial session.
              </p>
            </div>
            <div className="space-y-2 pt-1">
              <button
                onClick={() => setShowEndConfirm(false)}
                className="w-full py-2.5 px-4 rounded-xl text-xs font-semibold transition-opacity hover:opacity-90"
                style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
              >
                Continue Interview
              </button>
              <button
                onClick={handleConfirmEndEarly}
                className="w-full py-2.5 px-4 rounded-xl text-xs font-medium transition-colors hover:bg-red-500/10"
                style={{ color: '#f87171' }}
              >
                End Interview
              </button>
            </div>
          </div>
        </div>
      )}

      {/* INTERVIEW COMPLETION MODAL — copy is accurate to how the session
          actually ended: full completion vs. a manual early end. */}
      {(interviewState === 'completed' || interviewState === 'ended_early') && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4">
          <div
            className="w-full max-w-md rounded-3xl p-8 shadow-2xl text-center space-y-6"
            style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
          >
            <div
              className="w-14 h-14 rounded-2xl mx-auto flex items-center justify-center"
              style={
                interviewState === 'completed'
                  ? { backgroundColor: 'rgba(231, 232, 233,0.12)', color: PALETTE.accentBright }
                  : { backgroundColor: 'rgba(245,181,68,0.12)', color: '#f5b544' }
              }
            >
              {interviewState === 'completed' ? (
                <CheckCircle className="w-7 h-7" />
              ) : (
                <AlertTriangle className="w-7 h-7" />
              )}
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
                {interviewState === 'completed' ? 'Interview Complete' : 'Interview Ended'}
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                {interviewState === 'completed' ? (
                  <>
                    Great job! You completed all {totalQuestions} questions including behavioral,
                    system design, and algorithmic coding challenges.
                  </>
                ) : (
                  <>
                    You completed {currentQuestionIndex} of {totalQuestions} questions. A partial
                    report can be reviewed.
                  </>
                )}
              </p>
            </div>

            <div
              className="p-4 rounded-2xl flex items-center justify-around text-xs"
              style={{ backgroundColor: 'rgba(255,255,255,0.03)', border: `1px solid ${PALETTE.border}` }}
            >
              <div>
                <span className="block text-[11px]" style={{ color: PALETTE.textSecondary }}>Questions</span>
                <span className="font-semibold text-sm" style={{ color: PALETTE.textPrimary }}>
                  {interviewState === 'completed' ? totalQuestions : currentQuestionIndex} / {totalQuestions}
                </span>
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
                <span
                  className="font-semibold text-sm"
                  style={{ color: interviewState === 'completed' ? PALETTE.accentBright : '#f5b544' }}
                >
                  {interviewState === 'completed' ? 'Ready' : 'Partial'}
                </span>
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
