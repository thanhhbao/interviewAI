'use client';

import React, { useEffect, useState } from 'react';
import {
  Mic,
  RotateCcw,
  Volume2,
  FileText,
  Square,
  Sparkles,
  PhoneOff,
  Video,
  VideoOff,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';

export const VoiceControls: React.FC = () => {
  const {
    interviewState,
    startAnswer,
    finishAnswer,
    restartAnswer,
    repeatCurrentQuestion,
    endInterviewSession,
    answerTimerSeconds,
    isTranscriptOpen,
    setIsTranscriptOpen,
    isCameraActive,
    setIsCameraActive,
  } = useInterview();

  // Waveform heights simulation when recording
  const [waveBars, setWaveBars] = useState<number[]>([12, 18, 28, 16, 22, 34, 18, 14]);

  useEffect(() => {
    let animInterval: NodeJS.Timeout;
    if (interviewState === 'recording') {
      animInterval = setInterval(() => {
        setWaveBars([
          Math.floor(Math.random() * 28) + 8,
          Math.floor(Math.random() * 38) + 12,
          Math.floor(Math.random() * 45) + 14,
          Math.floor(Math.random() * 32) + 10,
          Math.floor(Math.random() * 48) + 16,
          Math.floor(Math.random() * 36) + 12,
          Math.floor(Math.random() * 26) + 8,
          Math.floor(Math.random() * 18) + 6,
        ]);
      }, 100);
    } else {
      setWaveBars([10, 10, 10, 10, 10, 10, 10, 10]);
    }
    return () => clearInterval(animInterval);
  }, [interviewState]);

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col items-center gap-4">
      {/* Waveform & Recording Timer Indicator */}
      {interviewState === 'recording' && (
        <div
          className="flex items-center gap-3 px-4 py-2 rounded-full text-xs font-medium"
          style={{ backgroundColor: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', color: '#f87171' }}
        >
          <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: '#f87171' }} />
          <span>Recording: {formatTimer(answerTimerSeconds)}</span>

          {/* Dynamic Audio Waveform */}
          <div className="flex items-center gap-1 h-6 px-1">
            {waveBars.map((height, i) => (
              <span
                key={i}
                className="w-1 rounded-full transition-all duration-100"
                style={{ height: `${height}px`, backgroundColor: '#f87171' }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Main Control Strip */}
      <div
        className="flex items-center gap-3 px-5 py-3 rounded-2xl shadow-lg backdrop-blur-md"
        style={{ backgroundColor: 'rgba(10,17,16,0.9)', border: `1px solid ${PALETTE.border}` }}
      >
        {/* Toggle Transcript Drawer */}
        <button
          onClick={() => setIsTranscriptOpen(!isTranscriptOpen)}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-colors hover:bg-white/[0.05]"
          style={
            isTranscriptOpen
              ? { backgroundColor: PALETTE.accent, color: '#07080a' }
              : { color: PALETTE.textSecondary }
          }
          title="Toggle transcript"
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Transcript</span>
        </button>

        {/* Repeat Question Button */}
        <button
          onClick={repeatCurrentQuestion}
          disabled={interviewState === 'ai_speaking' || interviewState === 'processing'}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-colors hover:bg-white/[0.05] disabled:opacity-40 disabled:pointer-events-none"
          style={{ color: PALETTE.textSecondary }}
          title="Repeat Question"
        >
          <Volume2 className="w-3.5 h-3.5" />
          <span>Repeat Question</span>
        </button>

        <div className="w-px h-6" style={{ backgroundColor: PALETTE.border }} />

        {/* PRIMARY PUSH-TO-TALK BUTTON */}
        {interviewState === 'recording' ? (
          <div className="flex items-center gap-2">
            <button
              onClick={() => finishAnswer()}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-white font-medium text-xs shadow-md transition-all duration-150 active:scale-95"
              style={{ backgroundColor: '#dc2626', boxShadow: '0 4px 14px rgba(220,38,38,0.25)' }}
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Finish Answer</span>
            </button>
            <button
              onClick={restartAnswer}
              className="p-2 rounded-xl transition-colors hover:bg-white/[0.05]"
              style={{ color: PALETTE.textSecondary }}
              title="Restart Answer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : interviewState === 'waiting_for_answer' ? (
          <button
            onClick={startAnswer}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-xs shadow-md transition-all duration-150 active:scale-95 hover:opacity-90"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a', boxShadow: '0 4px 14px rgba(85,229,207,0.2)' }}
          >
            <Mic className="w-3.5 h-3.5" />
            <span>Start Answer</span>
          </button>
        ) : interviewState === 'ai_speaking' ? (
          <div
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium"
            style={{ backgroundColor: 'rgba(255,255,255,0.04)', color: PALETTE.textSecondary }}
          >
            <span className="w-2 h-2 rounded-full animate-ping" style={{ backgroundColor: PALETTE.accentBright }} />
            <span>AI is speaking...</span>
          </div>
        ) : interviewState === 'processing' ? (
          <div
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium"
            style={{ backgroundColor: 'rgba(255,255,255,0.04)', color: PALETTE.textSecondary }}
          >
            <Sparkles className="w-3.5 h-3.5 animate-spin" style={{ color: PALETTE.accent }} />
            <span>Understanding your answer...</span>
          </div>
        ) : null}

        <div className="w-px h-6" style={{ backgroundColor: PALETTE.border }} />

        {/* Optional Self-View Video Preview Toggle */}
        <button
          onClick={() => setIsCameraActive(!isCameraActive)}
          className="p-2 rounded-xl text-xs transition-colors hover:bg-white/[0.05]"
          style={
            isCameraActive
              ? { backgroundColor: 'rgba(85,229,207,0.1)', color: PALETTE.accentBright }
              : { color: PALETTE.textSecondary }
          }
          title={isCameraActive ? 'Hide self-view camera' : 'Show self-view camera'}
        >
          {isCameraActive ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4" />}
        </button>

        {/* End Interview */}
        <button
          onClick={endInterviewSession}
          className="p-2 rounded-xl transition-colors hover:bg-red-500/10 hover:text-red-400"
          style={{ color: PALETTE.textSecondary }}
          title="End Interview"
        >
          <PhoneOff className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
