'use client';

import React from 'react';
import { X, Sparkles, User, Clock } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';

export const TranscriptDrawer: React.FC = () => {
  const { isTranscriptOpen, setIsTranscriptOpen, transcript, candidate } = useInterview();

  if (!isTranscriptOpen) return null;

  return (
    <div
      className="fixed inset-y-0 right-0 z-50 w-full max-w-md shadow-2xl flex flex-col"
      style={{ backgroundColor: PALETTE.surface, borderLeft: `1px solid ${PALETTE.border}` }}
    >
      {/* Header */}
      <div
        className="px-5 py-4 flex items-center justify-between"
        style={{ borderBottom: `1px solid ${PALETTE.border}` }}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: PALETTE.accentBright }} />
          <h3 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
            Interview Transcript
          </h3>
        </div>
        <button
          onClick={() => setIsTranscriptOpen(false)}
          className="p-1 rounded-lg transition-colors hover:bg-white/[0.06]"
          style={{ color: PALETTE.textSecondary }}
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {transcript.length === 0 ? (
          <div
            className="h-full flex flex-col items-center justify-center text-center p-6 text-xs"
            style={{ color: PALETTE.textSecondary }}
          >
            <Clock className="w-8 h-8 mb-2 opacity-50" />
            <p>Transcript will appear here as the conversation unfolds.</p>
          </div>
        ) : (
          transcript.map((msg) => {
            const isAi = msg.sender === 'ai';
            return (
              <div
                key={msg.id}
                className="flex flex-col gap-1.5 p-3.5 rounded-2xl text-xs leading-relaxed"
                style={
                  isAi
                    ? { backgroundColor: 'rgba(255,255,255,0.03)', border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }
                    : {
                        backgroundColor: 'rgba(85,229,207,0.06)',
                        border: `1px solid ${PALETTE.borderTeal}`,
                        color: PALETTE.textPrimary,
                        marginLeft: 16,
                      }
                }
              >
                <div className="flex items-center justify-between text-[11px] font-medium" style={{ color: PALETTE.textSecondary }}>
                  <div className="flex items-center gap-1.5">
                    {isAi ? (
                      <span className="flex items-center gap-1 font-semibold" style={{ color: PALETTE.accentBright }}>
                        <Sparkles className="w-3 h-3" />
                        AI Interviewer
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 font-semibold" style={{ color: PALETTE.textPrimary }}>
                        <User className="w-3 h-3" />
                        {candidate.name}
                      </span>
                    )}
                    {msg.isFollowUp && (
                      <span
                        className="px-1.5 py-0.2 rounded text-[10px]"
                        style={{ backgroundColor: 'rgba(245,181,68,0.1)', color: '#f5b544' }}
                      >
                        Follow-up
                      </span>
                    )}
                  </div>
                  <span>{msg.timestamp}</span>
                </div>
                <p className="whitespace-pre-wrap">{msg.text}</p>
              </div>
            );
          })
        )}
      </div>

      {/* Footer Info */}
      <div
        className="p-4 text-[11px] text-center"
        style={{ borderTop: `1px solid ${PALETTE.border}`, backgroundColor: 'rgba(255,255,255,0.02)', color: PALETTE.textSecondary }}
      >
        Real-time audio transcription · Private session
      </div>
    </div>
  );
};
