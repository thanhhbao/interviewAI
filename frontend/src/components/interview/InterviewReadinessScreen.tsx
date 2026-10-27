'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowRight, CheckCircle2, XCircle, Circle, Mic, Camera, FileText } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';
import { AppBackground } from '@/components/app/AppBackground';

// ── Readiness terminal — right-panel, static output + cursor ────────────────

function ReadinessTerminal({ resumeReady, roleDetected }: { resumeReady: boolean; roleDetected: boolean }) {
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    const id = setInterval(() => setShowCursor((v) => !v), 530);
    return () => clearInterval(id);
  }, []);

  const statusLine = (label: string, value: string, ok: boolean | null) => (
    <div className="flex items-baseline gap-0" key={label}>
      <span style={{ color: PALETTE.textSecondary, minWidth: '13ch', display: 'inline-block' }}>{label}</span>
      <span style={{ color: PALETTE.textSecondary }}>{' .... '}</span>
      <span style={{ color: ok === true ? '#7ecba1' : ok === false ? '#f87171' : PALETTE.textSecondary }}>
        {value}
      </span>
    </div>
  );

  return (
    <div
      className="rounded-[10px] overflow-hidden h-full"
      style={{
        backgroundColor: '#050908',
        border: `1px solid ${PALETTE.border}`,
        fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, monospace',
      }}
    >
      {/* Terminal chrome bar */}
      <div
        className="flex items-center gap-2 px-4 py-3 shrink-0"
        style={{ borderBottom: `1px solid ${PALETTE.border}` }}
      >
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />
        <span className="ml-3 text-[11px] tracking-[0.08em]" style={{ color: PALETTE.textSecondary }}>
          interview-session
        </span>
      </div>

      {/* Terminal content */}
      <div className="p-5 space-y-1.5 text-[12px] leading-[1.8]">
        <div style={{ color: PALETTE.accent }}>{'> PREPARING INTERVIEW'}</div>
        <div className="pt-1 pb-0.5" />

        {statusLine('resume', resumeReady ? 'ready' : 'missing', resumeReady ? true : false)}
        {statusLine('target role', roleDetected ? 'detected' : 'not set', roleDetected ? true : null)}
        {statusLine('microphone', 'ready', true)}
        {statusLine('camera', 'optional', null)}
        {statusLine('questions', resumeReady ? 'ready' : 'waiting', resumeReady ? true : null)}

        <div className="pt-3" />
        <div style={{ color: PALETTE.textSecondary }}>
          {'> '}
          <span style={{ color: resumeReady ? PALETTE.accent : '#f87171' }}>
            {resumeReady ? 'session ready' : 'add resume to continue'}
          </span>
          {' '}
          <span style={{ opacity: showCursor ? 1 : 0, color: PALETTE.accent }}>_</span>
        </div>

        {/* Subtle scanline overlay */}
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            backgroundImage:
              'repeating-linear-gradient(to bottom, transparent 0px, transparent 3px, rgba(0,0,0,0.05) 3px, rgba(0,0,0,0.05) 4px)',
            pointerEvents: 'none',
            borderRadius: 'inherit',
          }}
        />
      </div>
    </div>
  );
}

// ── Status badge used in checklist rows ──────────────────────────────────────

function StatusDot({ state }: { state: 'ok' | 'missing' | 'optional' }) {
  if (state === 'ok')
    return <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: '#7ecba1' }} />;
  if (state === 'missing')
    return <XCircle className="w-3.5 h-3.5 shrink-0" style={{ color: '#f87171' }} />;
  return <Circle className="w-3.5 h-3.5 shrink-0" style={{ color: PALETTE.textSecondary, opacity: 0.5 }} />;
}

// ── Label pill for "Required" / "Optional" ───────────────────────────────────

function Pill({ text, variant }: { text: string; variant: 'required' | 'optional' }) {
  return (
    <span
      className="text-[10px] font-mono tracking-[0.1em] px-2 py-0.5 rounded-[5px]"
      style={{
        color: variant === 'required' ? '#f5b544' : PALETTE.textSecondary,
        backgroundColor: variant === 'required' ? 'rgba(245,181,68,0.08)' : 'rgba(255,255,255,0.04)',
        border: `1px solid ${variant === 'required' ? 'rgba(245,181,68,0.2)' : PALETTE.border}`,
      }}
    >
      {text}
    </span>
  );
}

// ── Readiness checklist (extracted to avoid satisfies-in-JSX parse issues) ──

type ChecklistState = 'ok' | 'missing' | 'optional';
interface ChecklistRow { icon: React.ElementType; label: string; state: ChecklistState; note: string }

function ChecklistPanel({ resumeReady }: { resumeReady: boolean }) {
  const rows: ChecklistRow[] = [
    { icon: FileText, label: 'Resume',     state: resumeReady ? 'ok' : 'missing', note: resumeReady ? 'Loaded'   : 'Missing'  },
    { icon: Mic,      label: 'Microphone', state: 'ok',                            note: 'Ready'                              },
    { icon: Camera,   label: 'Camera',     state: 'optional',                      note: 'Optional'                           },
  ];

  return (
    <div className="rounded-[10px] overflow-hidden" style={{ border: `1px solid ${PALETTE.border}` }}>
      {rows.map(({ icon: Icon, label, state, note }, i) => (
        <div
          key={label}
          className="flex items-center gap-4 px-4 py-3"
          style={{ borderTop: i > 0 ? `1px solid ${PALETTE.border}` : undefined }}
        >
          <Icon className="w-4 h-4 shrink-0" style={{ color: PALETTE.textSecondary }} />
          <span className="flex-1 text-[13px]" style={{ color: PALETTE.textPrimary }}>{label}</span>
          <StatusDot state={state} />
          <span className="text-[12px] font-mono w-14 text-right" style={{ color: PALETTE.textSecondary }}>
            {note}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Root component ───────────────────────────────────────────────────────────

export const InterviewReadinessScreen: React.FC = () => {
  const router = useRouter();
  const {
    hasResumeContext,
    hasTargetJobContext,
    hasInterviewContext,
    targetJob,
    interviewType,
    difficulty,
    voice,
    jobStatus,
    matchStatus,
  } = useInterview();

  const resumeReady = hasResumeContext;
  const roleDetected = hasTargetJobContext;
  const isAnalyzing =
    hasResumeContext &&
    (jobStatus === 'analyzing' || matchStatus === 'analyzing');

  const sessionLabel: Record<string, string> = {
    full: 'Full interview',
    behavioral: 'Behavioral only',
    technical: 'Technical only',
    coding: 'Coding round',
  };

  const durationMap: Record<string, string> = {
    full: '~25 min',
    behavioral: '~15 min',
    technical: '~20 min',
    coding: '~20 min',
  };

  return (
    <div
      className="relative flex-1 flex flex-col"
      style={{ color: PALETTE.textPrimary }}
    >
      {/* Atmospheric dot-grid background — fixed position, behind everything. */}
      <AppBackground />

      {/* BODY — global CosmicNavbar above handles navigation */}
      <main className="flex-1 overflow-y-auto">
        <div className="min-h-full flex items-center justify-center py-12 px-6">
          <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start">

            {/* ── LEFT: Readiness content ─────────────────────────────── */}
            <div className="space-y-8">
              {/* Page title */}
              <div>
                <p className="text-[10px] font-mono tracking-[0.16em] mb-3" style={{ color: PALETTE.textSecondary }}>
                  INTERVIEW ROOM
                </p>
                <h1
                  className="text-2xl md:text-3xl font-semibold tracking-tight leading-[1.15]"
                  style={{ color: PALETTE.textPrimary }}
                >
                  Before we begin,
                  <br />
                  let's prepare your session.
                </h1>
              </div>

              {/* ── Resume row ── */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <FileText className="w-4 h-4" style={{ color: PALETTE.textSecondary }} />
                    <span className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>Resume</span>
                  </div>
                  <Pill text="REQUIRED" variant="required" />
                </div>

                {resumeReady ? (
                  <div
                    className="flex items-center gap-3 px-4 py-3 rounded-[8px]"
                    style={{ backgroundColor: 'rgba(126,203,161,0.06)', border: '1px solid rgba(126,203,161,0.18)' }}
                  >
                    <CheckCircle2 className="w-4 h-4 shrink-0" style={{ color: '#7ecba1' }} />
                    <span className="text-sm" style={{ color: PALETTE.textPrimary }}>Resume loaded</span>
                    <button
                      onClick={() => router.push('/resume-job')}
                      className="ml-auto text-[12px] transition-colors hover:text-white"
                      style={{ color: PALETTE.textSecondary }}
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2.5">
                    <p className="text-[13px] leading-relaxed" style={{ color: PALETTE.textSecondary }}>
                      Upload a resume so InterviewPal can tailor questions to your experience.
                    </p>
                    <button
                      onClick={() => router.push('/resume-job')}
                      className="group inline-flex items-center gap-2 px-4 py-2.5 rounded-[8px] text-sm font-medium transition-opacity hover:opacity-90"
                      style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                    >
                      Add your resume
                      <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" />
                    </button>
                  </div>
                )}
              </div>

              {/* ── Divider ── */}
              <div style={{ borderTop: `1px solid ${PALETTE.border}` }} />

              {/* ── Target role row ── */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>Target role</span>
                  </div>
                  <Pill text="OPTIONAL" variant="optional" />
                </div>

                <div className="flex items-center justify-between gap-4">
                  <p className="text-[13px]" style={{ color: roleDetected ? PALETTE.textPrimary : PALETTE.textSecondary }}>
                    {roleDetected
                      ? `${targetJob.title} · ${targetJob.company}`
                      : 'No target role set — questions will be general.'}
                  </p>
                  <button
                    onClick={() => router.push('/resume-job')}
                    className="shrink-0 text-[12px] transition-colors hover:text-white whitespace-nowrap"
                    style={{ color: PALETTE.textSecondary }}
                  >
                    {roleDetected ? 'Change role' : 'Add role'}
                  </button>
                </div>
              </div>

              {/* ── Divider ── */}
              <div style={{ borderTop: `1px solid ${PALETTE.border}` }} />

              {/* ── Session config ── */}
              <div className="space-y-3">
                <p className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>Session</p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: sessionLabel[interviewType] ?? 'Full interview', sub: durationMap[interviewType] ?? '~25 min' },
                    { label: difficulty.charAt(0).toUpperCase() + difficulty.slice(1) + ' difficulty', sub: '' },
                    { label: (voice.charAt(0).toUpperCase() + voice.slice(1)) + ' voice', sub: '' },
                  ].map(({ label, sub }) => (
                    <div
                      key={label}
                      className="px-3 py-2.5 rounded-[8px]"
                      style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
                    >
                      <p className="text-[12px] font-medium" style={{ color: PALETTE.textPrimary }}>{label}</p>
                      {sub && <p className="text-[11px] font-mono mt-0.5" style={{ color: PALETTE.textSecondary }}>{sub}</p>}
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Readiness checklist ── */}
              <ChecklistPanel resumeReady={resumeReady} />

              {/* ── CTA ── */}
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button
                  onClick={() => router.push('/setup')}
                  disabled={!hasInterviewContext}
                  className="group inline-flex items-center gap-2 px-5 py-3 rounded-[10px] text-sm font-semibold transition-opacity disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-90"
                  style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                >
                  Continue setup
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5 group-disabled:translate-x-0" />
                </button>

                {!resumeReady && (
                  <p className="text-[12px]" style={{ color: PALETTE.textSecondary }}>
                    Add your resume to continue.
                  </p>
                )}
                {isAnalyzing && (
                  <p className="text-[12px] font-mono" style={{ color: PALETTE.textSecondary }}>
                    Finishing job analysis...
                  </p>
                )}
              </div>
            </div>

            {/* ── RIGHT: Terminal panel ───────────────────────────────── */}
            <div className="relative lg:min-h-[460px]">
              <ReadinessTerminal resumeReady={resumeReady} roleDetected={roleDetected} />
            </div>

          </div>
        </div>
      </main>
    </div>
  );
};
