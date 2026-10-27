'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Mic, Volume2, Play, ArrowRight, ShieldCheck, Check,
  BrainCircuit, Settings2,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { InterviewType, InterviewDifficulty, InterviewVoice } from '@/lib/types/interview';
import { AppBackground } from '@/components/app/AppBackground';
import { InterviewReadinessScreen } from '@/components/interview/InterviewReadinessScreen';
import { PALETTE } from '@/lib/motion';

const panelStyle = { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` };

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: PALETTE.textSecondary }}>
      {children}
    </h3>
  );
}

// Mutually exclusive row of options
function OptionGrid<T extends string>({
  value,
  onChange,
  cols,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  cols?: number;
  options: { id: T; label: string; desc?: string; meta?: string }[];
}) {
  const colClass = cols === 2 ? 'grid-cols-2' : cols === 4 ? 'grid-cols-4' : 'grid-cols-3';
  return (
    <div className={`grid ${colClass} gap-2`}>
      {options.map(opt => {
        const sel = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            className="p-3 rounded-[10px] text-left transition-all"
            style={
              sel
                ? { backgroundColor: `${PALETTE.accent}14`, border: `1px solid ${PALETTE.accent}80` }
                : { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }
            }
          >
            <div className="flex items-center justify-between gap-1">
              <span className="text-xs font-semibold" style={{ color: PALETTE.textPrimary }}>{opt.label}</span>
              {opt.meta && <span className="text-[10px] font-mono shrink-0" style={{ color: PALETTE.textSecondary }}>{opt.meta}</span>}
            </div>
            {opt.desc && <p className="text-[11px] mt-0.5" style={{ color: PALETTE.textSecondary }}>{opt.desc}</p>}
          </button>
        );
      })}
    </div>
  );
}

function Toggle({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description?: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5" style={{ borderBottom: `1px solid ${PALETTE.border}` }}>
      <div>
        <p className="text-sm" style={{ color: PALETTE.textPrimary }}>{label}</p>
        {description && <p className="text-xs mt-0.5" style={{ color: PALETTE.textSecondary }}>{description}</p>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full transition-colors shrink-0 ${value ? 'bg-white/80' : 'bg-white/15'}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform bg-[#07080a] ${value ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
    </div>
  );
}

export default function InterviewSetupPage() {
  const router = useRouter();
  const {
    candidate,
    targetJob,
    resumeAnalysis,
    hasInterviewContext,
    hasTargetJobContext,
    interviewType,
    setInterviewType,
    difficulty,
    setDifficulty,
    voice,
    setVoice,
    startInterviewSession,
  } = useInterview();

  // ── Per-session interview configuration (local — not in context yet)
  const [interviewLang, setInterviewLang] = useState('English');
  const [interviewerStyle, setInterviewerStyle] = useState<'formal' | 'conversational' | 'technical'>('conversational');
  const [strictness, setStrictness] = useState<'lenient' | 'standard' | 'strict'>('standard');
  const [followUpDepth, setFollowUpDepth] = useState<'minimal' | 'moderate' | 'deep'>('moderate');
  const [adaptiveFollowUp, setAdaptiveFollowUp] = useState(true);
  const [aiStance, setAiStance] = useState<'supportive' | 'balanced' | 'challenging'>('balanced');

  // ── Audio testing
  const [isTestingMic, setIsTestingMic] = useState(false);
  const [micVolume, setMicVolume] = useState(0);
  const [isPlayingVoice, setIsPlayingVoice] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isTestingMic) {
      interval = setInterval(() => setMicVolume(Math.floor(Math.random() * 75) + 15), 100);
    } else {
      setMicVolume(0);
    }
    return () => clearInterval(interval);
  }, [isTestingMic]);

  const handlePreviewVoice = () => {
    setIsPlayingVoice(true);
    const text =
      voice === 'nova'
        ? 'Hello. I am Nova, your AI interviewer today. I am calm, thorough, and ready whenever you are.'
        : voice === 'atlas'
        ? 'Welcome. I am Atlas. We will evaluate your technical architecture, systems design, and problem solving.'
        : 'Hi! I am Aria. We will focus on your background, communication, and engineering leadership.';
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 1.0;
      utt.onend = () => setIsPlayingVoice(false);
      utt.onerror = () => setIsPlayingVoice(false);
      window.speechSynthesis.speak(utt);
    } else {
      setTimeout(() => setIsPlayingVoice(false), 3000);
    }
  };

  const handleStartInterview = () => {
    startInterviewSession(interviewType);
    router.push('/interview');
  };

  if (!hasInterviewContext) return <InterviewReadinessScreen />;

  const LANGS = ['English', 'Vietnamese', 'Japanese', 'Korean', 'Chinese (Simplified)', 'French', 'German', 'Spanish'];

  const interviewTypes: Array<{ id: InterviewType; label: string; desc: string; meta: string }> = [
    { id: 'full',       label: 'Full Interview',         desc: 'Behavioral, technical, system design, and coding.', meta: '~25 min' },
    { id: 'technical',  label: 'Technical & Design',     desc: 'Distributed systems, databases, architecture.',     meta: '~18 min' },
    { id: 'coding',     label: 'Coding',                 desc: 'Live algorithmic problem in the Monaco IDE.',       meta: '~15 min' },
    { id: 'behavioral', label: 'Behavioral',             desc: 'STAR framework, leadership, teamwork.',             meta: '~12 min' },
    { id: 'quick',      label: 'Quick Warm-up',          desc: '2-question session to get into flow.',              meta: '~8 min' },
  ];

  return (
    <div className="relative flex-1 flex flex-col">
      <AppBackground />
      <main className="relative flex-1 max-w-4xl w-full mx-auto px-6 md:px-8 py-10 md:py-14 space-y-10">

        {/* PAGE INTRO */}
        <div className="max-w-2xl space-y-2">
          <p className="text-[11px] font-mono tracking-[0.1em]" style={{ color: PALETTE.textSecondary }}>
            Interview preparation / Setup
          </p>
          <h1 className="text-2xl md:text-[28px] font-semibold tracking-tight" style={{ color: PALETTE.textPrimary }}>
            Interview Setup
          </h1>
          <p className="text-sm leading-relaxed" style={{ color: PALETTE.textSecondary }}>
            Configure this session. Every setting here applies only to the interview you are about to start.
          </p>
        </div>

        {/* CONTEXT CARD */}
        <div className="p-5 rounded-[14px] flex items-center justify-between text-xs" style={panelStyle}>
          <div className="space-y-1 min-w-0">
            <span className="text-[11px] font-semibold uppercase tracking-wider block" style={{ color: PALETTE.textSecondary }}>
              Session context
            </span>
            {hasTargetJobContext ? (
              <>
                <p className="font-semibold truncate" style={{ color: PALETTE.textPrimary }}>{targetJob.title}</p>
                <p style={{ color: PALETTE.textSecondary }}>{targetJob.company} · Resume + Job targeted</p>
              </>
            ) : (
              <>
                <p className="font-semibold truncate" style={{ color: PALETTE.textPrimary }}>{resumeAnalysis.inferredRole || 'Resume-based interview'}</p>
                <p style={{ color: PALETTE.textSecondary }}>Resume: {candidate.resumeFileName}</p>
              </>
            )}
          </div>
          <Link
            href="/resume-job"
            className="ml-4 shrink-0 px-3 py-1.5 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.04]"
            style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
          >
            Change
          </Link>
        </div>

        {/* ── 1. INTERVIEW TYPE ─────────────────────────────────────────── */}
        <div className="space-y-3">
          <SectionLabel>Interview type</SectionLabel>
          <OptionGrid
            value={interviewType}
            onChange={setInterviewType}
            cols={2}
            options={interviewTypes}
          />
        </div>

        {/* ── 2. SESSION CONFIGURATION ──────────────────────────────────── */}
        <div className="space-y-5">
          <SectionLabel>Session configuration</SectionLabel>

          {/* Language + difficulty on one row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Interview language</p>
              <select
                className="w-full text-xs rounded-[8px] px-3 py-2.5 outline-none cursor-pointer"
                style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary, appearance: 'none' }}
                value={interviewLang}
                onChange={e => setInterviewLang(e.target.value)}
              >
                {LANGS.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            </div>
            <div>
              <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Difficulty</p>
              <div className="grid grid-cols-3 gap-2 h-[42px]">
                {([
                  { id: 'easy' as InterviewDifficulty, label: 'Standard' },
                  { id: 'medium' as InterviewDifficulty, label: 'Medium' },
                  { id: 'hard' as InterviewDifficulty, label: 'Strict' },
                ] as const).map(d => (
                  <button
                    key={d.id}
                    type="button"
                    onClick={() => setDifficulty(d.id)}
                    className="rounded-[8px] text-xs font-medium transition-all h-full"
                    style={
                      difficulty === d.id
                        ? { backgroundColor: `${PALETTE.accent}14`, border: `1px solid ${PALETTE.accent}80`, color: PALETTE.textPrimary }
                        : { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }
                    }
                  >
                    {d.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Interviewer style */}
          <div>
            <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Interviewer style</p>
            <OptionGrid
              value={interviewerStyle}
              onChange={setInterviewerStyle}
              cols={3}
              options={[
                { id: 'formal',        label: 'Formal',         desc: 'Professional, structured pace' },
                { id: 'conversational',label: 'Conversational', desc: 'Relaxed, exploratory' },
                { id: 'technical',     label: 'Technical',      desc: 'Deep dives, precise probing' },
              ]}
            />
          </div>

          {/* Strictness */}
          <div>
            <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Evaluation strictness</p>
            <OptionGrid
              value={strictness}
              onChange={setStrictness}
              cols={3}
              options={[
                { id: 'lenient',  label: 'Lenient',  desc: 'Generous scoring' },
                { id: 'standard', label: 'Standard', desc: 'Balanced evaluation' },
                { id: 'strict',   label: 'Strict',   desc: 'No free passes' },
              ]}
            />
          </div>
        </div>

        {/* ── 3. SESSION BEHAVIOUR ──────────────────────────────────────── */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <SectionLabel>Session behaviour</SectionLabel>
            <BrainCircuit className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
          </div>

          {/* Follow-up depth */}
          <div>
            <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>Follow-up depth</p>
            <OptionGrid
              value={followUpDepth}
              onChange={setFollowUpDepth}
              cols={3}
              options={[
                { id: 'minimal',  label: 'Minimal',  desc: 'Move on quickly' },
                { id: 'moderate', label: 'Moderate', desc: '1–2 follow-ups' },
                { id: 'deep',     label: 'Deep',     desc: 'Probe every answer' },
              ]}
            />
          </div>

          {/* AI stance */}
          <div>
            <p className="text-[11px] mb-2" style={{ color: PALETTE.textSecondary }}>AI interviewer stance</p>
            <OptionGrid
              value={aiStance}
              onChange={setAiStance}
              cols={3}
              options={[
                { id: 'supportive',  label: 'Supportive',  desc: 'Encouraging, hints on request' },
                { id: 'balanced',    label: 'Balanced',    desc: 'Neutral, realistic' },
                { id: 'challenging', label: 'Challenging', desc: 'Pushback, high expectations' },
              ]}
            />
          </div>

          {/* Toggles */}
          <div className="p-4 rounded-[10px]" style={panelStyle}>
            <Toggle
              label="Adaptive follow-up"
              description="AI adjusts follow-up depth based on the quality of your previous answer."
              value={adaptiveFollowUp}
              onChange={setAdaptiveFollowUp}
            />
          </div>
        </div>

        {/* ── 4. VOICE & HARDWARE ───────────────────────────────────────── */}
        <div className="space-y-3">
          <SectionLabel>Voice & hardware</SectionLabel>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* AI Voice */}
            <div className="p-5 rounded-[14px] space-y-4" style={panelStyle}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold flex items-center gap-2" style={{ color: PALETTE.textPrimary }}>
                  <Volume2 className="w-4 h-4" style={{ color: PALETTE.accent }} />
                  AI voice
                </span>
                <button
                  onClick={handlePreviewVoice}
                  disabled={isPlayingVoice}
                  className="text-[11px] font-semibold flex items-center gap-1 transition-opacity disabled:opacity-50"
                  style={{ color: PALETTE.accent }}
                >
                  <Play className="w-3 h-3 fill-current" />
                  {isPlayingVoice ? 'Playing…' : 'Preview'}
                </button>
              </div>
              <div className="space-y-2">
                {([
                  { id: 'nova' as InterviewVoice,  label: 'Nova',  desc: 'Calm, clear, analytical' },
                  { id: 'atlas' as InterviewVoice, label: 'Atlas', desc: 'Professional, direct, structured' },
                  { id: 'aria' as InterviewVoice,  label: 'Aria',  desc: 'Supportive, conversational, warm' },
                ] as const).map(v => (
                  <button
                    key={v.id}
                    onClick={() => setVoice(v.id)}
                    className="w-full p-2.5 rounded-[8px] flex items-center justify-between text-left text-xs transition-colors"
                    style={
                      voice === v.id
                        ? { backgroundColor: `${PALETTE.accent}14`, border: `1px solid ${PALETTE.accent}80` }
                        : { border: `1px solid ${PALETTE.border}` }
                    }
                  >
                    <div>
                      <span className="font-semibold" style={{ color: PALETTE.textPrimary }}>{v.label}</span>
                      <span className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>{v.desc}</span>
                    </div>
                    {voice === v.id && <Check className="w-4 h-4 shrink-0" style={{ color: PALETTE.accent }} />}
                  </button>
                ))}
              </div>
            </div>

            {/* Microphone check */}
            <div className="p-5 rounded-[14px] space-y-4 flex flex-col justify-between" style={panelStyle}>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold flex items-center gap-2" style={{ color: PALETTE.textPrimary }}>
                    <Mic className="w-4 h-4" style={{ color: PALETTE.accent }} />
                    Microphone check
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ backgroundColor: `${PALETTE.accent}1F`, color: PALETTE.accent }}>
                    Active
                  </span>
                </div>
                <div className="space-y-2">
                  <label className="text-[11px] block" style={{ color: PALETTE.textSecondary }}>Input device</label>
                  <select
                    className="w-full text-xs rounded-[8px] px-3 py-2 outline-none"
                    style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                  >
                    <option>System default microphone</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-[11px]" style={{ color: PALETTE.textSecondary }}>
                    <span>Input level</span>
                    <span>{isTestingMic ? `${micVolume}%` : 'Idle'}</span>
                  </div>
                  <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}>
                    <div className="h-full transition-all duration-100" style={{ width: `${micVolume}%`, backgroundColor: PALETTE.accent }} />
                  </div>
                </div>
              </div>
              <button
                onClick={() => setIsTestingMic(v => !v)}
                className="w-full py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-white/[0.04]"
                style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
              >
                {isTestingMic ? 'Stop test' : 'Test microphone'}
              </button>
            </div>
          </div>
        </div>

        {/* ── START CTA ─────────────────────────────────────────────────── */}
        <div className="pt-2 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <p className="text-xs flex items-center gap-2" style={{ color: PALETTE.textSecondary }}>
            <ShieldCheck className="w-4 h-4" />
            Push-to-talk · Real-time feedback · Structured evaluation report
          </p>
          <button
            onClick={handleStartInterview}
            className="px-6 py-3 rounded-[10px] text-sm font-semibold shrink-0 flex items-center gap-2 transition-opacity hover:opacity-90 active:opacity-80"
            style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
          >
            Start Interview Room
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

      </main>
    </div>
  );
}
