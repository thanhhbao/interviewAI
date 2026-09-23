'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Mic,
  Volume2,
  Play,
  ArrowRight,
  ShieldCheck,
  Clock,
  Sparkles,
  Layers,
  Activity,
  Check,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { InterviewType, InterviewDifficulty, InterviewVoice } from '@/lib/types/interview';
import { TopHeader } from '@/components/layout/TopHeader';

export default function InterviewSetupPage() {
  const router = useRouter();
  const {
    candidate,
    targetJob,
    interviewType,
    setInterviewType,
    difficulty,
    setDifficulty,
    voice,
    setVoice,
    startInterviewSession,
  } = useInterview();

  // Audio testing states
  const [isTestingMic, setIsTestingMic] = useState(false);
  const [micVolume, setMicVolume] = useState(0);
  const [isPlayingVoice, setIsPlayingVoice] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');

  // Simulate Mic Volume Meter during test
  useEffect(() => {
    let animInterval: NodeJS.Timeout;
    if (isTestingMic) {
      animInterval = setInterval(() => {
        setMicVolume(Math.floor(Math.random() * 75) + 15);
      }, 100);
    } else {
      setMicVolume(0);
    }
    return () => clearInterval(animInterval);
  }, [isTestingMic]);

  // Preview Voice Action
  const handlePreviewVoice = () => {
    setIsPlayingVoice(true);
    const sampleText =
      voice === 'nova'
        ? 'Hello Alex. I am Nova, your AI interviewer today. I am calm, thorough, and ready whenever you are.'
        : voice === 'atlas'
        ? 'Welcome Alex. I am Atlas. We will evaluate your technical architecture, systems design, and problem solving.'
        : 'Hi Alex! I am Aria. We will focus on your background, communication, and engineering leadership.';

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(sampleText);
      utterance.rate = 1.0;
      utterance.onend = () => setIsPlayingVoice(false);
      utterance.onerror = () => setIsPlayingVoice(false);
      window.speechSynthesis.speak(utterance);
    } else {
      setTimeout(() => setIsPlayingVoice(false), 3000);
    }
  };

  const handleStartInterview = () => {
    startInterviewSession(interviewType);
    router.push('/interview');
  };

  const interviewTypes: Array<{
    id: InterviewType;
    label: string;
    desc: string;
    duration: string;
  }> = [
    {
      id: 'full',
      label: 'Full Comprehensive Interview',
      desc: 'Behavioral, technical deep dive, system design, and coding task.',
      duration: '~25 min',
    },
    {
      id: 'technical',
      label: 'Technical & System Design',
      desc: 'Focus on distributed systems, databases, and architectural trade-offs.',
      duration: '~18 min',
    },
    {
      id: 'coding',
      label: 'Coding Interview Mode',
      desc: 'Hands-on live algorithmic coding problem in our Monaco IDE.',
      duration: '~15 min',
    },
    {
      id: 'behavioral',
      label: 'Behavioral & Leadership',
      desc: 'STAR framework evaluation, past projects, and teamwork dynamics.',
      duration: '~12 min',
    },
    {
      id: 'quick',
      label: 'Quick Warm-up Practice',
      desc: 'A fast 2-question session to get into interview flow.',
      duration: '~8 min',
    },
  ];

  const difficultyLevels: Array<{ id: InterviewDifficulty; label: string }> = [
    { id: 'easy', label: 'Standard' },
    { id: 'medium', label: 'Medium (Realistic)' },
    { id: 'hard', label: 'Hard (Strict FAANG / High Bar)' },
  ];

  const voiceOptions: Array<{ id: InterviewVoice; label: string; desc: string }> = [
    { id: 'nova', label: 'Nova', desc: 'Calm, clear, and analytical' },
    { id: 'atlas', label: 'Atlas', desc: 'Professional, direct, and structured' },
    { id: 'aria', label: 'Aria', desc: 'Supportive, conversational, and warm' },
  ];

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      <TopHeader
        title="Interview Setup"
        subtitle="Configure your AI interviewer persona, difficulty, and hardware"
        showBack={true}
        backHref="/"
      />

      <main className="flex-1 max-w-4xl w-full mx-auto px-8 py-8 space-y-8">
        {/* CONTEXT CARD */}
        <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 flex items-center justify-between text-xs">
          <div className="space-y-1">
            <span className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider block">
              Session Context
            </span>
            <p className="font-semibold text-neutral-900 dark:text-neutral-100">
              {candidate.name} · {targetJob.title}
            </p>
            <p className="text-neutral-500">
              Resume: {candidate.resumeFileName} · Target: {targetJob.company}
            </p>
          </div>

          <Link
            href="/resume-job"
            className="px-3 py-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 text-neutral-700 dark:text-neutral-300 font-medium hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
          >
            Change Context
          </Link>
        </div>

        {/* INTERVIEW TYPE SELECTOR */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
            Interview Type
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {interviewTypes.map((t) => {
              const isSelected = interviewType === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setInterviewType(t.id)}
                  className={`p-4 rounded-2xl border text-left transition-all ${
                    isSelected
                      ? 'bg-indigo-50/50 dark:bg-indigo-950/40 border-indigo-500/80 ring-1 ring-indigo-500/50'
                      : 'bg-white dark:bg-[#121215] border-neutral-200/80 dark:border-neutral-800/80 hover:border-neutral-300 dark:hover:border-neutral-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                      {t.label}
                    </span>
                    <span className="text-[11px] font-mono text-neutral-400">{t.duration}</span>
                  </div>
                  <p className="text-[11px] text-neutral-500 dark:text-neutral-400 mt-1">
                    {t.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* DIFFICULTY SELECTOR */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
            Difficulty Level
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {difficultyLevels.map((d) => {
              const isSelected = difficulty === d.id;
              return (
                <button
                  key={d.id}
                  onClick={() => setDifficulty(d.id)}
                  className={`p-3 rounded-2xl border text-center transition-all ${
                    isSelected
                      ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950 font-semibold border-transparent shadow-xs'
                      : 'bg-white dark:bg-[#121215] text-neutral-700 dark:text-neutral-300 border-neutral-200/80 dark:border-neutral-800/80 hover:border-neutral-300'
                  }`}
                >
                  <span className="text-xs">{d.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* VOICE & HARDWARE SETTINGS */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* AI Voice Selector */}
          <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                <Volume2 className="w-4 h-4 text-indigo-500" />
                AI Interviewer Voice
              </span>

              <button
                onClick={handlePreviewVoice}
                disabled={isPlayingVoice}
                className="text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 flex items-center gap-1 transition-colors"
              >
                <Play className="w-3 h-3 fill-current" />
                <span>{isPlayingVoice ? 'Playing...' : 'Preview Voice'}</span>
              </button>
            </div>

            <div className="space-y-2">
              {voiceOptions.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setVoice(v.id)}
                  className={`w-full p-2.5 rounded-xl border flex items-center justify-between text-left text-xs transition-colors ${
                    voice === v.id
                      ? 'bg-indigo-50/60 dark:bg-indigo-950/40 border-indigo-400'
                      : 'border-neutral-200/60 dark:border-neutral-800/60 hover:bg-neutral-50 dark:hover:bg-neutral-900/60'
                  }`}
                >
                  <div>
                    <span className="font-semibold text-neutral-900 dark:text-neutral-100">
                      {v.label}
                    </span>
                    <span className="text-[11px] text-neutral-400 block">{v.desc}</span>
                  </div>
                  {voice === v.id && <Check className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />}
                </button>
              ))}
            </div>
          </div>

          {/* Microphone & Hardware Check */}
          <div className="p-5 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 space-y-4 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                  <Mic className="w-4 h-4 text-indigo-500" />
                  Microphone & Audio Input
                </span>
                <span className="text-[10px] text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/60 px-2 py-0.5 rounded-full font-medium">
                  Active
                </span>
              </div>

              <div className="space-y-2">
                <label className="text-[11px] text-neutral-400 block">Default Input Device</label>
                <select className="w-full bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-200 text-xs rounded-xl px-3 py-2 outline-none">
                  <option>Built-in Microphone (MacBook Pro)</option>
                  <option>External Studio USB Mic</option>
                  <option>AirPods Pro Bluetooth Mic</option>
                </select>
              </div>

              {/* Live Volume Meter */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-[11px] text-neutral-400">
                  <span>Input Signal</span>
                  <span>{isTestingMic ? `${micVolume}%` : 'Idle'}</span>
                </div>
                <div className="w-full h-2 rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all duration-100"
                    style={{ width: `${micVolume}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setIsTestingMic(!isTestingMic)}
              className="w-full py-2 rounded-xl border border-neutral-200 dark:border-neutral-800 text-neutral-700 dark:text-neutral-300 text-xs font-medium hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
            >
              {isTestingMic ? 'Stop Mic Test' : 'Test Microphone'}
            </button>
          </div>
        </div>

        {/* START INTERVIEW CTA */}
        <div className="pt-4 flex items-center justify-between">
          <p className="text-xs text-neutral-400 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-neutral-400" />
            <span>Push-to-talk audio · Real-time 3D animation · Instant structured feedback</span>
          </p>

          <button
            onClick={handleStartInterview}
            className="px-8 py-3.5 rounded-2xl bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 text-xs font-semibold shadow-lg flex items-center gap-2 transition-all active:scale-98"
          >
            <span>Start Interview Room</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </main>
    </div>
  );
}
