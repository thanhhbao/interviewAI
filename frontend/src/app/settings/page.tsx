'use client';

import React, { useState } from 'react';
import {
  User,
  Volume2,
  Mic,
  Moon,
  Sun,
  Shield,
  Trash2,
  Save,
  Check,
  Download,
  Laptop,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { TopHeader } from '@/components/layout/TopHeader';

export default function SettingsPage() {
  const { candidate, setCandidate, voice, setVoice, difficulty, setDifficulty, resetSession } =
    useInterview();

  const [savedToast, setSavedToast] = useState(false);
  const [profileName, setProfileName] = useState(candidate.name);
  const [profileTitle, setProfileTitle] = useState(candidate.title);
  const [profileSummary, setProfileSummary] = useState(candidate.summary);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    setCandidate((prev) => ({
      ...prev,
      name: profileName,
      title: profileTitle,
      summary: profileSummary,
    }));
    setSavedToast(true);
    setTimeout(() => setSavedToast(false), 2500);
  };

  const handleToggleTheme = (theme: 'light' | 'dark') => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen pb-16">
      <TopHeader title="Settings" subtitle="Manage your profile, interview preferences, and hardware" />

      <main className="flex-1 max-w-4xl w-full mx-auto px-8 py-8 space-y-8">
        {/* Toast Notification */}
        {savedToast && (
          <div className="fixed bottom-6 right-6 z-50 px-4 py-2.5 rounded-xl bg-neutral-900 dark:bg-white text-white dark:text-neutral-950 text-xs font-semibold shadow-xl flex items-center gap-2 animate-bounce">
            <Check className="w-4 h-4 text-emerald-400 dark:text-emerald-600" />
            <span>Settings saved successfully</span>
          </div>
        )}

        {/* CANDIDATE PROFILE SECTION */}
        <section className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-5">
          <div className="flex items-center gap-2.5 pb-2 border-b border-neutral-100 dark:border-neutral-800/60">
            <User className="w-4 h-4 text-indigo-500" />
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              Candidate Profile
            </h2>
          </div>

          <form onSubmit={handleSaveProfile} className="space-y-4 text-xs">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-neutral-500 block">Full Name</label>
                <input
                  type="text"
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  className="w-full bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl px-3 py-2 text-neutral-900 dark:text-neutral-100 outline-none focus:border-indigo-500 transition-colors"
                />
              </div>

              <div className="space-y-1">
                <label className="text-neutral-500 block">Current / Target Title</label>
                <input
                  type="text"
                  value={profileTitle}
                  onChange={(e) => setProfileTitle(e.target.value)}
                  className="w-full bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl px-3 py-2 text-neutral-900 dark:text-neutral-100 outline-none focus:border-indigo-500 transition-colors"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-neutral-500 block">Professional Summary</label>
              <textarea
                rows={3}
                value={profileSummary}
                onChange={(e) => setProfileSummary(e.target.value)}
                className="w-full bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl px-3 py-2 text-neutral-900 dark:text-neutral-100 outline-none focus:border-indigo-500 transition-colors leading-relaxed"
              />
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 text-xs font-semibold shadow-xs transition-colors"
              >
                <Save className="w-3.5 h-3.5" />
                <span>Save Profile</span>
              </button>
            </div>
          </form>
        </section>

        {/* INTERVIEW PREFERENCES */}
        <section className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-5">
          <div className="flex items-center gap-2.5 pb-2 border-b border-neutral-100 dark:border-neutral-800/60">
            <Volume2 className="w-4 h-4 text-indigo-500" />
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              Voice & Interview Persona
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="space-y-2">
              <label className="text-neutral-500 block">Default AI Voice</label>
              <div className="space-y-1.5">
                {[
                  { id: 'nova', label: 'Nova (Calm & Analytical)' },
                  { id: 'atlas', label: 'Atlas (Structured & Direct)' },
                  { id: 'aria', label: 'Aria (Warm & Conversational)' },
                ].map((v) => (
                  <button
                    key={v.id}
                    onClick={() => setVoice(v.id as any)}
                    className={`w-full p-2.5 rounded-xl border text-left flex items-center justify-between transition-colors ${
                      voice === v.id
                        ? 'bg-indigo-50 dark:bg-indigo-950/40 border-indigo-500'
                        : 'border-neutral-200/60 dark:border-neutral-800/60 hover:bg-neutral-50 dark:hover:bg-neutral-900'
                    }`}
                  >
                    <span>{v.label}</span>
                    {voice === v.id && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-neutral-500 block">Default Evaluation Rigor</label>
              <div className="space-y-1.5">
                {[
                  { id: 'easy', label: 'Standard Practice' },
                  { id: 'medium', label: 'Realistic Company Standard' },
                  { id: 'hard', label: 'Strict Senior / High-Bar Benchmarking' },
                ].map((d) => (
                  <button
                    key={d.id}
                    onClick={() => setDifficulty(d.id as any)}
                    className={`w-full p-2.5 rounded-xl border text-left flex items-center justify-between transition-colors ${
                      difficulty === d.id
                        ? 'bg-indigo-50 dark:bg-indigo-950/40 border-indigo-500'
                        : 'border-neutral-200/60 dark:border-neutral-800/60 hover:bg-neutral-50 dark:hover:bg-neutral-900'
                    }`}
                  >
                    <span>{d.label}</span>
                    {difficulty === d.id && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* APPEARANCE & THEME */}
        <section className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-5">
          <div className="flex items-center gap-2.5 pb-2 border-b border-neutral-100 dark:border-neutral-800/60">
            <Sun className="w-4 h-4 text-indigo-500" />
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              Appearance & Theme
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => handleToggleTheme('light')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900 text-xs font-medium text-neutral-800 dark:text-neutral-200 transition-colors"
            >
              <Sun className="w-4 h-4 text-amber-500" />
              <span>Light Theme (Off-White)</span>
            </button>

            <button
              onClick={() => handleToggleTheme('dark')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900 text-xs font-medium text-neutral-800 dark:text-neutral-200 transition-colors"
            >
              <Moon className="w-4 h-4 text-indigo-400" />
              <span>Dark Theme (Neutral Zinc)</span>
            </button>
          </div>
        </section>

        {/* PRIVACY & DATA CONTROLS */}
        <section className="p-7 rounded-3xl bg-white dark:bg-[#121215] border border-neutral-200/80 dark:border-neutral-800/80 shadow-xs space-y-4">
          <div className="flex items-center gap-2.5 pb-2 border-b border-neutral-100 dark:border-neutral-800/60">
            <Shield className="w-4 h-4 text-indigo-500" />
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              Privacy & Local Data Controls
            </h2>
          </div>

          <p className="text-xs text-neutral-500 leading-relaxed">
            InterviewAI is candidate-first. No video-evaluation or facial-emotion analysis is
            conducted. All interview audio and transcripts remain in your browser session.
          </p>

          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={resetSession}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-red-200 dark:border-red-900/60 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 text-xs font-medium transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Reset Local Interview History</span>
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
