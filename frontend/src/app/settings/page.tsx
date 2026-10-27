'use client';

import React, { useState } from 'react';
import {
  Sun, Moon, Monitor, Shield, Trash2, Check, Bell,
  Download, Mic, Volume2, Eye,
} from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { AppBackground } from '@/components/app/AppBackground';
import { PALETTE } from '@/lib/motion';
import { AuthGuard } from '@/components/layout/AuthGuard';

// ─── Design tokens ──────────────────────────────────────────────────────────

const panel = { backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` };

function SectionHeader({ icon: Icon, title, subtitle }: {
  icon: React.ElementType;
  title: string;
  subtitle?: React.ReactNode;
}) {
  return (
    <div className="pb-4 mb-5" style={{ borderBottom: `1px solid ${PALETTE.borderTeal}` }}>
      <div className="flex items-center gap-2.5">
        <Icon className="w-4 h-4 shrink-0" style={{ color: PALETTE.accent }} />
        <h2 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>{title}</h2>
      </div>
      {subtitle && <p className="text-xs mt-1" style={{ color: PALETTE.textSecondary }}>{subtitle}</p>}
    </div>
  );
}

function PlaceholderBadge({ label }: { label?: string }) {
  return (
    <span
      className="text-[10px] font-mono px-1.5 py-0.5 rounded"
      style={{ backgroundColor: 'rgba(245,181,68,0.1)', color: '#f5b544', border: '1px solid rgba(245,181,68,0.2)' }}
    >
      {label ?? 'COMING SOON'}
    </span>
  );
}

function Toggle({
  label,
  description,
  value,
  onChange,
  disabled,
  placeholder,
}: {
  label: string;
  description?: string;
  value: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5" style={{ borderBottom: `1px solid ${PALETTE.border}` }}>
      <div className="min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm" style={{ color: disabled ? PALETTE.textSecondary : PALETTE.textPrimary }}>{label}</p>
          {placeholder && <PlaceholderBadge label={placeholder} />}
        </div>
        {description && <p className="text-xs mt-0.5" style={{ color: PALETTE.textSecondary }}>{description}</p>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full transition-colors shrink-0 disabled:opacity-40 disabled:cursor-not-allowed ${value ? 'bg-white/80' : 'bg-white/15'}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform bg-[#07080a] ${value ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────

function SettingsPageInner() {
  const { resetSession } = useInterview();

  // ── Appearance
  const [theme, setTheme] = useState<'system' | 'light' | 'dark'>('dark');
  const applyTheme = (t: 'system' | 'light' | 'dark') => {
    setTheme(t);
    const root = document.documentElement;
    if (t === 'dark')   { root.setAttribute('data-theme', 'dark'); root.classList.add('dark'); }
    else if (t === 'light') { root.setAttribute('data-theme', 'light'); root.classList.remove('dark'); }
    else                { root.removeAttribute('data-theme'); }
  };

  // ── Accessibility (local)
  const [reducedMotion, setReducedMotion] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [largerText, setLargerText] = useState(false);

  // ── Audio defaults (local — device-specific, never server-side)
  const [autoPlayVoice, setAutoPlayVoice] = useState(true);
  const [captionsDefault, setCaptionsDefault] = useState(true);

  // ── Privacy & data (local toggles)
  const [saveHistory, setSaveHistory] = useState(true);
  const [saveTranscripts, setSaveTranscripts] = useState(true);
  const [saveAudio, setSaveAudio] = useState(false);
  const [resetDone, setResetDone] = useState(false);

  // ── Notifications (placeholder)
  const [notifReport, setNotifReport] = useState(true);
  const [notifSecurity, setNotifSecurity] = useState(true);

  const handleReset = () => {
    resetSession();
    setResetDone(true);
    setTimeout(() => setResetDone(false), 2500);
  };

  return (
    <div className="relative flex-1 flex flex-col min-h-screen">
      <AppBackground />
      <main className="relative z-10 w-full max-w-[1280px] mx-auto px-6 lg:px-10 py-10 pb-16">

        {/* Breadcrumb + heading */}
        <div className="mb-6">
          <p className="text-[11px] font-mono tracking-[0.1em] mb-1" style={{ color: PALETTE.textSecondary }}>
            Account / Settings
          </p>
          <p className="text-sm" style={{ color: PALETTE.textSecondary }}>
            Global app preferences. Interview configuration is set per-session in{' '}
            <a href="/setup" className="underline underline-offset-2" style={{ color: PALETTE.accent }}>Interview Setup</a>.
          </p>
        </div>

        {/* Two-column card grid — lg: 2 cols, mobile: 1 col */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 lg:gap-6">

          {/* ── 1. APPEARANCE ──────────────────────────── col 1, row 1 ── */}
          <section className="rounded-[14px] p-6" style={panel}>
            <SectionHeader icon={Sun} title="Appearance" />
            <div className="grid grid-cols-3 gap-2">
              {([
                { id: 'system', label: 'System', icon: Monitor },
                { id: 'light',  label: 'Light',  icon: Sun },
                { id: 'dark',   label: 'Dark',   icon: Moon },
              ] as const).map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => applyTheme(id)}
                  className="flex flex-col items-center gap-1.5 py-3 rounded-[8px] text-xs transition-colors"
                  style={
                    theme === id
                      ? { backgroundColor: `${PALETTE.accent}12`, border: `1px solid ${PALETTE.accent}55`, color: PALETTE.textPrimary }
                      : { border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }
                  }
                >
                  <Icon className="w-4 h-4" />
                  {label}
                  {theme === id && <Check className="w-3 h-3" style={{ color: PALETTE.accent }} />}
                </button>
              ))}
            </div>
          </section>

          {/* ── 2. ACCESSIBILITY ───────────────────────── col 2, row 1 ── */}
          <section className="rounded-[14px] p-6" style={panel}>
            <SectionHeader
              icon={Eye}
              title="Accessibility"
              subtitle="Visual and motion preferences. Applied globally."
            />
            <div className="space-y-1">
              <Toggle
                label="Reduce motion"
                description="Minimise animations and transitions across the app."
                value={reducedMotion}
                onChange={v => { setReducedMotion(v); document.documentElement.setAttribute('data-reduce-motion', String(v)); }}
              />
              <Toggle
                label="High contrast"
                description="Increase text and border contrast."
                value={highContrast}
                onChange={setHighContrast}
                disabled
                placeholder="COMING SOON"
              />
              <Toggle
                label="Larger text"
                description="Increase base font size."
                value={largerText}
                onChange={setLargerText}
                disabled
                placeholder="COMING SOON"
              />
            </div>
          </section>

          {/* ── 3. AUDIO DEFAULTS ──────────────────────── col 1, row 2 ── */}
          <section className="rounded-[14px] p-6" style={panel}>
            <SectionHeader
              icon={Volume2}
              title="Audio defaults"
              subtitle="Default audio behaviour for every session. Can be overridden in Interview Setup."
            />
            <div className="space-y-1">
              <Toggle
                label="Auto-play AI voice"
                description="AI speaks each question aloud automatically."
                value={autoPlayVoice}
                onChange={setAutoPlayVoice}
              />
              <Toggle
                label="Show captions by default"
                description="Display live subtitles during the interview."
                value={captionsDefault}
                onChange={setCaptionsDefault}
              />
            </div>
            <div className="mt-4 space-y-3">
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <Mic className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
                  <p className="text-xs font-medium" style={{ color: PALETTE.textSecondary }}>Default microphone</p>
                  <PlaceholderBadge />
                </div>
                <select disabled
                  className="w-full text-xs rounded-[8px] px-3 py-2 outline-none opacity-40 cursor-not-allowed"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                >
                  <option>System default</option>
                </select>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <Volume2 className="w-3.5 h-3.5" style={{ color: PALETTE.textSecondary }} />
                  <p className="text-xs font-medium" style={{ color: PALETTE.textSecondary }}>Default speaker</p>
                  <PlaceholderBadge />
                </div>
                <select disabled
                  className="w-full text-xs rounded-[8px] px-3 py-2 outline-none opacity-40 cursor-not-allowed"
                  style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}`, color: PALETTE.textPrimary }}
                >
                  <option>System default</option>
                </select>
              </div>
            </div>
          </section>

          {/* ── 4. PRIVACY & DATA ──────────────────────── col 2, row 2 ── */}
          <section className="rounded-[14px] p-6" style={panel}>
            <SectionHeader
              icon={Shield}
              title="Privacy & data"
              subtitle="No video evaluation or facial analysis is conducted. Audio stays in your browser session."
            />
            <div className="space-y-1">
              <Toggle
                label="Save interview history"
                description="Persist session records and scores across visits."
                value={saveHistory}
                onChange={setSaveHistory}
                placeholder="LOCAL ONLY"
              />
              <Toggle
                label="Save transcripts"
                description="Keep full conversation transcripts for later review."
                value={saveTranscripts}
                onChange={setSaveTranscripts}
                placeholder="LOCAL ONLY"
              />
              <Toggle
                label="Save audio recordings"
                description="Store audio of your answers. Off by default."
                value={saveAudio}
                onChange={setSaveAudio}
                placeholder="LOCAL ONLY"
              />
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-4">
              <button disabled
                className="flex items-center gap-1.5 px-4 py-2 rounded-[8px] text-xs font-medium opacity-40 cursor-not-allowed"
                style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }}
              >
                <Download className="w-3.5 h-3.5" />
                Export data
                <PlaceholderBadge />
              </button>
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 px-4 py-2 rounded-[8px] text-xs font-medium transition-colors hover:bg-red-500/10"
                style={{ border: '1px solid rgba(224,150,120,0.3)', color: '#e09678' }}
              >
                <Trash2 className="w-3.5 h-3.5" />
                {resetDone ? 'Cleared' : 'Clear local history'}
              </button>
            </div>
          </section>

          {/* ── 5. NOTIFICATIONS ──────────── col 1 + col 2 (full) ── */}
          <section className="lg:col-span-2 rounded-[14px] p-6" style={panel}>
            <SectionHeader
              icon={Bell}
              title="Notifications"
              subtitle={<>Email notification preferences. <PlaceholderBadge label="REQUIRES EMAIL SERVICE" /></>}
            />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-10 space-y-1 lg:space-y-0">
              <Toggle
                label="Interview report ready"
                description="Email when your evaluation report is available."
                value={notifReport}
                onChange={setNotifReport}
                disabled
                placeholder="COMING SOON"
              />
              <Toggle
                label="Security alerts"
                description="Email on sign-in from a new device or password change."
                value={notifSecurity}
                onChange={setNotifSecurity}
                disabled
                placeholder="COMING SOON"
              />
            </div>
          </section>

        </div>
      </main>
    </div>
  );
}

export default function SettingsPage() {
  return (
    <AuthGuard>
      <SettingsPageInner />
    </AuthGuard>
  );
}
