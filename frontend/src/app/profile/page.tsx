'use client';

import React, { useState, useEffect } from 'react';
import {
  User, Mail, Lock, LogOut, Link2, Github, Linkedin, Globe,
  ShieldCheck, ShieldAlert, Trash2, Check, AlertCircle,
  ChevronDown, Pencil, X,
} from 'lucide-react';
import { createClient } from '@/lib/supabase/client';
import { useAuth } from '@/context/AuthContext';
import { AuthGuard } from '@/components/layout/AuthGuard';
import { AppBackground } from '@/components/app/AppBackground';
import { PALETTE } from '@/lib/motion';

// ─── Design tokens ──────────────────────────────────────────────────────────

const panel = {
  backgroundColor: PALETTE.surface,
  border: `1px solid ${PALETTE.border}`,
};
const inputCls = 'w-full rounded-[8px] px-3 py-2.5 text-sm outline-none transition-colors focus:border-white/20';
const inputStyle = {
  backgroundColor: PALETTE.surface2,
  border: `1px solid ${PALETTE.border}`,
  color: PALETTE.textPrimary,
};
const labelCls = 'block text-xs mb-1.5';

// ─── Primitives ──────────────────────────────────────────────────────────────

function SectionHeader({ icon: Icon, title, subtitle }: {
  icon: React.ElementType;
  title: string;
  subtitle?: string;
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

function FieldMsg({ msg }: { msg: { text: string; type: 'ok' | 'err' } | null }) {
  if (!msg) return null;
  return (
    <div className={`flex items-center gap-2 text-[13px] rounded-lg px-3 py-2 ${
      msg.type === 'ok'
        ? 'text-emerald-400 bg-emerald-950/40 border border-emerald-900/50'
        : 'text-red-400 bg-red-950/40 border border-red-900/50'
    }`}>
      {msg.type === 'ok' ? <Check className="w-3.5 h-3.5 shrink-0" /> : <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
      <span>{msg.text}</span>
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

/** Display a value in read mode — shows a placeholder dash when empty */
function ReadValue({ value, placeholder }: { value: string; placeholder?: string }) {
  return value
    ? <p className="text-sm" style={{ color: PALETTE.textPrimary }}>{value}</p>
    : <p className="text-sm italic" style={{ color: PALETTE.textSecondary }}>{placeholder ?? '—'}</p>;
}

// ─── Main page ───────────────────────────────────────────────────────────────

function ProfilePageInner() {
  const { user, signOut } = useAuth();
  const supabase = createClient();
  const meta = user?.user_metadata ?? {};

  // ── Edit modes
  const [editingProfile, setEditingProfile] = useState(false);
  const [editingEmail,   setEditingEmail]   = useState(false);
  const [showPwForm,     setShowPwForm]     = useState(false);

  // ── Form state (identity + professional combined)
  const emptyProf = () => ({
    displayName:        meta.full_name          ?? '',
    current_role:       meta.current_role       ?? '',
    target_role:        meta.target_role        ?? '',
    years_experience:   meta.years_experience   ?? '',
    interview_language: meta.interview_language ?? 'English',
    github_url:         meta.github_url         ?? '',
    linkedin_url:       meta.linkedin_url       ?? '',
    portfolio_url:      meta.portfolio_url      ?? '',
  });
  const [draft, setDraft] = useState(emptyProf);

  // ── Save state
  const [saving,  setSaving]  = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ text: string; type: 'ok' | 'err' } | null>(null);

  // ── Email change
  const [newEmail,    setNewEmail]    = useState('');
  const [emailMsg,    setEmailMsg]    = useState<{ text: string; type: 'ok' | 'err' } | null>(null);
  const [emailSaving, setEmailSaving] = useState(false);

  // ── Password change
  const [newPw,    setNewPw]    = useState('');
  const [confirmPw,setConfirmPw]= useState('');
  const [pwMsg,    setPwMsg]    = useState<{ text: string; type: 'ok' | 'err' } | null>(null);
  const [pwSaving, setPwSaving] = useState(false);

  // ── Danger zone
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  useEffect(() => {
    if (!user) return;
    setDraft(emptyProf());
  }, [user?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const initials = (meta.full_name || user?.email || '?')
    .split(/[\s@]/).filter(Boolean).slice(0, 2)
    .map((s: string) => s[0].toUpperCase()).join('');

  const isEmailVerified = !!user?.email_confirmed_at;
  const providers = (user?.identities ?? []).map((i: { provider: string }) => i.provider);
  const LANGS = ['English', 'Vietnamese', 'Japanese', 'Korean', 'Chinese (Simplified)', 'French', 'German', 'Spanish'];

  function startEdit() {
    setDraft(emptyProf());
    setSaveMsg(null);
    setEditingProfile(true);
  }

  function cancelEdit() {
    setDraft(emptyProf());
    setSaveMsg(null);
    setEditingProfile(false);
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true); setSaveMsg(null);
    const { displayName, ...profFields } = draft;
    const { error } = await supabase.auth.updateUser({
      data: { full_name: displayName, ...profFields },
    });
    setSaveMsg(error
      ? { text: error.message, type: 'err' }
      : { text: 'Profile saved.', type: 'ok' });
    if (!error) setEditingProfile(false);
    setSaving(false);
  }

  async function handleEmailChange(e: React.FormEvent) {
    e.preventDefault();
    if (!newEmail.trim()) return;
    setEmailSaving(true); setEmailMsg(null);
    const { error } = await supabase.auth.updateUser({ email: newEmail });
    setEmailMsg(error
      ? { text: error.message, type: 'err' }
      : { text: 'Confirmation sent — check your new inbox.', type: 'ok' });
    if (!error) { setNewEmail(''); setEditingEmail(false); }
    setEmailSaving(false);
  }

  async function handlePasswordChange(e: React.FormEvent) {
    e.preventDefault();
    if (newPw !== confirmPw) { setPwMsg({ text: 'Passwords do not match.', type: 'err' }); return; }
    if (newPw.length < 6)   { setPwMsg({ text: 'Minimum 6 characters.', type: 'err' }); return; }
    setPwSaving(true); setPwMsg(null);
    const { error } = await supabase.auth.updateUser({ password: newPw });
    if (!error) { setNewPw(''); setConfirmPw(''); setShowPwForm(false); }
    setPwMsg(error
      ? { text: error.message, type: 'err' }
      : { text: 'Password updated.', type: 'ok' });
    setPwSaving(false);
  }

  return (
    <div className="relative flex-1 flex flex-col min-h-screen">
      <AppBackground />

      <main className="relative z-10 w-full max-w-[1280px] mx-auto px-6 lg:px-10 py-10 pb-16">

        {/* Breadcrumb */}
        <p className="text-[11px] font-mono tracking-[0.1em] mb-6" style={{ color: PALETTE.textSecondary }}>
          Account / Profile
        </p>

        <form onSubmit={handleSave}>
          {/* Save / Edit bar — only visible when editing */}
          {editingProfile && (
            <div className="flex items-center justify-between mb-5 px-4 py-3 rounded-[10px]"
              style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}>
              <span className="text-xs" style={{ color: PALETTE.textSecondary }}>
                Editing profile — changes won't be saved until you click Save
              </span>
              <div className="flex items-center gap-2">
                {saveMsg && (
                  <span className={`text-xs flex items-center gap-1.5 ${saveMsg.type === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {saveMsg.type === 'ok' ? <Check className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                    {saveMsg.text}
                  </span>
                )}
                <button
                  type="button"
                  onClick={cancelEdit}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs font-medium border transition-colors hover:bg-white/[0.03]"
                  style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-[8px] text-xs font-medium transition-opacity hover:opacity-90 disabled:opacity-40"
                  style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                >
                  <Check className="w-3.5 h-3.5" />
                  {saving ? 'Saving…' : 'Save changes'}
                </button>
              </div>
            </div>
          )}

          {/* Two-column card grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 lg:gap-6">

            {/* ── 1. IDENTITY ────────────────────────── col 1, row 1 ── */}
            <section className="rounded-[14px] p-6" style={panel}>
              <div className="flex items-start justify-between mb-0">
                <SectionHeader icon={User} title="Identity" />
                {!editingProfile && (
                  <button
                    type="button"
                    onClick={startEdit}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-[8px] border transition-colors hover:bg-white/[0.03] -mt-1"
                    style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                  >
                    <Pencil className="w-3 h-3" />
                    Edit
                  </button>
                )}
              </div>

              {/* Avatar + account summary */}
              <div className="flex items-center gap-3 mb-5">
                <div
                  className="w-11 h-11 rounded-full flex items-center justify-center text-sm font-semibold shrink-0 select-none"
                  style={{
                    background: 'linear-gradient(135deg, #1a2e2a 0%, #0d1f1c 100%)',
                    color: '#E7E8E9',
                    border: `1px solid ${PALETTE.border}`,
                  }}
                >
                  {initials}
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate" style={{ color: PALETTE.textPrimary }}>
                    {meta.full_name || user?.email?.split('@')[0] || '—'}
                  </p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <p className="text-xs truncate" style={{ color: PALETTE.textSecondary }}>{user?.email}</p>
                    {isEmailVerified
                      ? <ShieldCheck className="w-3 h-3 shrink-0 text-emerald-400" />
                      : <ShieldAlert className="w-3 h-3 shrink-0 text-amber-400" />}
                    <span className={`text-[10px] ${isEmailVerified ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {isEmailVerified ? 'verified' : 'unverified'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Display name */}
              <div>
                <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Display name</label>
                {editingProfile ? (
                  <input
                    className={inputCls} style={inputStyle}
                    placeholder="Your name"
                    value={draft.displayName}
                    onChange={e => setDraft(d => ({ ...d, displayName: e.target.value }))}
                  />
                ) : (
                  <ReadValue value={meta.full_name ?? ''} placeholder="Not set" />
                )}
              </div>

              {/* Email change */}
              <div className="mt-5 pt-5" style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs font-medium" style={{ color: PALETTE.textSecondary }}>Email address</p>
                  {!editingEmail && (
                    <button
                      type="button"
                      onClick={() => { setEditingEmail(true); setEmailMsg(null); }}
                      className="text-[11px] px-2.5 py-1 rounded-[6px] border transition-colors hover:bg-white/[0.03]"
                      style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                    >
                      Change
                    </button>
                  )}
                </div>
                <p className="text-sm mb-3" style={{ color: PALETTE.textPrimary }}>{user?.email}</p>

                {editingEmail && (
                  <form onSubmit={handleEmailChange} className="space-y-3">
                    <input
                      type="email" className={inputCls} style={inputStyle}
                      placeholder="New email address"
                      value={newEmail} onChange={e => setNewEmail(e.target.value)}
                      autoFocus
                    />
                    {emailMsg && <FieldMsg msg={emailMsg} />}
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => { setEditingEmail(false); setNewEmail(''); setEmailMsg(null); }}
                        className="px-3 py-1.5 rounded-[8px] text-xs border transition-colors hover:bg-white/[0.03]"
                        style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={emailSaving || !newEmail.trim()}
                        className="px-4 py-1.5 rounded-[8px] text-xs font-medium transition-opacity hover:opacity-90 disabled:opacity-40"
                        style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                      >
                        {emailSaving ? 'Sending…' : 'Confirm change'}
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </section>

            {/* ── 2. PROFESSIONAL PROFILE ──────────── col 2, row 1 ── */}
            <section className="rounded-[14px] p-6" style={panel}>
              <div className="flex items-start justify-between mb-0">
                <SectionHeader
                  icon={User}
                  title="Professional profile"
                  subtitle="Used to personalise your interview sessions."
                />
                {!editingProfile && (
                  <button
                    type="button"
                    onClick={startEdit}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-[8px] border transition-colors hover:bg-white/[0.03] -mt-1"
                    style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                  >
                    <Pencil className="w-3 h-3" />
                    Edit
                  </button>
                )}
              </div>

              <div className="space-y-4">
                {/* Current role + Target role */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Current role</label>
                    {editingProfile
                      ? <input className={inputCls} style={inputStyle} placeholder="e.g. SWE"
                          value={draft.current_role}
                          onChange={e => setDraft(d => ({ ...d, current_role: e.target.value }))} />
                      : <ReadValue value={meta.current_role ?? ''} placeholder="Not set" />}
                  </div>
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Target role</label>
                    {editingProfile
                      ? <input className={inputCls} style={inputStyle} placeholder="e.g. Senior SWE"
                          value={draft.target_role}
                          onChange={e => setDraft(d => ({ ...d, target_role: e.target.value }))} />
                      : <ReadValue value={meta.target_role ?? ''} placeholder="Not set" />}
                  </div>
                </div>

                {/* Years + Language */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Years of experience</label>
                    {editingProfile
                      ? <input type="number" min="0" max="50" className={inputCls} style={inputStyle} placeholder="e.g. 5"
                          value={draft.years_experience}
                          onChange={e => setDraft(d => ({ ...d, years_experience: e.target.value }))} />
                      : <ReadValue value={meta.years_experience ? `${meta.years_experience} yrs` : ''} placeholder="Not set" />}
                  </div>
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Interview language</label>
                    {editingProfile
                      ? <select className={inputCls + ' cursor-pointer'} style={{ ...inputStyle, appearance: 'none' as const }}
                          value={draft.interview_language}
                          onChange={e => setDraft(d => ({ ...d, interview_language: e.target.value }))}>
                          {LANGS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                      : <ReadValue value={meta.interview_language ?? 'English'} />}
                  </div>
                </div>

                {/* GitHub + LinkedIn */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>
                      <Github className="inline w-3 h-3 mr-1" />GitHub
                    </label>
                    {editingProfile
                      ? <input type="url" className={inputCls} style={inputStyle} placeholder="github.com/…"
                          value={draft.github_url}
                          onChange={e => setDraft(d => ({ ...d, github_url: e.target.value }))} />
                      : <ReadValue value={meta.github_url ?? ''} placeholder="Not set" />}
                  </div>
                  <div>
                    <label className={labelCls} style={{ color: PALETTE.textSecondary }}>
                      <Linkedin className="inline w-3 h-3 mr-1" />LinkedIn
                    </label>
                    {editingProfile
                      ? <input type="url" className={inputCls} style={inputStyle} placeholder="linkedin.com/in/…"
                          value={draft.linkedin_url}
                          onChange={e => setDraft(d => ({ ...d, linkedin_url: e.target.value }))} />
                      : <ReadValue value={meta.linkedin_url ?? ''} placeholder="Not set" />}
                  </div>
                </div>

                {/* Portfolio */}
                <div>
                  <label className={labelCls} style={{ color: PALETTE.textSecondary }}>
                    <Globe className="inline w-3 h-3 mr-1" />Portfolio / website
                  </label>
                  {editingProfile
                    ? <input type="url" className={inputCls} style={inputStyle} placeholder="https://yoursite.com"
                        value={draft.portfolio_url}
                        onChange={e => setDraft(d => ({ ...d, portfolio_url: e.target.value }))} />
                    : <ReadValue value={meta.portfolio_url ?? ''} placeholder="Not set" />}
                </div>
              </div>
            </section>

            {/* ── 3. CONNECTED ACCOUNTS ────────────── col 1, row 2 ── */}
            <section className="rounded-[14px] p-6" style={panel}>
              <SectionHeader icon={Link2} title="Connected accounts"
                subtitle="Auth providers currently linked to this account." />
              {providers.length === 0 ? (
                <p className="text-sm" style={{ color: PALETTE.textSecondary }}>No linked providers.</p>
              ) : (
                <div className="space-y-2">
                  {providers.map((provider: string) => (
                    <div key={provider} className="flex items-center gap-3 px-3 py-2.5 rounded-[8px]"
                      style={{ border: `1px solid ${PALETTE.border}`, backgroundColor: PALETTE.surface2 }}>
                      {provider === 'github' && <Github className="w-4 h-4 shrink-0" style={{ color: PALETTE.textPrimary }} />}
                      {provider === 'email'  && <Mail   className="w-4 h-4 shrink-0" style={{ color: PALETTE.textPrimary }} />}
                      {!['github', 'email'].includes(provider) && <Link2 className="w-4 h-4 shrink-0" style={{ color: PALETTE.textPrimary }} />}
                      <span className="text-sm capitalize" style={{ color: PALETTE.textPrimary }}>{provider}</span>
                      <span className="ml-auto text-[10px] text-emerald-400">connected</span>
                    </div>
                  ))}
                </div>
              )}
              <p className="text-xs mt-3" style={{ color: PALETTE.textSecondary }}>
                Linking additional providers requires re-authenticating.{' '}
                <PlaceholderBadge label="MULTI-PROVIDER LINK" />
              </p>
            </section>

            {/* ── 4. SECURITY ──────────────────────── col 2, row 2 ── */}
            <section className="rounded-[14px] p-6" style={panel}>
              <SectionHeader icon={Lock} title="Security" />

              <div className="space-y-1 mb-5">
                <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                  Signed in as <span style={{ color: PALETTE.textPrimary }}>{user?.email}</span>
                </p>
                {user?.last_sign_in_at && (
                  <p className="text-xs" style={{ color: PALETTE.textSecondary }}>
                    Last sign-in:{' '}
                    <span style={{ color: PALETTE.textPrimary }}>
                      {new Date(user.last_sign_in_at).toLocaleString()}
                    </span>
                  </p>
                )}
              </div>

              {/* Password — revealed on demand */}
              {providers.includes('email') && (
                <div className="mb-4">
                  {!showPwForm ? (
                    <button
                      type="button"
                      onClick={() => { setShowPwForm(true); setPwMsg(null); }}
                      className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-[8px] border transition-colors hover:bg-white/[0.03]"
                      style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                    >
                      <Lock className="w-3.5 h-3.5" />
                      Change password
                      <ChevronDown className="w-3.5 h-3.5" />
                    </button>
                  ) : (
                    <form onSubmit={handlePasswordChange} className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className={labelCls} style={{ color: PALETTE.textSecondary }}>New password</label>
                          <input type="password" className={inputCls} style={inputStyle}
                            placeholder="Min 6 characters" autoFocus
                            value={newPw} onChange={e => setNewPw(e.target.value)} />
                        </div>
                        <div>
                          <label className={labelCls} style={{ color: PALETTE.textSecondary }}>Confirm</label>
                          <input type="password" className={inputCls} style={inputStyle}
                            placeholder="Repeat"
                            value={confirmPw} onChange={e => setConfirmPw(e.target.value)} />
                        </div>
                      </div>
                      {pwMsg && <FieldMsg msg={pwMsg} />}
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => { setShowPwForm(false); setNewPw(''); setConfirmPw(''); setPwMsg(null); }}
                          className="px-3 py-1.5 rounded-[8px] text-xs border transition-colors hover:bg-white/[0.03]"
                          style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          disabled={pwSaving || !newPw}
                          className="px-4 py-1.5 rounded-[8px] text-xs font-medium transition-opacity hover:opacity-90 disabled:opacity-40"
                          style={{ backgroundColor: PALETTE.accent, color: '#07080a' }}
                        >
                          {pwSaving ? 'Updating…' : 'Update password'}
                        </button>
                      </div>
                    </form>
                  )}
                </div>
              )}

              <div className="pt-4" style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                <button
                  type="button"
                  onClick={signOut}
                  className="flex items-center gap-2 px-4 py-2 rounded-[8px] text-sm font-medium border transition-colors hover:bg-white/[0.03]"
                  style={{ borderColor: PALETTE.border, color: PALETTE.textSecondary }}
                >
                  <LogOut className="w-3.5 h-3.5" />
                  Sign out
                </button>
              </div>
            </section>

            {/* ── 5. DANGER ZONE ─────────── col 1 + col 2 (full) ── */}
            <section
              className="lg:col-span-2 rounded-[14px] p-6"
              style={{ ...panel, borderColor: 'rgba(239,68,68,0.15)' }}
            >
              <SectionHeader icon={Trash2} title="Danger zone" />
              {!deleteConfirm ? (
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-medium" style={{ color: PALETTE.textPrimary }}>Delete account</p>
                    <p className="text-xs mt-1" style={{ color: PALETTE.textSecondary }}>
                      Permanently deletes your account and all data. This action cannot be undone.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setDeleteConfirm(true)}
                    className="shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-[8px] text-sm font-medium border border-red-900/40 text-red-400 hover:bg-red-950/20 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Delete account
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm text-red-400">Are you sure? This is irreversible.</p>
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => setDeleteConfirm(false)}
                      className="px-4 py-2 rounded-[8px] text-sm font-medium transition-colors hover:bg-white/[0.03]"
                      style={{ border: `1px solid ${PALETTE.border}`, color: PALETTE.textSecondary }}
                    >
                      Cancel
                    </button>
                    <button
                      type="button" disabled
                      className="flex items-center gap-2 px-4 py-2 rounded-[8px] text-sm font-medium border border-red-900/40 text-red-400 opacity-40 cursor-not-allowed"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Confirm delete
                      <PlaceholderBadge label="NEEDS SERVER ENDPOINT" />
                    </button>
                  </div>
                </div>
              )}
            </section>

          </div>
        </form>
      </main>
    </div>
  );
}

export default function ProfilePage() {
  return (
    <AuthGuard>
      <ProfilePageInner />
    </AuthGuard>
  );
}
