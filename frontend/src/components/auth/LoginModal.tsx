'use client';

import React, { useEffect, useState } from 'react';
import { createClient } from '@/lib/supabase/client';
import { useAuth } from '@/context/AuthContext';
import { Github, X } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

export function LoginModal() {
  const { loginModalOpen, closeLoginModal } = useAuth();
  const [tab, setTab] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const supabase = createClient();

  // Reset form when modal opens
  useEffect(() => {
    if (loginModalOpen) {
      setEmail('');
      setPassword('');
      setError(null);
      setMessage(null);
      setTab('login');
    }
  }, [loginModalOpen]);

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') closeLoginModal();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [closeLoginModal]);

  async function handleEmailAuth(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setLoading(true);
    try {
      if (tab === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // AuthContext onAuthStateChange closes the modal automatically
      } else {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
        });
        if (error) throw error;
        setMessage('Check your email to confirm your account.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  async function handleGitHub() {
    setError(null);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });
    if (error) setError(error.message);
  }

  return (
    <AnimatePresence>
      {loginModalOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-[90] bg-black/60 backdrop-blur-sm"
            onClick={closeLoginModal}
          />

          {/* Modal */}
          <motion.div
            key="modal"
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-0 z-[91] flex items-center justify-center px-4 pointer-events-none"
          >
            <div
              className="w-full max-w-sm rounded-2xl border border-neutral-800 bg-[#0d0d0f] p-8 shadow-2xl pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-[11px] font-mono tracking-widest text-neutral-500 uppercase mb-0.5">
                    InterviewAI
                  </p>
                  <h2 className="text-xl font-semibold text-neutral-100">
                    {tab === 'login' ? 'Sign in' : 'Create account'}
                  </h2>
                </div>
                <button
                  onClick={closeLoginModal}
                  className="p-1.5 rounded-lg text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Tab toggle */}
              <div className="flex rounded-lg bg-neutral-900 p-1 mb-5">
                {(['login', 'signup'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => { setTab(t); setError(null); setMessage(null); }}
                    className={`flex-1 py-1.5 text-[13px] font-medium rounded-md transition-colors ${
                      tab === t
                        ? 'bg-neutral-800 text-neutral-100'
                        : 'text-neutral-500 hover:text-neutral-300'
                    }`}
                  >
                    {t === 'login' ? 'Sign in' : 'Sign up'}
                  </button>
                ))}
              </div>

              {/* GitHub */}
              <button
                onClick={handleGitHub}
                className="w-full flex items-center justify-center gap-2.5 py-2.5 rounded-lg border border-neutral-800 bg-neutral-900 text-neutral-200 text-sm font-medium hover:bg-neutral-800 transition-colors mb-4"
              >
                <Github className="w-4 h-4" />
                Continue with GitHub
              </button>

              {/* Divider */}
              <div className="flex items-center gap-3 mb-4">
                <div className="flex-1 h-px bg-neutral-800" />
                <span className="text-[12px] text-neutral-600">or</span>
                <div className="flex-1 h-px bg-neutral-800" />
              </div>

              {/* Email form */}
              <form onSubmit={handleEmailAuth} className="flex flex-col gap-3">
                <input
                  type="email"
                  placeholder="Email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-lg border border-neutral-800 bg-neutral-900 text-neutral-100 text-sm placeholder-neutral-600 focus:outline-none focus:border-neutral-600 transition-colors"
                />
                <input
                  type="password"
                  placeholder="Password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-lg border border-neutral-800 bg-neutral-900 text-neutral-100 text-sm placeholder-neutral-600 focus:outline-none focus:border-neutral-600 transition-colors"
                />

                {error && (
                  <p className="text-[13px] text-red-400 bg-red-950/40 border border-red-900/50 rounded-lg px-3 py-2">
                    {error}
                  </p>
                )}
                {message && (
                  <p className="text-[13px] text-emerald-400 bg-emerald-950/40 border border-emerald-900/50 rounded-lg px-3 py-2">
                    {message}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-2.5 rounded-lg bg-neutral-100 text-neutral-900 text-sm font-semibold hover:bg-white transition-colors disabled:opacity-50"
                >
                  {loading ? 'Loading…' : tab === 'login' ? 'Sign in' : 'Create account'}
                </button>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
