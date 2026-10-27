'use client';

import React, { Suspense, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import { Github } from 'lucide-react';

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get('next') ?? '/';
  const callbackError = params.get('error');

  const [tab, setTab] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(
    callbackError ? 'Authentication failed. Please try again.' : null
  );
  const [message, setMessage] = useState<string | null>(null);

  const supabase = createClient();

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (user) router.replace(next);
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleEmailAuth(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setLoading(true);
    try {
      if (tab === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        router.replace(next);
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
        redirectTo: `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`,
      },
    });
    if (error) setError(error.message);
  }

  return (
    <div className="w-full max-w-sm">
      {/* Logo */}
      <div className="mb-8 text-center">
        <span className="text-[13px] font-mono tracking-widest text-neutral-500 uppercase">
          InterviewAI
        </span>
        <h1 className="mt-2 text-2xl font-semibold text-neutral-100">
          {tab === 'login' ? 'Sign in' : 'Create account'}
        </h1>
      </div>

      {/* Tab toggle */}
      <div className="flex rounded-lg bg-neutral-900 p-1 mb-6">
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

      {/* GitHub OAuth */}
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

      {/* Email/Password form */}
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
  );
}

export default function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4 bg-[#09090B]">
      <Suspense fallback={<span className="text-neutral-500 text-sm">Loading…</span>}>
        <LoginForm />
      </Suspense>
    </div>
  );
}
