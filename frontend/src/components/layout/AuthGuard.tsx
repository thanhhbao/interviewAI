'use client';

import React, { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, loading, openLoginModal } = useAuth();

  useEffect(() => {
    if (!loading && !user) {
      openLoginModal();
    }
  }, [user, loading, openLoginModal]);

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <span className="text-[13px] text-neutral-500">Loading…</span>
      </div>
    );
  }

  // Show content only when logged in; modal handles the unauthenticated state
  if (!user) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <span className="text-[13px] text-neutral-500">Sign in to continue.</span>
      </div>
    );
  }

  return <>{children}</>;
}
