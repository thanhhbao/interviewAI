'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';

interface AuthLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
}

/**
 * Drop-in replacement for <Link> on protected routes.
 * - Logged in  → navigates to href
 * - Logged out → opens LoginModal
 */
export function AuthLink({ href, children, className, style, onClick }: AuthLinkProps) {
  const { user, openLoginModal } = useAuth();
  const router = useRouter();

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    onClick?.();
    if (user) {
      router.push(href);
    } else {
      openLoginModal();
    }
  }

  return (
    <a href={href} onClick={handleClick} className={className} style={style}>
      {children}
    </a>
  );
}
