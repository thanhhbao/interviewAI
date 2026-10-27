'use client';

import React from 'react';
import { X } from 'lucide-react';
import { PALETTE } from '@/lib/motion';

interface ModalShellProps {
  title: string;
  eyebrow?: string;
  onClose: () => void;
  headerExtra?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  maxWidthClassName?: string;
}

// One shared dark modal shell for Edit Profile / Job Editor / Resume
// Optimizer, so the three don't each invent their own surface/border/radius
// language. Business logic for each modal stays where it already lives —
// this only owns the backdrop, panel chrome, header and footer slots.
export function ModalShell({
  title,
  eyebrow,
  onClose,
  headerExtra,
  footer,
  children,
  maxWidthClassName = 'max-w-xl',
}: ModalShellProps) {
  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div
        className={`w-full ${maxWidthClassName} max-h-[85vh] rounded-[16px] flex flex-col overflow-hidden shadow-[0_40px_100px_-30px_rgba(0,0,0,0.6)]`}
        style={{ backgroundColor: PALETTE.surface, border: `1px solid ${PALETTE.border}` }}
      >
        <div
          className="px-6 py-4 flex items-center justify-between shrink-0"
          style={{ borderBottom: `1px solid ${PALETTE.borderTeal}` }}
        >
          <div>
            {eyebrow && (
              <p className="text-[11px] font-mono tracking-[0.08em] mb-0.5" style={{ color: PALETTE.accent }}>
                {eyebrow}
              </p>
            )}
            <h3 className="text-sm font-semibold" style={{ color: PALETTE.textPrimary }}>
              {title}
            </h3>
          </div>
          <div className="flex items-center gap-3">
            {headerExtra}
            <button
              onClick={onClose}
              className="p-1.5 rounded-[8px] transition-colors hover:bg-white/[0.06]"
              style={{ color: PALETTE.textSecondary }}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">{children}</div>

        {footer && (
          <div
            className="px-6 py-4 flex items-center justify-end gap-2 shrink-0"
            style={{ borderTop: `1px solid ${PALETTE.borderTeal}` }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
