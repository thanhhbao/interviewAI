'use client';

import React from 'react';
import { FileText, Upload } from 'lucide-react';
import { PALETTE } from '@/lib/motion';

interface FileRowProps {
  fileName: string;
  status?: string;
  onReplace: () => void;
  replaceLabel?: string;
}

// Compact uploaded-file identity row (REF-C inspired) — replaces the large
// upload target once parsing succeeds. Deliberately just one row, not the
// full file-collection/gallery pattern REF-C demos.
export function FileRow({ fileName, status = 'Parsed', onReplace, replaceLabel = 'Replace' }: FileRowProps) {
  return (
    <div
      className="flex items-center gap-3 px-3.5 py-3 rounded-[10px]"
      style={{ backgroundColor: PALETTE.surface2, border: `1px solid ${PALETTE.border}` }}
    >
      <div
        className="w-8 h-8 rounded-[8px] flex items-center justify-center shrink-0"
        style={{ backgroundColor: PALETTE.surfaceFaint, color: PALETTE.accent }}
      >
        <FileText className="w-4 h-4" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium truncate" style={{ color: PALETTE.textPrimary }}>
          {fileName}
        </p>
        <p className="text-[11px] font-mono" style={{ color: PALETTE.textSecondary }}>
          {status}
        </p>
      </div>
      <button
        onClick={onReplace}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-[8px] text-[11px] font-medium shrink-0 transition-colors hover:bg-white/[0.04]"
        style={{ color: PALETTE.textSecondary, border: `1px solid ${PALETTE.border}` }}
      >
        <Upload className="w-3 h-3" />
        <span>{replaceLabel}</span>
      </button>
    </div>
  );
}
