'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useSafeReducedMotion } from '@/lib/motion';

// Terminal content lines. Each entry has:
// - text: the string to render
// - style: 'prompt' | 'indent' | 'check' | 'label' | 'question' | 'status' | 'blank'
// - delay: ms after the previous line before this line appears
type LineStyle = 'prompt' | 'indent' | 'check' | 'status' | 'question' | 'blank' | 'listening';

interface TerminalLine {
  text: string;
  style: LineStyle;
  delay: number; // ms after previous line
}

const LINES: TerminalLine[] = [
  { text: '> INITIALIZING INTERVIEW SESSION', style: 'prompt', delay: 300 },
  { text: '', style: 'blank', delay: 80 },
  { text: '  Role: Senior Software Engineer', style: 'indent', delay: 120 },
  { text: '  Company: Nova Technologies', style: 'indent', delay: 80 },
  { text: '  Match score: 94%', style: 'indent', delay: 80 },
  { text: '', style: 'blank', delay: 200 },
  { text: '> ANALYZING RESUME', style: 'prompt', delay: 160 },
  { text: '  ✓ Backend systems', style: 'check', delay: 120 },
  { text: '  ✓ Distributed architecture', style: 'check', delay: 100 },
  { text: '  ✓ Python / Go / AWS', style: 'check', delay: 100 },
  { text: '', style: 'blank', delay: 200 },
  { text: '> PREPARING INTERVIEW', style: 'prompt', delay: 160 },
  { text: '  Behavioral ........ ready', style: 'status', delay: 120 },
  { text: '  System Design ..... ready', style: 'status', delay: 100 },
  { text: '  Coding ............ ready', style: 'status', delay: 100 },
  { text: '', style: 'blank', delay: 300 },
  { text: '> INTERVIEWER ONLINE', style: 'prompt', delay: 160 },
  { text: '', style: 'blank', delay: 200 },
  { text: '  "Tell me about a system', style: 'question', delay: 120 },
  { text: '   you designed to handle', style: 'question', delay: 80 },
  { text: '   high traffic."', style: 'question', delay: 60 },
  { text: '', style: 'blank', delay: 600 },
  { text: 'LISTENING...', style: 'listening', delay: 120 },
];

// Pause before looping (ms)
const LOOP_PAUSE = 3000;

// Color tokens for terminal content
const COLOR_PROMPT = '#E7E8E9';     // bright white for > prompts
const COLOR_INDENT = '#9A9893';     // muted for detail lines
const COLOR_CHECK = '#6EBF91';      // subtle green for check marks
const COLOR_STATUS_LABEL = '#9A9893';
const COLOR_STATUS_VALUE = '#6EBF91';
const COLOR_QUESTION = '#C8C6C0';  // warm mid-white for speech
const COLOR_LISTENING = '#6EBF91'; // green accent

interface GlitchSlice {
  y: number;   // % from top of screen
  h: number;   // % height
  dx: number;  // px shift
  blur: number; // px blur
}

function randomGlitchSlices(): GlitchSlice[] {
  const count = Math.floor(Math.random() * 2) + 3; // 3-4 slices
  return Array.from({ length: count }, (_, i) => ({
    y: 5 + (i * 80 / count) + Math.random() * 10,
    h: 3 + Math.random() * 7,
    dx: (Math.random() - 0.5) * 8, // ±4px
    blur: Math.random() < 0.4 ? 0.5 + Math.random() * 1 : 0,
  }));
}

function LineRenderer({ line, glitchDx = 0 }: { line: TerminalLine; glitchDx?: number }) {
  if (line.style === 'blank') return <div className="h-[0.6em]" />;

  if (line.style === 'status') {
    // Split on the last 'ready'
    const parts = line.text.match(/^(.*?)(ready)$/);
    if (parts) {
      return (
        <div
          className="font-mono text-[clamp(7px,1.1vw,11px)] leading-[1.7]"
          style={{ color: COLOR_STATUS_LABEL, transform: `translateX(${glitchDx}px)` }}
        >
          {parts[1]}
          <span style={{ color: COLOR_STATUS_VALUE }}>{parts[2]}</span>
        </div>
      );
    }
  }

  if (line.style === 'check') {
    const checkMark = line.text.match(/✓/);
    if (checkMark) {
      const [before, after] = line.text.split('✓');
      return (
        <div
          className="font-mono text-[clamp(7px,1.1vw,11px)] leading-[1.7]"
          style={{ transform: `translateX(${glitchDx}px)` }}
        >
          <span style={{ color: COLOR_STATUS_LABEL }}>{before}</span>
          <span style={{ color: COLOR_CHECK }}>✓</span>
          <span style={{ color: COLOR_INDENT }}>{after}</span>
        </div>
      );
    }
  }

  const colorMap: Record<LineStyle, string> = {
    prompt: COLOR_PROMPT,
    indent: COLOR_INDENT,
    check: COLOR_CHECK,
    status: COLOR_STATUS_LABEL,
    question: COLOR_QUESTION,
    blank: 'transparent',
    listening: COLOR_LISTENING,
  };

  return (
    <div
      className="font-mono text-[clamp(7px,1.1vw,11px)] leading-[1.7]"
      style={{
        color: colorMap[line.style],
        fontWeight: line.style === 'prompt' ? 600 : 400,
        transform: `translateX(${glitchDx}px)`,
        letterSpacing: line.style === 'listening' ? '0.12em' : undefined,
      }}
    >
      {line.text}
    </div>
  );
}

// Screen bounds as % of the 500×500 computer-removebg-preview.png image.
// Measured from pixel analysis: screen top≈19%, left≈14% (conservative for
// curved CRT bezel), right≈88% → width≈74%, bottom≈60% → height≈41%.
const SCREEN = { top: 19, left: 14, width: 74, height: 41 } as const;

export const TerminalScreen: React.FC = () => {
  const prefersReducedMotion = useSafeReducedMotion();
  const [visibleCount, setVisibleCount] = useState(0);
  const [showCursor, setShowCursor] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  const [glitchSlices, setGlitchSlices] = useState<GlitchSlice[]>([]);
  const [flickerOpacity, setFlickerOpacity] = useState(1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const glitchTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const flickerTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Progressive line reveal
  const scheduleNextLine = useCallback((index: number) => {
    if (index >= LINES.length) {
      // All lines shown — pause then loop
      timerRef.current = setTimeout(() => {
        setVisibleCount(0);
        timerRef.current = setTimeout(() => scheduleNextLine(0), 200);
      }, LOOP_PAUSE);
      return;
    }
    const delay = LINES[index].delay;
    timerRef.current = setTimeout(() => {
      setVisibleCount(index + 1);
      scheduleNextLine(index + 1);
    }, delay);
  }, []);

  useEffect(() => {
    scheduleNextLine(0);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [scheduleNextLine]);

  // Cursor blink
  useEffect(() => {
    if (prefersReducedMotion) return;
    const id = setInterval(() => setShowCursor(v => !v), 530);
    return () => clearInterval(id);
  }, [prefersReducedMotion]);

  // Base subtle flicker (very rare, low-frequency)
  useEffect(() => {
    if (prefersReducedMotion) return;
    const scheduleFlicker = () => {
      const delay = 4000 + Math.random() * 8000; // every 4-12s
      flickerTimerRef.current = setTimeout(() => {
        setFlickerOpacity(0.88);
        setTimeout(() => {
          setFlickerOpacity(1);
          scheduleFlicker();
        }, 60);
      }, delay);
    };
    scheduleFlicker();
    return () => {
      if (flickerTimerRef.current) clearTimeout(flickerTimerRef.current as unknown as number);
    };
  }, [prefersReducedMotion]);

  // Glitch on hover
  useEffect(() => {
    if (!isHovered || prefersReducedMotion) {
      setGlitchSlices([]);
      if (glitchTimerRef.current) clearInterval(glitchTimerRef.current);
      return;
    }
    const update = () => {
      if (Math.random() < 0.65) {
        setGlitchSlices(randomGlitchSlices());
        setTimeout(() => setGlitchSlices([]), 60 + Math.random() * 80);
      }
    };
    glitchTimerRef.current = setInterval(update, 180 + Math.random() * 140);
    return () => {
      if (glitchTimerRef.current) clearInterval(glitchTimerRef.current);
      setGlitchSlices([]);
    };
  }, [isHovered, prefersReducedMotion]);

  return (
    <div
      aria-hidden="true"
      style={{
        position: 'absolute',
        top: `${SCREEN.top}%`,
        left: `${SCREEN.left}%`,
        width: `${SCREEN.width}%`,
        height: `${SCREEN.height}%`,
        overflow: 'hidden',
        borderRadius: '2px',
        cursor: 'default',
        opacity: flickerOpacity,
        transition: 'opacity 40ms linear',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* SVG filter definitions for glitch/blur effects */}
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <defs>
          <filter id="crt-noise">
            <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" result="noise" />
            <feColorMatrix type="saturate" values="0" in="noise" result="grayNoise" />
            <feBlend in="SourceGraphic" in2="grayNoise" mode="overlay" result="blend" />
            <feComposite in="blend" in2="SourceGraphic" operator="in" />
          </filter>
          <filter id="rgb-split">
            <feOffset in="SourceGraphic" dx="1" dy="0" result="r" />
            <feOffset in="SourceGraphic" dx="-1" dy="0" result="b" />
            <feBlend in="r" in2="b" mode="screen" />
          </filter>
        </defs>
      </svg>

      {/* Scanlines overlay — very subtle */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: 'repeating-linear-gradient(to bottom, transparent 0px, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      />

      {/* Screen vignette */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.45) 100%)',
          pointerEvents: 'none',
          zIndex: 6,
        }}
      />

      {/* Phosphor glow layer — very faint green tint on hovered state */}
      <motion.div
        animate={{ opacity: isHovered ? 0.06 : 0.02 }}
        transition={{ duration: 0.4 }}
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(ellipse at 50% 40%, rgba(110,191,145,0.6) 0%, transparent 70%)',
          pointerEvents: 'none',
          zIndex: 4,
        }}
      />

      {/* Terminal content */}
      <div
        ref={contentRef}
        style={{
          position: 'absolute',
          inset: 0,
          padding: '8% 9%',
          zIndex: 3,
          overflow: 'hidden',
        }}
      >
        {LINES.slice(0, visibleCount).map((line, i) => (
          <LineRenderer
            key={i}
            line={line}
            glitchDx={
              glitchSlices.length > 0
                ? (() => {
                    // Find if this line's approximate position matches a glitch slice
                    const linePercent = (i / LINES.length) * 100;
                    const hit = glitchSlices.find(s => linePercent >= s.y && linePercent < s.y + s.h);
                    return hit ? hit.dx : 0;
                  })()
                : 0
            }
          />
        ))}

        {/* Blinking cursor */}
        {visibleCount > 0 && (
          <span
            className="font-mono text-[clamp(7px,1.1vw,11px)]"
            style={{
              color: COLOR_PROMPT,
              opacity: showCursor ? 1 : 0,
              transition: 'opacity 60ms',
              display: 'inline-block',
              width: '0.55em',
              height: '1.1em',
              backgroundColor: COLOR_PROMPT,
              verticalAlign: 'middle',
              marginLeft: '1px',
            }}
          />
        )}
      </div>

      {/* Glitch tearing overlay — plain conditional render, no AnimatePresence.
          The effect lasts 60–140ms; an enter/exit animation on top of that
          is imperceptible and caused key-collision crashes when the same
          numeric keys (0,1,2) re-entered while prior exits were still pending. */}
      {!prefersReducedMotion && glitchSlices.map((slice, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: `${slice.y}%`,
            height: `${slice.h}%`,
            zIndex: 7,
            transform: `translateX(${slice.dx}px)`,
            filter: slice.blur > 0 ? `blur(${slice.blur}px)` : undefined,
            overflow: 'hidden',
            pointerEvents: 'none',
          }}
        >
          <div
            style={{
              position: 'absolute',
              inset: 0,
              padding: '8% 9%',
              top: `-${slice.y}%`,
              height: `${100 / (slice.h / 100)}%`,
              background: 'rgba(4,6,6,0.92)',
            }}
          >
            {LINES.slice(0, visibleCount).map((line, j) => (
              <LineRenderer key={j} line={line} />
            ))}
          </div>
        </div>
      ))}

      {/* RGB separation — plain motion.div, opacity driven by isHovered.
          No AnimatePresence: the element stays mounted and just fades in/out,
          which avoids any unmount/remount race that could corrupt the DOM. */}
      {!prefersReducedMotion && (
        <motion.div
          animate={{ opacity: isHovered ? 1 : 0 }}
          transition={{ duration: isHovered ? 0.2 : 0.3 }}
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 8,
            pointerEvents: 'none',
            mixBlendMode: 'screen',
          }}
        >
          {/* R channel +1px */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              padding: '8% 9%',
              transform: 'translateX(1px)',
              opacity: 0.08,
            }}
          >
            {LINES.slice(0, visibleCount).map((line, i) => (
              <div
                key={i}
                className="font-mono text-[clamp(7px,1.1vw,11px)] leading-[1.7]"
                style={{ color: '#ff4444' }}
              >
                {line.text}
              </div>
            ))}
          </div>
          {/* B channel -1px */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              padding: '8% 9%',
              transform: 'translateX(-1px)',
              opacity: 0.06,
            }}
          >
            {LINES.slice(0, visibleCount).map((line, i) => (
              <div
                key={i}
                className="font-mono text-[clamp(7px,1.1vw,11px)] leading-[1.7]"
                style={{ color: '#4488ff' }}
              >
                {line.text}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};
