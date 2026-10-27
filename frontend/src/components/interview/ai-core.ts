'use client';

// Ported from the official SmoothUI registry
// (github.com/educlopez/smoothui — packages/smoothui/components/ai-core),
// trimmed to what SiriOrb.tsx actually uses: the state->motion preset table
// and a simulated amplitude generator (no real microphone/AnalyserNode
// wiring — the orb represents the AI's own voice, not the candidate's, and
// a plausible simulated envelope is enough to look alive). Import path
// `motion/react` in the original became `framer-motion` here, same API,
// already a project dependency.
import { type MotionValue, useAnimationFrame, useMotionValue } from 'framer-motion';
import { useEffect } from 'react';

export type AIState = 'idle' | 'listening' | 'thinking' | 'streaming' | 'done' | 'error';

export type AIStateMotif = 'breathe' | 'receive' | 'scan' | 'pulse' | 'ping' | 'fault';

export type AIStateAccent = 'success' | 'danger' | null;

export type AIStateMotion = {
  accent: AIStateAccent;
  glow: number;
  hueRotate: number;
  intensity: number;
  motif: AIStateMotif;
  pulseSeconds: number;
  turbulence: number;
  tumble: number;
  reactivity: number;
  saturation: number;
  scale: number;
  speed: number;
};

export const AI_STATE_MOTION: Record<AIState, AIStateMotion> = {
  done: {
    accent: 'success',
    glow: 0.7,
    hueRotate: 0,
    intensity: 0.4,
    motif: 'ping',
    pulseSeconds: 0.65,
    reactivity: 0,
    saturation: 1,
    scale: 1.1,
    speed: 0.8,
    tumble: 0.02,
    turbulence: 0.08,
  },
  error: {
    accent: 'danger',
    glow: 0.25,
    hueRotate: 0,
    intensity: 0.5,
    motif: 'fault',
    pulseSeconds: 0.9,
    reactivity: 0,
    saturation: 0.3,
    scale: 0.96,
    speed: 1,
    tumble: 0,
    turbulence: 0.55,
  },
  idle: {
    accent: null,
    glow: 0.15,
    hueRotate: 0,
    intensity: 0.3,
    motif: 'breathe',
    pulseSeconds: 4.5,
    reactivity: 0,
    saturation: 0.75,
    scale: 0.94,
    speed: 0.6,
    tumble: 0.012,
    turbulence: 0.14,
  },
  listening: {
    accent: null,
    glow: 0.6,
    hueRotate: 0,
    intensity: 0.75,
    motif: 'receive',
    pulseSeconds: 1.6,
    reactivity: 1,
    saturation: 1.05,
    scale: 1.06,
    speed: 1,
    tumble: 0.03,
    turbulence: 0.42,
  },
  streaming: {
    accent: null,
    glow: 0.45,
    hueRotate: -10,
    intensity: 0.6,
    motif: 'pulse',
    pulseSeconds: 1.25,
    reactivity: 0.6,
    saturation: 1,
    scale: 1.02,
    speed: 1.4,
    tumble: 0.06,
    turbulence: 0.5,
  },
  thinking: {
    accent: null,
    glow: 0.35,
    hueRotate: 18,
    intensity: 1,
    motif: 'scan',
    pulseSeconds: 1.1,
    reactivity: 0.15,
    saturation: 1,
    scale: 1,
    speed: 2.4,
    tumble: 0.14,
    turbulence: 0.95,
  },
};

export const getAIStateMotion = (state: AIState | undefined): AIStateMotion =>
  AI_STATE_MOTION[state ?? 'idle'] ?? AI_STATE_MOTION.idle;

export type AIAmplitude = number | MotionValue<number> | undefined;

const isMotionValue = (value: AIAmplitude): value is MotionValue<number> =>
  typeof value === 'object' && value !== null && 'get' in value;

export const useAmplitudeValue = (amplitude: AIAmplitude): MotionValue<number> => {
  const fallback = useMotionValue(0);
  const numeric = typeof amplitude === 'number' ? amplitude : null;

  useEffect(() => {
    if (numeric !== null) {
      fallback.set(numeric);
    }
  }, [numeric, fallback]);

  return isMotionValue(amplitude) ? amplitude : fallback;
};

/**
 * Amplitude generator for the orb — no microphone involved. Produces a
 * plausible speech-like envelope whose energy follows the current
 * {@link AIState}, so the orb looks reactive while the AI is
 * speaking/thinking without needing to tap real TTS audio output.
 */
export const useSimulatedAmplitude = (state: AIState = 'idle'): MotionValue<number> => {
  const amplitude = useMotionValue(0);
  const motion = getAIStateMotion(state);

  useAnimationFrame((time) => {
    const t = time / 1000;
    const envelope =
      0.5 +
      0.3 * Math.sin(t * 2.1 * motion.speed) +
      0.14 * Math.sin(t * 5.3 * motion.speed + 1.7) +
      0.06 * Math.sin(t * 11.7 * motion.speed + 0.4);

    amplitude.set(Math.min(1, Math.max(0, envelope * motion.intensity)));
  });

  return amplitude;
};
