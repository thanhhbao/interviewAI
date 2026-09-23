'use client';

import { useEffect, useRef } from 'react';
import { useSafeReducedMotion } from '@/lib/motion';

/**
 * Shared driver for the act background canvases. Every effect built on this
 * hook is AUTONOMOUS — it runs on its own clock via requestAnimationFrame
 * and is never a function of scroll position. The only thing gating it is
 * visibility: a ResizeObserver keeps the backing store correct as the
 * container's actual size changes (not just window resize), and an
 * IntersectionObserver pauses the rAF loop while the act is off-screen so
 * only one act's animation is ever actually costing CPU/GPU at a time.
 * Under prefers-reduced-motion it draws exactly one static frame and stops.
 */
export function useCanvasLoop(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void,
  onResize?: (w: number, h: number) => void
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prefersReducedMotion = useSafeReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize(rect: { width: number; height: number }) {
      const canvas2 = canvasRef.current;
      if (!canvas2) return;
      width = rect.width;
      height = rect.height;
      canvas2.width = Math.round(width * dpr);
      canvas2.height = Math.round(height * dpr);
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      onResize?.(width, height);
    }
    resize(canvas.getBoundingClientRect());

    const resizeObserver = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) resize(box);
    });
    resizeObserver.observe(canvas);

    let raf = 0;
    let running = false;
    const start = performance.now();

    function frame(now: number) {
      draw(ctx!, width, height, (now - start) / 1000);
      if (running) raf = requestAnimationFrame(frame);
    }

    function play() {
      if (running) return;
      running = true;
      raf = requestAnimationFrame(frame);
    }
    function pause() {
      running = false;
      cancelAnimationFrame(raf);
    }

    if (prefersReducedMotion) {
      draw(ctx, width, height, 0);
      return () => resizeObserver.disconnect();
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) play();
        else pause();
      },
      { threshold: 0.01 }
    );
    observer.observe(canvas);

    return () => {
      observer.disconnect();
      resizeObserver.disconnect();
      pause();
    };
  }, [draw, onResize, prefersReducedMotion]);

  return canvasRef;
}
