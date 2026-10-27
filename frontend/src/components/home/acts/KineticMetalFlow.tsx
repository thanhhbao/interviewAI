'use client';

import React, { useEffect, useRef } from 'react';
import { useSafeReducedMotion } from '@/lib/motion';

// Native clone of the Framer "Kinetic Metal Flow" component's actual
// mechanism: a continuously deforming surface, not a static gradient. A
// raw WebGL1 fragment shader domain-warps several layered sine fields to
// produce a slow, heavy, hypnotic flowing-liquid-metal look, then maps the
// result through a neutral black-to-white ramp instead of chrome/silver
// (was dark-teal; the accent moved to monochrome, so this did too).
// Fully autonomous — driven by its own clock, never by scroll position.

const VERTEX_SHADER = `
  attribute vec2 a_position;
  varying vec2 v_uv;
  void main() {
    v_uv = (a_position + 1.0) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  precision highp float;
  uniform vec2 u_resolution;
  uniform float u_time;
  varying vec2 v_uv;

  float wave(vec2 uv, float freq, float speed, float phase) {
    return sin(uv.x * freq + u_time * speed + phase) * cos(uv.y * freq * 0.6 - u_time * speed * 0.7 + phase);
  }

  void main() {
    vec2 uv = v_uv;
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.15;

    vec2 p = uv * 2.6;
    p.x += sin(p.y * 1.3 + t) * 0.7;
    p.y += cos(p.x * 1.1 - t * 0.8) * 0.7;

    float n = 0.0;
    n += wave(p, 1.5, 0.22, 0.0) * 0.5;
    n += wave(p * 1.7, 2.3, -0.16, 1.7) * 0.3;
    n += wave(p * 2.9, 3.1, 0.11, 3.1) * 0.2;
    n = n * 0.5 + 0.5;

    vec3 dark = vec3(0.03, 0.03, 0.035);
    vec3 deep = vec3(0.06, 0.06, 0.065);
    vec3 shadow = vec3(0.14, 0.14, 0.145);
    vec3 highlight = vec3(0.32, 0.32, 0.33);
    vec3 accent = vec3(0.65, 0.65, 0.66);
    vec3 bright = vec3(1.0, 1.0, 1.0);

    vec3 col = mix(dark, deep, smoothstep(0.0, 0.42, n));
    col = mix(col, shadow, smoothstep(0.35, 0.62, n));
    col = mix(col, highlight, smoothstep(0.6, 0.82, n));

    float sparkle = smoothstep(0.86, 0.97, n);
    col = mix(col, accent, sparkle * 0.55);
    float rare = smoothstep(0.965, 1.0, n);
    col = mix(col, bright, rare);

    gl_FragColor = vec4(col, 1.0);
  }
`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || 'shader compile error');
  }
  return shader;
}

export function KineticMetalFlow({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prefersReducedMotion = useSafeReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl', { alpha: false, antialias: true });
    if (!gl) return;

    const program = gl.createProgram()!;
    gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || 'program link error');
    }
    gl.useProgram(program);

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const resLoc = gl.getUniformLocation(program, 'u_resolution');
    const timeLoc = gl.getUniformLocation(program, 'u_time');

    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    let width = 0;
    let height = 0;

    function resize(rect: { width: number; height: number }) {
      const c = canvasRef.current;
      if (!c || !gl) return;
      width = Math.max(1, Math.round(rect.width * dpr));
      height = Math.max(1, Math.round(rect.height * dpr));
      c.width = width;
      c.height = height;
      gl.viewport(0, 0, width, height);
    }
    resize(canvas.getBoundingClientRect());

    const resizeObserver = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) resize(box);
    });
    resizeObserver.observe(canvas);

    function renderFrame(t: number) {
      if (!gl) return;
      gl.uniform2f(resLoc, width, height);
      gl.uniform1f(timeLoc, t);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    let raf = 0;
    let running = false;
    const start = performance.now();

    function frame(now: number) {
      renderFrame((now - start) / 1000);
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
      renderFrame(0);
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
  }, [prefersReducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.55 }}
    />
  );
}
