'use client';

import React, { useEffect, useRef, useState } from 'react';

const VERTEX_SHADER = `
  attribute vec2 a_position;
  varying vec2 v_uv;

  void main() {
    v_uv = (a_position + 1.0) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// Calibrated per explicit direction against a second reference screenshot: the
// scan is a single THIN line that projects as a RECTANGLE, hugging just
// inside the photo's naturally dark border regions (ceiling above the lamp,
// the blurred foreground silhouettes left/right, the floor below) — not a
// wide horizontal wash. It starts at the top (where the dark ceiling band
// is), then reaches the two side margins and the floor, tracing the full
// rectangle. Modelled as two mirrored 3-segment paths that both start at
// top-centre, run out along the top edge to a corner, down that side, and
// in along the bottom edge to bottom-centre — so the whole frame closes.
const FRAGMENT_SHADER = `
  precision highp float;

  uniform sampler2D u_color;
  uniform sampler2D u_depth;
  uniform vec2 u_pointer;
  uniform float u_strength;
  uniform float u_time;
  uniform vec2 u_resolution;
  uniform vec2 u_image_resolution;
  varying vec2 v_uv;

  float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
  }

  float noise1(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash1(i), hash1(i + 1.0), u);
  }

  // Distance + normalized arc-length (0 at uvA, 1 at uvD) to the 3-segment
  // polyline uvA -> uvB -> uvC -> uvD, with a small depth-driven wobble so
  // the thin line still reads as flowing through the room rather than a
  // perfectly ruled vector rectangle.
  vec2 rectField(vec2 p, vec2 res, vec2 uvA, vec2 uvB, vec2 uvC, vec2 uvD, float seed) {
    vec2 pA = uvA * res;
    vec2 pB = uvB * res;
    vec2 pC = uvC * res;
    vec2 pD = uvD * res;

    vec2 dir1 = pB - pA;
    float len1 = length(dir1);
    vec2 dirN1 = dir1 / len1;
    vec2 perp1 = vec2(-dirN1.y, dirN1.x);
    float t1 = clamp(dot(p - pA, dirN1) / len1, 0.0, 1.0);
    vec2 base1 = pA + dirN1 * (t1 * len1);

    vec2 dir2 = pC - pB;
    float len2 = length(dir2);
    vec2 dirN2 = dir2 / len2;
    vec2 perp2 = vec2(-dirN2.y, dirN2.x);
    float t2 = clamp(dot(p - pB, dirN2) / len2, 0.0, 1.0);
    vec2 base2 = pB + dirN2 * (t2 * len2);

    vec2 dir3 = pD - pC;
    float len3 = length(dir3);
    vec2 dirN3 = dir3 / len3;
    vec2 perp3 = vec2(-dirN3.y, dirN3.x);
    float t3 = clamp(dot(p - pC, dirN3) / len3, 0.0, 1.0);
    vec2 base3 = pC + dirN3 * (t3 * len3);

    float amp1 = 0.8 + 2.6 * texture2D(u_depth, clamp(base1 / res, 0.0, 1.0)).r;
    float w1 = (noise1(t1 * len1 * 0.04 + seed) - 0.5) * 2.0 * amp1;
    float amp2 = 0.8 + 2.6 * texture2D(u_depth, clamp(base2 / res, 0.0, 1.0)).r;
    float w2 = (noise1(t2 * len2 * 0.04 + seed) - 0.5) * 2.0 * amp2;
    float amp3 = 0.8 + 2.6 * texture2D(u_depth, clamp(base3 / res, 0.0, 1.0)).r;
    float w3 = (noise1(t3 * len3 * 0.04 + seed) - 0.5) * 2.0 * amp3;

    vec2 c1 = base1 + perp1 * w1;
    vec2 c2 = base2 + perp2 * w2;
    vec2 c3 = base3 + perp3 * w3;

    float d1 = length(p - c1);
    float d2 = length(p - c2);
    float d3 = length(p - c3);
    float totalLen = len1 + len2 + len3;

    if (d1 <= d2 && d1 <= d3) {
      return vec2(d1, (t1 * len1) / totalLen);
    }
    if (d2 <= d3) {
      return vec2(d2, (len1 + t2 * len2) / totalLen);
    }
    return vec2(d3, (len1 + len2 + t3 * len3) / totalLen);
  }

  // Traveling reveal/erase window along the arc-length parameter — draws
  // once (t: 0 -> 1), holds, then erases in the same direction.
  float rectMask(float t, float startTime, float drawDur, float hold, float eraseDur, float localTime) {
    float lt = localTime - startTime;
    float drawFront = clamp(lt / drawDur, 0.0, 1.0);
    float eraseFront = clamp((lt - drawDur - hold) / eraseDur, 0.0, 1.0);
    float eps = 0.01;
    float reveal = 1.0 - smoothstep(drawFront - eps, drawFront + eps, t);
    float erase = smoothstep(eraseFront - eps, eraseFront + eps, t);
    return reveal * erase;
  }

  void main() {
    // Proportional aspect-ratio cover projection
    vec2 uv = v_uv;
    if (u_resolution.x > 0.0 && u_resolution.y > 0.0 && u_image_resolution.x > 0.0 && u_image_resolution.y > 0.0) {
      float screenAspect = u_resolution.x / u_resolution.y;
      float imageAspect = u_image_resolution.x / u_image_resolution.y;
      if (screenAspect > imageAspect) {
        float scale = imageAspect / screenAspect;
        uv.y = (v_uv.y - 0.5) * scale + 0.5;
      } else {
        float scale = screenAspect / imageAspect;
        uv.x = (v_uv.x - 0.5) * scale + 0.5;
      }
    }

    float depth = texture2D(u_depth, uv).r;
    float parallax = (depth - 0.5) * 2.0 * u_strength;
    vec2 colorUv = clamp(uv + u_pointer * parallax, 0.0, 1.0);
    vec4 color = texture2D(u_color, colorUv);

    const float CYCLE = 3.6;
    const float DRAW = 1.3;
    const float HOLD = 0.25;
    const float ERASE = 1.3;

    float cycleT = mod(u_time, CYCLE);
    vec2 p = v_uv * u_resolution;

    // v_uv.y = 0 is the canvas BOTTOM and 1 is the TOP (WebGL convention).
    // Rectangle inset just inside the photo's dark margins: ceiling above the
    // lamp, the blurred foreground silhouettes left/right, the floor below.
    // Right branch: top-centre -> top-right -> bottom-right -> bottom-centre.
    vec2 rightField = rectField(
      p, u_resolution,
      vec2(0.5, 0.94), vec2(0.965, 0.94), vec2(0.965, 0.05), vec2(0.5, 0.05),
      12.9
    );
    // Left branch mirrors it: top-centre -> top-left -> bottom-left -> bottom-centre.
    vec2 leftField = rectField(
      p, u_resolution,
      vec2(0.5, 0.94), vec2(0.035, 0.94), vec2(0.035, 0.05), vec2(0.5, 0.05),
      47.3
    );

    float rightMask = rectMask(rightField.y, 0.0, DRAW, HOLD, ERASE, cycleT);
    float leftMask = rectMask(leftField.y, 0.04, DRAW, HOLD, ERASE, cycleT);

    // Thin line: a tight bright core with only a narrow soft glow — not a
    // wide wash across the frame.
    float rightCore = 1.0 - smoothstep(0.0, 1.1, rightField.x);
    float rightGlow = (1.0 - smoothstep(0.0, 5.0, rightField.x)) * 0.18;
    float leftCore = 1.0 - smoothstep(0.0, 1.1, leftField.x);
    float leftGlow = (1.0 - smoothstep(0.0, 5.0, leftField.x)) * 0.18;

    float scanLine = clamp(
      max((rightCore + rightGlow) * rightMask, (leftCore + leftGlow) * leftMask),
      0.0,
      1.0
    );

    vec3 scanColor = vec3(47.0, 237.0, 164.0) / 255.0;
    color.rgb = 1.0 - (1.0 - color.rgb) * (1.0 - scanColor * scanLine);
    gl_FragColor = color;
  }
`;

interface HeroDepthScanProps {
  imageSrc: string;
  depthMapSrc: string;
  alt: string;
  className?: string;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load ${src}`));
    image.src = src;
  });
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Unable to create WebGL shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'Unknown shader error';
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createTexture(gl: WebGLRenderingContext, image: HTMLImageElement, unit: number) {
  const texture = gl.createTexture();
  if (!texture) throw new Error('Unable to create WebGL texture');
  gl.activeTexture(unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  return texture;
}

export const HeroDepthScan: React.FC<HeroDepthScanProps> = ({
  imageSrc,
  depthMapSrc,
  alt,
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [useFallback, setUseFallback] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: true,
      powerPreference: 'high-performance',
    });

    if (!gl || reduceMotion) {
      setUseFallback(true);
      return;
    }

    let animationFrame = 0;
    let disposed = false;
    let resizeObserver: ResizeObserver | undefined;
    let releaseResources: (() => void) | undefined;
    const pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };

    const handlePointerMove = (event: PointerEvent) => {
      const bounds = canvas.getBoundingClientRect();
      pointer.targetX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2.0;
      pointer.targetY = ((event.clientY - bounds.top) / bounds.height - 0.5) * -2.0;
    };
    const resetPointer = () => {
      pointer.targetX = 0;
      pointer.targetY = 0;
    };

    Promise.all([loadImage(imageSrc), loadImage(depthMapSrc)])
      .then(([colorImage, depthImage]) => {
        if (disposed) return;

        const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        const program = gl.createProgram();
        if (!program) throw new Error('Unable to create WebGL program');
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
          throw new Error(gl.getProgramInfoLog(program) || 'Unable to link WebGL program');
        }
        gl.useProgram(program);

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
          gl.STATIC_DRAW,
        );
        const position = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

        const colorTexture = createTexture(gl, colorImage, gl.TEXTURE0);
        const depthTexture = createTexture(gl, depthImage, gl.TEXTURE1);
        gl.uniform1i(gl.getUniformLocation(program, 'u_color'), 0);
        gl.uniform1i(gl.getUniformLocation(program, 'u_depth'), 1);
        gl.uniform1f(gl.getUniformLocation(program, 'u_strength'), 0.012);

        const resolutionLocation = gl.getUniformLocation(program, 'u_resolution');
        const imageResolutionLocation = gl.getUniformLocation(program, 'u_image_resolution');
        gl.uniform2f(
          imageResolutionLocation,
          colorImage.naturalWidth || 1916,
          colorImage.naturalHeight || 821
        );

        const pointerLocation = gl.getUniformLocation(program, 'u_pointer');
        const timeLocation = gl.getUniformLocation(program, 'u_time');

        const resize = () => {
          const bounds = canvas.getBoundingClientRect();
          const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
          const width = Math.max(1, Math.round(bounds.width * pixelRatio));
          const height = Math.max(1, Math.round(bounds.height * pixelRatio));
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
          gl.viewport(0, 0, width, height);
          gl.uniform2f(resolutionLocation, width, height);
        };

        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(canvas);
        resize();

        let startTime: number | undefined;
        const render = (timestamp: number) => {
          startTime ??= timestamp;
          pointer.x += (pointer.targetX - pointer.x) * 0.075;
          pointer.y += (pointer.targetY - pointer.y) * 0.075;
          gl.uniform2f(pointerLocation, pointer.x, pointer.y);
          gl.uniform1f(timeLocation, (timestamp - startTime) / 1000.0);
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          animationFrame = window.requestAnimationFrame(render);
        };

        canvas.addEventListener('pointermove', handlePointerMove);
        canvas.addEventListener('pointerleave', resetPointer);
        animationFrame = window.requestAnimationFrame(render);

        releaseResources = () => {
          gl.deleteTexture(colorTexture);
          gl.deleteTexture(depthTexture);
          gl.deleteBuffer(buffer);
          gl.deleteProgram(program);
          gl.deleteShader(vertexShader);
          gl.deleteShader(fragmentShader);
        };
      })
      .catch((error) => {
        console.error('Hero depth scan failed:', error);
        if (!disposed) setUseFallback(true);
      });

    return () => {
      disposed = true;
      window.cancelAnimationFrame(animationFrame);
      resizeObserver?.disconnect();
      canvas.removeEventListener('pointermove', handlePointerMove);
      canvas.removeEventListener('pointerleave', resetPointer);
      releaseResources?.();
    };
  }, [depthMapSrc, imageSrc]);

  if (useFallback) {
    return <img src={imageSrc} alt={alt} className={`object-cover ${className || ''}`} draggable={false} />;
  }

  return <canvas ref={canvasRef} aria-label={alt} className={className} />;
};
