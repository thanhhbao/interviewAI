'use client';

import React, { useMemo } from 'react';
import * as THREE from 'three';
import { CORRIDOR_FAR_Z, CORRIDOR_NEAR_Z, terrainHeight } from './gardenField';

// Implied dark ground — not realistic soil, just enough of a plane that
// the flowers read as rooted in a world instead of floating in empty
// black space. A soft procedural vignette/noise texture stands in for
// grazing-light variation.
function buildGroundTexture(): HTMLCanvasElement {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#040503';
  ctx.fillRect(0, 0, size, size);

  let seed = 8821;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed % 1000) / 1000;
  };
  for (let i = 0; i < 40; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = 40 + rand() * 120;
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    const alpha = 0.03 + rand() * 0.04;
    grad.addColorStop(0, `rgba(255,255,255,${alpha})`);
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas;
}

const WIDTH = 80;

interface TerrainProps {
  // Temporary, dev-only visualization for validating the valley shape in
  // isolation from flower density — never shipped/exposed outside the
  // ?terrainDebug=1 dev-preview toggle (see DandelionGarden.tsx).
  debugWireframe?: boolean;
}

export function Terrain({ debugWireframe }: TerrainProps) {
  const texture = useMemo(() => {
    const tex = new THREE.CanvasTexture(buildGroundTexture());
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(6, 10);
    return tex;
  }, []);

  const length = CORRIDOR_NEAR_Z - CORRIDOR_FAR_Z + 60;
  const centerZ = (CORRIDOR_NEAR_Z + CORRIDOR_FAR_Z) / 2;

  // Gentle rolling ground — same `terrainHeight()` every plant's root Y
  // samples (see gardenField.ts) — so the ground the plants stand on and
  // the ground the plants ARE rooted at are the same surface, not two
  // independent approximations of "flat".
  const geometry = useMemo(() => {
    const segX = 40;
    const segZ = 64;
    const geo = new THREE.PlaneGeometry(WIDTH, length, segX, segZ);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const lx = pos.getX(i);
      const ly = pos.getY(i); // local Y, becomes world Z after the -90deg X rotation below
      const worldZ = ly + centerZ;
      pos.setZ(i, terrainHeight(lx, worldZ));
    }
    geo.computeVertexNormals();
    return geo;
  }, [length, centerZ]);

  return (
    <mesh position={[0, 0, centerZ]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow={false} geometry={geometry}>
      {debugWireframe ? (
        <meshBasicMaterial color="#ffffff" wireframe fog toneMapped={false} />
      ) : (
        <meshBasicMaterial map={texture} color="#050604" fog toneMapped={false} />
      )}
    </mesh>
  );
}
