'use client';

import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import type { SeedInstance } from './gardenField';
import { getGardenTextures } from './gardenTextures';

const LOOP_HEIGHT = 5; // world units a seed drifts upward before recycling to the bottom
const WIND_X_BIAS = 0.22; // shared drift direction — same wind that sways the flowers

interface SeedFieldProps {
  instances: SeedInstance[];
  active: boolean;
}

// Detached seeds as a single InstancedMesh (one draw call for the whole
// field, near or far). Each seed loops upward through a fixed window and
// recycles to the bottom — with per-instance phase/speed this reads as
// continuous atmospheric drift, not a visible reset, without needing to
// track individual seed lifetimes in React state.
export function SeedField({ instances, active }: SeedFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const elapsedRef = useRef(0);
  const textures = useMemo(() => getGardenTextures(), []);

  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(1, 1);
    return geo;
  }, []);

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        map: textures.seed,
        transparent: true,
        alphaTest: 0.05,
        depthWrite: false,
        toneMapped: false,
        fog: true,
      }),
    [textures]
  );

  const applyPose = (t: number) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    instances.forEach((s, i) => {
      const localT = active ? ((s.phase + t * s.speed) % (Math.PI * 2)) / (Math.PI * 2) : (s.phase % (Math.PI * 2)) / (Math.PI * 2);
      const y = s.y + localT * LOOP_HEIGHT;
      const wrappedY = ((y + 3) % (LOOP_HEIGHT + 3)) - 3; // recycle band, stays within a bounded window
      // Horizontal drift is a function of the same bounded `localT` the
      // vertical loop uses (not raw elapsed time), so it resets together
      // with the Y recycle instead of accumulating without bound over a
      // long session. `WIND_X_BIAS` gives every seed the same directional
      // lean; `driftX` is the per-instance amount of it.
      const sway = active ? Math.sin(t * s.wobble + s.phase) * 0.35 : 0;
      dummy.position.set(s.x + localT * s.driftX * (1 + WIND_X_BIAS) + sway, wrappedY, s.z);
      dummy.rotation.set(0, 0, active ? Math.sin(t * 0.6 + s.phase) * 0.4 : 0);
      dummy.scale.setScalar(s.scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  };

  React.useEffect(() => {
    applyPose(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instances]);

  useFrame((_, delta) => {
    if (!active) return;
    elapsedRef.current += delta;
    applyPose(elapsedRef.current);
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, instances.length]} frustumCulled={false} />;
}
