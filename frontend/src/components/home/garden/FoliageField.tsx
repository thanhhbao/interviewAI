'use client';

import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import type { FoliageInstance } from './gardenField';
import { getGardenTextures, FOLIAGE_STATE_COUNT } from './gardenTextures';

// Same shared low-frequency wind as FlowerField, smaller amplitude still —
// foliage is mass, not motion; it should barely register as moving.
const WIND_FREQ = 0.3;
const WIND_AMP = 0.02;

interface FoliageFieldProps {
  instances: FoliageInstance[];
  active: boolean;
}

// Cheap dark vegetation mass, one InstancedMesh per texture variant (same
// batching pattern as FlowerField's HeadBatch). No stem, no scene-graph
// offset — just billboards rooted directly at terrain height, filling the
// black gaps between flower heads.
export function FoliageField({ instances, active }: FoliageFieldProps) {
  const textures = useMemo(() => getGardenTextures(), []);
  const byVariant = useMemo(() => {
    const groups: FoliageInstance[][] = Array.from({ length: FOLIAGE_STATE_COUNT }, () => []);
    instances.forEach((f) => groups[f.variant].push(f));
    return groups;
  }, [instances]);

  return (
    <>
      {byVariant.map((group, i) =>
        group.length > 0 ? <FoliageBatch key={i} group={group} texture={textures.foliage[i]} active={active} /> : null
      )}
    </>
  );
}

function useFoliageGeometry() {
  return useMemo(() => {
    const geo = new THREE.PlaneGeometry(0.85, 0.62);
    geo.translate(0, 0.31, 0);
    // See the matching comment in FlowerField.tsx's useHeadGeometry — an
    // all-white per-vertex color attribute so material.vertexColors=true
    // doesn't multiply everything to black; setColorAt below is what
    // actually applies per-instance brightness.
    const vertexCount = geo.attributes.position.count;
    geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3).fill(1), 3));
    return geo;
  }, []);
}

// Same camera-facing billboard treatment as FlowerField's HeadBatch, for
// the same reason — a fixed-rotation plane viewed edge-on at close range
// stretches into a huge distorted shape, which at carpet density is a
// near-certainty rather than an edge case.
function FoliageBatch({ group, texture, active }: { group: FoliageInstance[]; texture: THREE.Texture; active: boolean }) {
  const { camera } = useThree();
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const geometry = useFoliageGeometry();
  const posVec = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  const rollQuat = useMemo(() => new THREE.Quaternion(), []);
  const scaleVec = useMemo(() => new THREE.Vector3(), []);
  const finalMatrix = useMemo(() => new THREE.Matrix4(), []);
  const zAxis = useMemo(() => new THREE.Vector3(0, 0, 1), []);
  const colorTmp = useMemo(() => new THREE.Color(), []);
  const elapsedRef = useRef(0);

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        alphaTest: 0.08,
        depthWrite: false,
        toneMapped: false,
        fog: true,
        vertexColors: true,
      }),
    [texture]
  );

  const applyPose = (windAngle: number) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    group.forEach((f, i) => {
      posVec.set(f.x, f.y, f.z);
      quat.copy(camera.quaternion);
      rollQuat.setFromAxisAngle(zAxis, f.rotationY + (active ? windAngle : 0));
      quat.multiply(rollQuat);
      scaleVec.setScalar(f.scale);
      finalMatrix.compose(posVec, quat, scaleVec);
      mesh.setMatrixAt(i, finalMatrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  };

  React.useEffect(() => {
    applyPose(0);
    const mesh = meshRef.current;
    if (mesh) {
      group.forEach((f, i) => {
        mesh.setColorAt(i, colorTmp.setScalar(f.brightness));
      });
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [group]);

  useFrame((_, delta) => {
    elapsedRef.current += active ? delta : 0;
    applyPose(active ? Math.sin(elapsedRef.current * WIND_FREQ) * WIND_AMP : 0);
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, group.length]} frustumCulled={false} />;
}
