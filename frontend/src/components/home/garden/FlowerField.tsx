'use client';

import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import type { FlowerInstance } from './gardenField';
import { getGardenTextures, FLOWER_STATE_COUNT } from './gardenTextures';

// Shared low-frequency wind — hundreds of small flowers swaying together
// as one field, not per-flower theatrics. Amplitude is intentionally
// small (this is a meadow, not individual dramatic plants).
const WIND_FREQ = 0.3;
const WIND_BASE_AMP = 0.035;
const WIND_DETAIL_FREQ = 0.85;
const WIND_DETAIL_AMP = 0.2;

interface FlowerFieldProps {
  instances: FlowerInstance[];
  active: boolean;
}

// A meadow flower is two small real parts: a thin, dark, low-cost stem
// (real 3D geometry, rooted in the terrain, not a billboard) and a small
// bright head billboard at its tip. At carpet density (hundreds of
// instances) both stay tiny on screen — this is deliberately NOT the
// macro/giant-plant treatment from the previous pass.
export function FlowerField({ instances, active }: FlowerFieldProps) {
  const textures = useMemo(() => getGardenTextures(), []);
  const byVariant = useMemo(() => {
    const groups: FlowerInstance[][] = Array.from({ length: FLOWER_STATE_COUNT }, () => []);
    instances.forEach((f) => groups[f.variant].push(f));
    return groups;
  }, [instances]);

  return (
    <>
      <StemBatch instances={instances} active={active} />
      {byVariant.map((group, i) =>
        group.length > 0 ? <HeadBatch key={i} group={group} texture={textures.flowers[i]} active={active} /> : null
      )}
    </>
  );
}

function useStemGeometry() {
  return useMemo(() => {
    // Very thin, very cheap (3 radial segments — at meadow scale nobody
    // can tell it's not round), pivoted at its base.
    const geo = new THREE.CylinderGeometry(0.008, 0.016, 1, 3, 1, true);
    geo.translate(0, 0.5, 0);
    return geo;
  }, []);
}

function StemBatch({ instances, active }: { instances: FlowerInstance[]; active: boolean }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const geometry = useStemGeometry();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const elapsedRef = useRef(0);

  // Dark gray/graphite, not bright white — per the monochrome adaptation,
  // flower heads stay the brightest element, foliage recedes.
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#3a423f',
        transparent: true,
        opacity: 0.7,
        toneMapped: false,
        fog: true,
      }),
    []
  );

  const applyPose = (windAngle: number) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    instances.forEach((f, i) => {
      const sway = active ? windAngle * f.windAmp + Math.sin(elapsedRef.current * WIND_DETAIL_FREQ + f.windPhase) * WIND_DETAIL_AMP * f.windAmp : 0;
      dummy.position.set(f.x, f.y, f.z);
      dummy.rotation.set(f.tilt, f.rotationY, sway);
      const thickness = 0.6 + f.scale * 0.5;
      dummy.scale.set(thickness, f.stemHeight, thickness);
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
    applyPose(Math.sin(elapsedRef.current * WIND_FREQ) * WIND_BASE_AMP);
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, instances.length]} frustumCulled={false} />;
}

function useHeadGeometry() {
  return useMemo(() => {
    const geo = new THREE.PlaneGeometry(0.72, 0.72);
    geo.translate(0, 0.36, 0);
    // material.vertexColors=true (see HeadBatch) independently requires a
    // per-VERTEX `color` attribute on the geometry, separate from the
    // per-INSTANCE color set via setColorAt below — without one, three.js
    // multiplies by an unbound (black) attribute and every instance
    // renders invisible, regardless of instance color. All-white here so
    // it's a no-op; setColorAt is the only thing that actually modulates
    // brightness.
    const vertexCount = geo.attributes.position.count;
    geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3).fill(1), 3));
    return geo;
  }, []);
}

// Heads billboard to face the camera every frame (compose position +
// camera quaternion + a per-instance roll for petal-pattern variety),
// instead of the earlier fixed-rotation plane. At carpet density the
// near view cone reliably contains a flower nearly along the camera's
// look direction; a fixed-rotation plane viewed edge-on there stretches
// into a huge distorted shape under perspective. True billboarding
// removes that failure mode regardless of distance/angle — standard for
// flat sprite impostors, and doesn't affect the real 3D (non-billboard)
// stems in StemBatch above.
function HeadBatch({ group, texture, active }: { group: FlowerInstance[]; texture: THREE.Texture; active: boolean }) {
  const { camera } = useThree();
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const geometry = useHeadGeometry();
  const rootDummy = useMemo(() => new THREE.Object3D(), []);
  const offsetDummy = useMemo(() => new THREE.Object3D(), []);
  const tmpMatrix = useMemo(() => new THREE.Matrix4(), []);
  const posVec = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  const rollQuat = useMemo(() => new THREE.Quaternion(), []);
  const scaleVec = useMemo(() => new THREE.Vector3(), []);
  const finalMatrix = useMemo(() => new THREE.Matrix4(), []);
  const zAxis = useMemo(() => new THREE.Vector3(0, 0, 1), []);
  const elapsedRef = useRef(0);

  const colorTmp = useMemo(() => new THREE.Color(), []);

  // vertexColors lets a shared material still carry a per-instance depth
  // brightness (multiplied into the texture sample) — near flowers stay
  // near-full brightness, far ones dim toward the fog color instead of
  // contributing to a hard, equally-bright horizon line.
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        alphaTest: 0.1,
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
      const sway = active ? windAngle * f.windAmp + Math.sin(elapsedRef.current * WIND_DETAIL_FREQ + f.windPhase) * WIND_DETAIL_AMP * f.windAmp : 0;
      rootDummy.position.set(f.x, f.y, f.z);
      rootDummy.rotation.set(f.tilt, 0, sway);
      rootDummy.updateMatrix();

      offsetDummy.position.set(0, f.stemHeight, 0);
      offsetDummy.updateMatrix();

      tmpMatrix.copy(rootDummy.matrix).multiply(offsetDummy.matrix);
      posVec.setFromMatrixPosition(tmpMatrix);

      quat.copy(camera.quaternion);
      rollQuat.setFromAxisAngle(zAxis, f.rotationY);
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
    applyPose(active ? Math.sin(elapsedRef.current * WIND_FREQ) * WIND_BASE_AMP : 0);
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, group.length]} frustumCulled={false} />;
}
