'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, ContactShadows, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { AvatarState } from '@/lib/types/interview';

interface AvatarMeshProps {
  state: AvatarState;
  isCompact?: boolean;
  enableMouseTracking?: boolean;
}

/**
 * Procedural Sci-Fi Robot mirroring the reference image:
 * White glossy chassis, dark visor, glowing cyan-blue dual ring eyes,
 * circular headphone ear cups, chest reactor, and real-time cursor hover tracking.
 */
function SciFiRobotAvatar({
  state,
  isCompact = false,
  enableMouseTracking = true,
}: AvatarMeshProps) {
  const rootGroupRef = useRef<THREE.Group>(null);
  const headGroupRef = useRef<THREE.Group>(null);
  const leftEyeRef = useRef<THREE.Group>(null);
  const rightEyeRef = useRef<THREE.Group>(null);
  const mouthRef = useRef<THREE.Mesh>(null);
  const chestCoreRef = useRef<THREE.Mesh>(null);
  const orbitalRingRef = useRef<THREE.Mesh>(null);

  // Materials
  const whiteArmorMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#f8fafc'),
        roughness: 0.18,
        metalness: 0.1,
        clearcoat: 0.9,
        clearcoatRoughness: 0.15,
      }),
    []
  );

  const darkVisorMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#0a0c16'),
        roughness: 0.25,
        metalness: 0.8,
        clearcoat: 0.8,
      }),
    []
  );

  const cyanGlowMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#FFFFFF'),
        emissive: new THREE.Color('#FFFFFF'),
        emissiveIntensity: 2.2,
        roughness: 0.2,
      }),
    []
  );

  const deepBlueGlowMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#FFFFFF'),
        emissive: new THREE.Color('#E7E8E9'),
        emissiveIntensity: 1.8,
        roughness: 0.3,
      }),
    []
  );

  const metallicJointMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#1e2235'),
        roughness: 0.4,
        metalness: 0.8,
      }),
    []
  );

  // Animation & Mouse Tracking Loop
  useFrame((stateContext) => {
    const t = stateContext.clock.getElapsedTime();
    const pointer = stateContext.pointer; // normalized [-1, 1]

    if (!rootGroupRef.current || !headGroupRef.current) return;

    // 1. BASE BREATHING & FLOATING
    const breathing = Math.sin(t * 1.8) * 0.035;
    rootGroupRef.current.position.y = breathing;

    // 2. MOUSE / CURSOR HOVER TRACKING
    if (enableMouseTracking) {
      const targetHeadRotY = pointer.x * 0.72; // look left/right
      const targetHeadRotX = -pointer.y * 0.48; // look up/down
      const targetHeadRotZ = -pointer.x * 0.08; // subtle natural tilt

      headGroupRef.current.rotation.y = THREE.MathUtils.lerp(
        headGroupRef.current.rotation.y,
        targetHeadRotY,
        0.085
      );
      headGroupRef.current.rotation.x = THREE.MathUtils.lerp(
        headGroupRef.current.rotation.x,
        targetHeadRotX,
        0.085
      );
      headGroupRef.current.rotation.z = THREE.MathUtils.lerp(
        headGroupRef.current.rotation.z,
        targetHeadRotZ,
        0.085
      );

      // Eye pupil tracking
      const pupilShiftX = pointer.x * 0.025;
      const pupilShiftY = pointer.y * 0.02;

      if (leftEyeRef.current) {
        leftEyeRef.current.position.x = -0.2 + pupilShiftX;
        leftEyeRef.current.position.y = 0.04 + pupilShiftY;
      }
      if (rightEyeRef.current) {
        rightEyeRef.current.position.x = 0.2 + pupilShiftX;
        rightEyeRef.current.position.y = 0.04 + pupilShiftY;
      }
    }

    // 3. CONVERSATIONAL STATES DYNAMICS
    if (state === 'speaking') {
      if (mouthRef.current) {
        const mouthScale = 1 + Math.abs(Math.sin(t * 9)) * 1.6;
        mouthRef.current.scale.set(1.1, mouthScale, 1);
      }
      if (chestCoreRef.current) {
        chestCoreRef.current.scale.setScalar(1 + Math.sin(t * 7) * 0.15);
      }
    } else if (state === 'listening') {
      if (mouthRef.current) mouthRef.current.scale.set(1, 0.4, 1);
      if (chestCoreRef.current) {
        chestCoreRef.current.scale.setScalar(1 + Math.sin(t * 2.5) * 0.05);
      }
    } else if (state === 'thinking') {
      if (orbitalRingRef.current) {
        orbitalRingRef.current.rotation.x = t * 1.5;
        orbitalRingRef.current.rotation.y = t * 2.2;
      }
      if (chestCoreRef.current) {
        chestCoreRef.current.scale.setScalar(1.1 + Math.sin(t * 5) * 0.2);
      }
    } else {
      if (mouthRef.current) mouthRef.current.scale.set(1, 0.6, 1);
    }
  });

  return (
    <group ref={rootGroupRef} scale={isCompact ? 0.72 : 0.95} position={[0, -0.32, 0]}>
      {/* ROBOT UPPER TORSO & SHOULDERS */}
      <group position={[0, -0.75, 0]}>
        {/* Chest main armor */}
        <mesh material={whiteArmorMaterial} position={[0, 0, 0]}>
          <boxGeometry args={[1.3, 0.85, 0.65]} />
        </mesh>

        {/* Shoulder caps */}
        <mesh material={whiteArmorMaterial} position={[-0.8, 0.2, 0]}>
          <sphereGeometry args={[0.26, 24, 24]} />
        </mesh>
        <mesh material={whiteArmorMaterial} position={[0.8, 0.2, 0]}>
          <sphereGeometry args={[0.26, 24, 24]} />
        </mesh>

        {/* Glowing Chest Reactor Core */}
        <mesh ref={chestCoreRef} position={[0, 0.08, 0.34]} rotation={[Math.PI / 2, 0, 0]} material={cyanGlowMaterial}>
          <cylinderGeometry args={[0.12, 0.12, 0.04, 32]} />
        </mesh>
        {/* Chest Core Outer Ring */}
        <mesh position={[0, 0.08, 0.33]} material={deepBlueGlowMaterial}>
          <torusGeometry args={[0.16, 0.02, 16, 32]} />
        </mesh>
      </group>

      {/* ARTICULATED NECK */}
      <group position={[0, -0.22, 0]}>
        <mesh material={metallicJointMaterial}>
          <cylinderGeometry args={[0.22, 0.26, 0.28, 32]} />
        </mesh>
        <mesh material={deepBlueGlowMaterial} position={[0, 0, 0]}>
          <torusGeometry args={[0.25, 0.02, 16, 32]} />
        </mesh>
      </group>

      {/* HEAD GROUP (TRACKS MOUSE / POINTER) */}
      <group ref={headGroupRef} position={[0, 0.38, 0]}>
        {/* Main Sleek White Skull */}
        <mesh material={whiteArmorMaterial} position={[0, 0, -0.04]}>
          <sphereGeometry args={[0.62, 36, 36]} />
        </mesh>

        {/* Dark Visor Faceplate */}
        <mesh material={darkVisorMaterial} position={[0, -0.02, 0.42]}>
          <boxGeometry args={[0.78, 0.48, 0.28]} />
        </mesh>

        {/* LEFT EYE (GLOWING CYAN RING + PUPIL) */}
        <group ref={leftEyeRef} position={[-0.2, 0.04, 0.55]}>
          {/* Outer Ring */}
          <mesh material={cyanGlowMaterial}>
            <torusGeometry args={[0.085, 0.016, 16, 32]} />
          </mesh>
          {/* Inner Glowing Pupil */}
          <mesh material={cyanGlowMaterial} position={[0, 0, -0.02]}>
            <sphereGeometry args={[0.045, 16, 16]} />
          </mesh>
          {/* Eye aperture light */}
          <pointLight color="#FFFFFF" intensity={0.6} distance={0.8} />
        </group>

        {/* RIGHT EYE (GLOWING CYAN RING + PUPIL) */}
        <group ref={rightEyeRef} position={[0.2, 0.04, 0.55]}>
          {/* Outer Ring */}
          <mesh material={cyanGlowMaterial}>
            <torusGeometry args={[0.085, 0.016, 16, 32]} />
          </mesh>
          {/* Inner Glowing Pupil */}
          <mesh material={cyanGlowMaterial} position={[0, 0, -0.02]}>
            <sphereGeometry args={[0.045, 16, 16]} />
          </mesh>
          {/* Eye aperture light */}
          <pointLight color="#FFFFFF" intensity={0.6} distance={0.8} />
        </group>

        {/* Dynamic Voice Aperture / Mouth */}
        <mesh
          ref={mouthRef}
          position={[0, -0.16, 0.54]}
          rotation={[0, 0, Math.PI / 2]}
          material={deepBlueGlowMaterial}
        >
          <capsuleGeometry args={[0.015, 0.16, 8, 16]} />
        </mesh>

        {/* PROMINENT CIRCULAR HEADPHONES / EAR CUPS (AS SEEN IN REFERENCE) */}
        {/* Left Headphone */}
        <group position={[-0.64, 0.05, -0.02]} rotation={[0, 0, Math.PI / 2]}>
          <mesh material={whiteArmorMaterial}>
            <cylinderGeometry args={[0.26, 0.28, 0.14, 32]} />
          </mesh>
          <mesh material={cyanGlowMaterial} position={[0, 0.08, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.19, 0.02, 16, 32]} />
          </mesh>
          <mesh material={darkVisorMaterial} position={[0, 0.07, 0]}>
            <cylinderGeometry args={[0.17, 0.17, 0.02, 32]} />
          </mesh>
        </group>

        {/* Right Headphone */}
        <group position={[0.64, 0.05, -0.02]} rotation={[0, 0, -Math.PI / 2]}>
          <mesh material={whiteArmorMaterial}>
            <cylinderGeometry args={[0.26, 0.28, 0.14, 32]} />
          </mesh>
          <mesh material={cyanGlowMaterial} position={[0, 0.08, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.19, 0.02, 16, 32]} />
          </mesh>
          <mesh material={darkVisorMaterial} position={[0, 0.07, 0]}>
            <cylinderGeometry args={[0.17, 0.17, 0.02, 32]} />
          </mesh>
        </group>

        {/* Head Antenna / Sensor Pin */}
        <mesh position={[0, 0.62, -0.05]} material={metallicJointMaterial}>
          <cylinderGeometry args={[0.03, 0.03, 0.14, 16]} />
        </mesh>
        <mesh position={[0, 0.7, -0.05]} material={cyanGlowMaterial}>
          <sphereGeometry args={[0.04, 16, 16]} />
        </mesh>

        {/* Orbital Thinking Rings (active during processing/thinking) */}
        {state === 'thinking' && (
          <mesh ref={orbitalRingRef} position={[0, 0, 0]} material={cyanGlowMaterial}>
            <torusGeometry args={[0.95, 0.025, 16, 64]} />
          </mesh>
        )}
      </group>

      {/* Atmospheric Shadow Below */}
      <ContactShadows position={[0, -1.35, 0]} opacity={0.6} scale={4} blur={1.6} far={2.5} />
    </group>
  );
}

/**
 * Checks for custom model `/models/avatar.glb`.
 * If not present or errors, seamlessly falls back to the high-detail SciFiRobotAvatar.
 */
function CustomModelLoader({
  state,
  isCompact = false,
  enableMouseTracking = true,
}: AvatarMeshProps) {
  const [hasCustomModel, setHasCustomModel] = useState<boolean | null>(null);

  useEffect(() => {
    // Check if user placed avatar.glb in /models/avatar.glb
    fetch('/models/avatar.glb', { method: 'HEAD' })
      .then((res) => {
        if (res.ok) setHasCustomModel(true);
        else setHasCustomModel(false);
      })
      .catch(() => setHasCustomModel(false));
  }, []);

  if (hasCustomModel) {
    return <LoadedCustomGLB state={state} isCompact={isCompact} enableMouseTracking={enableMouseTracking} />;
  }

  return (
    <SciFiRobotAvatar
      state={state}
      isCompact={isCompact}
      enableMouseTracking={enableMouseTracking}
    />
  );
}

function LoadedCustomGLB({
  state,
  isCompact,
  enableMouseTracking,
}: AvatarMeshProps) {
  const gltf = useGLTF('/models/avatar.glb');
  const groupRef = useRef<THREE.Group>(null);

  useFrame((stateContext) => {
    if (!groupRef.current) return;
    const t = stateContext.clock.getElapsedTime();
    groupRef.current.position.y = Math.sin(t * 1.8) * 0.04;

    if (enableMouseTracking) {
      const pointer = stateContext.pointer;
      groupRef.current.rotation.y = THREE.MathUtils.lerp(
        groupRef.current.rotation.y,
        pointer.x * 0.6,
        0.08
      );
      groupRef.current.rotation.x = THREE.MathUtils.lerp(
        groupRef.current.rotation.x,
        -pointer.y * 0.35,
        0.08
      );
    }
  });

  return (
    <group ref={groupRef} scale={isCompact ? 0.75 : 1} position={[0, -0.4, 0]}>
      <primitive object={gltf.scene} />
    </group>
  );
}

export const InterviewAvatar3D: React.FC<AvatarMeshProps> = ({
  state,
  isCompact = false,
  enableMouseTracking = true,
}) => {
  return (
    <div
      className={`relative w-full h-full select-none ${
        isCompact ? 'min-h-[160px]' : 'min-h-[340px]'
      }`}
    >
      <Canvas
        camera={{ position: [0, 0.15, 3.4], fov: isCompact ? 40 : 36 }}
        gl={{ antialias: true, alpha: true }}
        className="w-full h-full"
      >
        {/* Cosmic Ambient Lighting */}
        <ambientLight intensity={0.8} />

        {/* Key Lighting from top-right */}
        <directionalLight position={[3, 5, 4]} intensity={1.8} color="#ffffff" />

        {/* Electric Blue Rim Light from behind/sides */}
        <pointLight position={[-3, 2, -2]} intensity={2.5} color="#FFFFFF" />
        <pointLight position={[3, 2, -2]} intensity={2.5} color="#E7E8E9" />

        {/* Chest bounce fill light */}
        <pointLight position={[0, -1.5, 2]} intensity={0.7} color="#E7E8E9" />

        <Float speed={1.2} rotationIntensity={0.06} floatIntensity={0.15}>
          <CustomModelLoader
            state={state}
            isCompact={isCompact}
            enableMouseTracking={enableMouseTracking}
          />
        </Float>
      </Canvas>
    </div>
  );
};
