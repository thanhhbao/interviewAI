'use client';

import React, { useMemo } from 'react';
import * as THREE from 'three';

// Same broad-shallow-valley shape validated in the V2 geometry pass
// (center lower, edges rise gradually, no steep banks), reimplemented
// standalone here — V3 is isolated and does not import from the V2
// garden directory.
export function uShapeHeight(x: number, halfWidth: number, rise: number, exponent: number): number {
  const t = Math.min(1, Math.abs(x) / halfWidth);
  return Math.pow(t, exponent) * rise;
}

interface MeadowLayerProps {
  texture: THREE.Texture;
  aspect: number; // texture width / height, so the plane doesn't stretch the bake
  width: number;
  z: number;
  y?: number;
  curveHalfWidth: number;
  curveRise: number;
  curveExponent: number;
  renderOrder: number;
}

// A single dense, pre-baked meadow "sheet" — a curved ground plane so its
// own silhouette also follows the U-shape (reinforcing what's already
// baked into the texture's alpha mask, not the sole source of the curve).
// Fully static: geometry is built once, no useFrame, no per-frame update.
export function MeadowLayer({ texture, aspect, width, z, y = 0, curveHalfWidth, curveRise, curveExponent, renderOrder }: MeadowLayerProps) {
  const depth = width / aspect;

  const geometry = useMemo(() => {
    const segX = 48;
    const segZ = 24;
    const geo = new THREE.PlaneGeometry(width, depth, segX, segZ);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const lx = pos.getX(i);
      const h = uShapeHeight(lx, curveHalfWidth, curveRise, curveExponent);
      pos.setZ(i, h);
    }
    geo.computeVertexNormals();
    return geo;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, depth, curveHalfWidth, curveRise, curveExponent]);

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        alphaTest: 0.02,
        depthWrite: false,
        toneMapped: false,
      }),
    [texture]
  );

  return (
    <mesh
      geometry={geometry}
      material={material}
      position={[0, y, z]}
      rotation={[-Math.PI / 2, 0, 0]}
      renderOrder={renderOrder}
    />
  );
}
