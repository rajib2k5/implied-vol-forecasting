/**
 * The only module that touches three.js / React Three Fiber. It is loaded with React.lazy from
 * TermStructurePage, so the 3D code is a separate chunk that is downloaded only when the
 * 3D view is actually shown (and not at all when WebGL is unavailable).
 *
 * It decides nothing: positions, colours and triangles come precomputed from
 * lib/termStructure.ts (where they are unit-tested); this file only uploads them to the GPU.
 */
import { useEffect, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { BufferAttribute, BufferGeometry, Color, DoubleSide, SRGBColorSpace } from 'three'
import type { RibbonGeometry } from '../lib/termStructure'

/**
 * The palette in lib/termStructure.ts is in ordinary sRGB (it is also used for the DOM legend),
 * but three.js treats vertex colours as linear and encodes the result to sRGB on output, which
 * would wash the colours out. Convert once here so the mesh matches the legend swatches.
 */
function toLinear(srgb: Float32Array): Float32Array {
  const out = new Float32Array(srgb.length)
  const c = new Color()
  for (let i = 0; i < srgb.length; i += 3) {
    c.setRGB(srgb[i], srgb[i + 1], srgb[i + 2], SRGBColorSpace)
    out[i] = c.r
    out[i + 1] = c.g
    out[i + 2] = c.b
  }
  return out
}

function RibbonMesh({ geometry }: { geometry: RibbonGeometry }) {
  const bufferGeometry = useMemo(() => {
    const g = new BufferGeometry()
    g.setAttribute('position', new BufferAttribute(geometry.positions, 3))
    g.setAttribute('color', new BufferAttribute(toLinear(geometry.colors), 3))
    g.setIndex(new BufferAttribute(geometry.indices, 1))
    g.computeVertexNormals() // lighting needs normals; they follow from positions + indices
    return g
  }, [geometry])
  // We created the geometry, so we free its GPU buffers (dispose={null} stops R3F doing it twice).
  useEffect(() => () => bufferGeometry.dispose(), [bufferGeometry])
  return (
    <>
      <mesh geometry={bufferGeometry} dispose={null}>
        <meshStandardMaterial vertexColors side={DoubleSide} roughness={0.85} metalness={0} />
      </mesh>
      <mesh geometry={bufferGeometry} dispose={null}>
        <meshBasicMaterial wireframe color="#1b1e24" transparent opacity={0.22} />
      </mesh>
    </>
  )
}

export default function RibbonCanvas({ geometry }: { geometry: RibbonGeometry }) {
  return (
    <div className="ribbon-stage">
      {/* frameloop="demand": render only when the camera moves, so an idle page costs no GPU. */}
      <Canvas camera={{ position: [7.5, 5.5, 9], fov: 42 }} dpr={[1, 2]} frameloop="demand">
        <ambientLight intensity={0.6} />
        <directionalLight position={[6, 10, 8]} intensity={2} />
        <RibbonMesh geometry={geometry} />
        <gridHelper args={[10, 10, '#888888', '#555555']} position={[0, -0.02, 0]} />
        <OrbitControls target={[0, 1, 0]} minDistance={6} maxDistance={24} maxPolarAngle={Math.PI * 0.49} />
      </Canvas>
    </div>
  )
}
