/**
 * The only module that touches three.js / React Three Fiber. It is loaded with React.lazy from
 * TermStructurePage (the default export, the ribbon) and from the Evaluation view's error surface
 * (`ErrorSurfaceCanvas`), so the 3D code is a separate chunk that is downloaded only when a 3D view
 * is actually shown (and not at all when WebGL is unavailable).
 *
 * It decides nothing: positions, colours and triangles come precomputed from lib/termStructure.ts
 * and lib/errorSurface.ts (where they are unit-tested); this file only uploads them to the GPU.
 */
import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { BufferAttribute, BufferGeometry, Color, DoubleSide, SRGBColorSpace } from 'three'
import type { SurfaceLayout } from '../lib/errorSurface'
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

const TOKENS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8', '--ink', '--muted']

/**
 * The page's colour tokens as the stylesheet resolves them for the current theme. WebGL cannot read
 * CSS variables, so the bars take the same `--series-N` values the 2D charts use, read here and read
 * again when the theme toggle changes `<html data-theme>`.
 */
function useColourTokens(): Record<string, string> {
  const read = () => {
    const style = getComputedStyle(document.documentElement)
    return Object.fromEntries(TOKENS.map((name) => [name, style.getPropertyValue(name).trim()]))
  }
  const [tokens, setTokens] = useState(read)
  useEffect(() => {
    const observer = new MutationObserver(() => setTokens(read()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])
  return tokens
}

/**
 * The error surface: one box per (model, horizon, regime) cell of `layout`, on a translucent plane at
 * ratio 1 (the random walk), with the axis labels as DOM text (drei `Html`: no font file, nothing from
 * another origin). `onHover(i, true)` / `onHover(i, false)` report the bar under the pointer.
 */
export function ErrorSurfaceCanvas({ layout, onHover }: { layout: SurfaceLayout; onHover: (bar: number, on: boolean) => void }) {
  const tokens = useColourTokens()
  const colour = (slot: number | null) => (slot === null ? tokens['--ink'] : tokens[`--series-${slot}`]) || '#888888'
  const margin = 0.6
  // The DOM labels mount once the canvas exists: mounted in the canvas's first commit, the first
  // label's content never appeared (drei 10.7 / fiber 9.8 / React 19.3, seen in Chromium).
  const [created, setCreated] = useState(false)
  return (
    <div className="surface-stage">
      {/* frameloop="demand": render only when the camera moves or a bar changes, so an idle page costs no GPU. */}
      <Canvas camera={{ position: [2, 5, 8.5], fov: 42 }} dpr={[1, 2]} frameloop="demand" onCreated={() => setCreated(true)}>
        <ambientLight intensity={0.75} />
        <directionalLight position={[6, 10, 8]} intensity={1.8} />
        {layout.bars.map((b, i) => (
          <mesh
            key={`${b.modelId}-${b.horizon}-${b.regime.scheme}-${b.regime.regime}`}
            position={b.position}
            onPointerOver={(e) => {
              e.stopPropagation()
              onHover(i, true)
            }}
            onPointerOut={() => onHover(i, false)}
          >
            <boxGeometry args={b.size} />
            <meshStandardMaterial color={colour(b.slot)} roughness={0.75} metalness={0} />
          </mesh>
        ))}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[layout.plane.width + margin, layout.plane.depth + margin]} />
          <meshBasicMaterial color={tokens['--muted'] || '#888888'} transparent opacity={0.22} side={DoubleSide} depthWrite={false} />
        </mesh>
        {created && layout.horizonLabels.map((l, i) => (
          <Html key={`h-${i}`} position={l.position} center className="surface-label head">
            {l.text}
          </Html>
        ))}
        {/* Regime names end at the left edge; model names slant down from the foot of their column. */}
        {created && layout.regimeLabels.map((l, i) => (
          <Html key={`r-${i}`} position={l.position} className="surface-anchor">
            <span className="surface-label end">{l.text}</span>
          </Html>
        ))}
        {created && layout.modelLabels.map((l, i) => (
          <Html key={`m-${i}`} position={l.position} className="surface-anchor">
            <span className="surface-label model slant">{l.text}</span>
          </Html>
        ))}
        <OrbitControls target={[0, -0.5, 0]} minDistance={5} maxDistance={26} />
      </Canvas>
    </div>
  )
}
