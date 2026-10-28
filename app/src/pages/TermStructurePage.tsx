import { lazy, Suspense, useMemo, useState } from 'react'
import fixture from '../data/term-structure.synthetic.json'
import { Heatmap } from '../lib/charts'
import { fmt } from '../lib/format'
import {
  BACKWARDATION_RGB,
  CONTANGO_RGB,
  buildRibbonGeometry,
  parseTermStructure,
  regimeCounts,
  regimeSummary,
  rgbToHex,
  type TermStructureData,
} from '../lib/termStructure'
import { detectWebGL } from '../lib/webgl'
import { RenderBoundary } from './RenderBoundary'

// Lazy: three.js + R3F + drei stay out of the main bundle until this view needs them.
const RibbonCanvas = lazy(() => import('./RibbonCanvas'))

/** Same numbers as the 3D ribbon, as a flat tenor x date heatmap, for browsers without WebGL. */
function RibbonFallback({ data, reason }: { data: TermStructureData; reason: string }) {
  const rows = data.tenors.map((_, t) => data.levels.map((row) => row[t]))
  return (
    <div className="card">
      <p role="status" className="sub">
        The 3D view is unavailable ({reason}). Here is the same data as a flat heatmap: rows are tenors ({data.tenors.join(', ')}), columns
        run from the oldest date on the left to the newest on the right, and a brighter cell is a higher implied volatility.
      </p>
      <Heatmap rows={rows} yLabels={data.tenors} />
    </div>
  )
}

export default function TermStructurePage() {
  const data = useMemo(() => parseTermStructure(fixture), [])
  const geometry = useMemo(() => buildRibbonGeometry(data), [data])
  const [webgl] = useState(() => detectWebGL())
  const counts = regimeCounts(data)
  const { min, max } = geometry.levelRange

  return (
    <section className="term-structure">
      <h2>VIX term-structure ribbon: placeholder</h2>
      <div>
        {data.synthetic && <span className="badge synthetic">synthetic data</span>}
        <span className="badge">{data.dates.length} dates</span>
        <span className="badge">{data.tenors.length} tenors</span>
      </div>
      <p className="sub">
        {data.synthetic
          ? `This ribbon is drawn from a generated fixture (seed ${data.generator?.seed ?? 'n/a'}), not from CBOE data. `
          : ''}
        It stands in for the real VIX9D to VIX1Y history that milestone 1 will load. Height is implied volatility, left to right is maturity, front to
        back is time (newest in front).
      </p>

      <div className="grid">
        <div className="stat">
          <div className="k">Contango dates</div>
          <div className="v">{counts.contango}</div>
        </div>
        <div className="stat">
          <div className="k">Backwardation dates</div>
          <div className={`v ${counts.backwardation > 0 ? 'down' : ''}`}>{counts.backwardation}</div>
        </div>
        <div className="stat">
          <div className="k">Level range (vol pts)</div>
          <div className="v">
            {fmt(min)} to {fmt(max)}
          </div>
        </div>
      </div>

      <h2>Ribbon</h2>
      <figure className="ribbon" aria-label={`Implied volatility by tenor and date. ${regimeSummary(data)}.`}>
        {webgl ? (
          <RenderBoundary fallback={<RibbonFallback data={data} reason="the 3D renderer failed to start" />}>
            <Suspense fallback={<p className="sub">Loading the 3D renderer...</p>}>
              <RibbonCanvas geometry={geometry} />
            </Suspense>
          </RenderBoundary>
        ) : (
          <RibbonFallback data={data} reason="this browser has no WebGL" />
        )}
        <figcaption className="legend">
          <span>
            <i className="swatch" style={{ background: rgbToHex(CONTANGO_RGB) }} /> contango: long end above short end
          </span>
          <span>
            <i className="swatch" style={{ background: rgbToHex(BACKWARDATION_RGB) }} /> backwardation: curve inverted
          </span>
          <span className="sub">Drag to rotate, scroll or pinch to zoom, right-drag to pan.</span>
        </figcaption>
      </figure>
      <p className="sub">{regimeSummary(data)}.</p>
    </section>
  )
}
